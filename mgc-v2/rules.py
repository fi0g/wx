"""Position sizing and backtest acceptance for the MGC V2.0 $10,000 strategy.

The sizing function is the reference implementation. The Pine script and the
strategy document must match these constants.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


POINT_VALUE = 10.0
TICK = 0.1
MIN_STOP = 5.0
MAX_STOP = 12.0
TARGET_RISK_PCT = 0.01
MAX_RISK_PCT = 0.012
MIN_EQUITY = 8500.0
MAINTENANCE_MARGIN = 2085.0
MARGIN_TO_EQUITY = 0.25
MARGIN_PAUSE_EQUITY_SHARE = 0.30
ATR_MULT = 1.2

# Acceptance gates. Live annual targets are documented separately.
MIN_TRADES = 100
MIN_YEARS = 3.0
MIN_CAGR = 0.15
MAX_DRAWDOWN = 0.12
MIN_PROFIT_FACTOR = 1.35
MIN_EXPECTANCY_R = 0.20
WORST_MONTH = -0.06
WORST_YEAR = -0.08
MIN_SIDE_TRADES = 30
MIN_SIDE_PF = 1.10
DROP_WINNERS = 5
MIN_PF_AFTER_DROP = 1.15
MONTE_CARLO_PATHS = 1000
MONTE_CARLO_SEED = 20261009
MONTE_CARLO_DD = 0.18
OOS_DAYS = 365


def round_tick(value: float, tick: float = TICK) -> float:
    """Round a positive price distance to the MGC tick, half-up."""
    steps = math.floor(value / tick + 0.5 + 1e-8)
    return round(steps * tick, 10)


def stop_distance(bar_range: float, atr: float, atr_mult: float = ATR_MULT) -> float | None:
    """Return the stop distance in $/oz, or None when the signal must be skipped.

    `bar_range` is the signal bar's high minus low, plus two ticks. The stop is
    never tightened inside that bar. Distances under the minimum are widened
    to the minimum. Distances over the maximum are skipped.
    """
    if bar_range <= 0 or atr <= 0:
        return None
    raw = max(bar_range, atr * atr_mult, MIN_STOP)
    raw = round_tick(raw)
    if raw > MAX_STOP:
        return None
    return raw


def _hard_cap(equity: float) -> int:
    if equity < 20_000:
        return 1
    if equity < 40_000:
        return 2
    return 3


@dataclass(frozen=True)
class OrderPlan:
    action: str
    contracts: int = 0
    stop_distance: float | None = None
    risk_usd: float = 0.0
    risk_pct: float = 0.0
    elevated: bool = False
    reason: str = ""


def plan_order(
    equity: float,
    bar_range: float,
    atr: float,
    maintenance: float = MAINTENANCE_MARGIN,
    atr_mult: float = ATR_MULT,
) -> OrderPlan:
    """Decide whether a signal is tradable and how many MGC contracts to use."""
    if equity <= MIN_EQUITY:
        return OrderPlan("skip", reason="权益触及 8500 美元停机线")
    if maintenance <= 0:
        return OrderPlan("skip", reason="保证金无效")
    if maintenance > equity * MARGIN_PAUSE_EQUITY_SHARE:
        return OrderPlan("skip", reason="单手保证金超过权益的 30%")

    dist = stop_distance(bar_range, atr, atr_mult)
    if dist is None:
        return OrderPlan("skip", reason="止损须落在 5 到 12 美元/盎司，且不能缩到信号 K 线内部")

    risk_one = dist * POINT_VALUE
    if risk_one > equity * MAX_RISK_PCT:
        return OrderPlan(
            "skip",
            stop_distance=dist,
            risk_usd=risk_one,
            risk_pct=risk_one / equity,
            reason="单手风险超过权益的 1.2%",
        )

    by_risk = math.floor((equity * TARGET_RISK_PCT) / risk_one)
    by_margin = math.floor((equity * MARGIN_TO_EQUITY) / maintenance)
    hard = _hard_cap(equity)
    contracts = max(0, min(by_risk, by_margin, hard))
    elevated = False
    if contracts == 0:
        contracts = 1
        elevated = True
    if risk_one * contracts > equity * TARGET_RISK_PCT:
        elevated = True

    return OrderPlan(
        action="trade",
        contracts=contracts,
        stop_distance=dist,
        risk_usd=risk_one * contracts,
        risk_pct=(risk_one * contracts) / equity,
        elevated=elevated,
        reason="超目标风险但未破 1.2% 上限" if elevated else "目标风险内",
    )


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str


@dataclass
class Trade:
    exit_time: datetime
    side: str
    pnl_usd: float
    risk_usd: float

    @property
    def r_multiple(self) -> float:
        return self.pnl_usd / self.risk_usd


def _parse_time(value: str) -> datetime:
    text = value.strip().replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析时间: {value}")


def _normalize_side(value: str) -> str:
    text = value.strip().lower()
    if text in {"long", "l", "buy", "多"}:
        return "long"
    if text in {"short", "s", "sell", "空"}:
        return "short"
    raise ValueError(f"方向必须是 long 或 short: {value}")


def load_trades(path: str | Path) -> list[Trade]:
    trades: list[Trade] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"exit_time", "side", "pnl_usd", "risk_usd"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV 表头必须包含 exit_time,side,pnl_usd,risk_usd")
        for row in reader:
            risk = float(row["risk_usd"])
            if risk <= 0:
                raise ValueError("risk_usd 必须大于 0")
            trades.append(
                Trade(
                    exit_time=_parse_time(row["exit_time"]),
                    side=_normalize_side(row["side"]),
                    pnl_usd=float(row["pnl_usd"]),
                    risk_usd=risk,
                )
            )
    if not trades:
        raise ValueError("交易记录为空")
    return trades


def _profit_factor(pnls: list[float]) -> float | None:
    wins = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    if losses == 0:
        return None
    return wins / losses


def _max_drawdown(pnls: list[float], start_equity: float) -> float:
    equity = start_equity
    peak = start_equity
    worst = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        if peak > 0:
            worst = max(worst, (peak - equity) / peak)
    return worst


def _percentile_nearest_rank(values: list[float], q: float) -> float:
    ordered = sorted(values)
    rank = math.ceil(q * len(ordered)) - 1
    return ordered[min(max(rank, 0), len(ordered) - 1)]


@dataclass
class AcceptanceReport:
    passed: bool
    gates: list[Gate] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def text(self) -> str:
        lines = ["通过" if self.passed else "未通过", ""]
        for gate in self.gates:
            mark = "PASS" if gate.passed else "FAIL"
            lines.append(f"[{mark}] {gate.name}: {gate.detail}")
        return "\n".join(lines)


def evaluate(
    trades: list[Trade],
    start_equity: float = 10_000.0,
    side: str | None = None,
) -> AcceptanceReport:
    """Score a closed-trade list against the V2.0 acceptance gates.

    PnL values must already include commission and slippage. `side` restricts
    the book to one direction when a side has been disabled and retested.
    """
    if start_equity <= 0:
        raise ValueError("起始权益必须大于 0")
    book = trades if side is None else [t for t in trades if t.side == side]
    book = sorted(book, key=lambda t: t.exit_time)
    gates: list[Gate] = []

    def add(name: str, passed: bool, detail: str) -> None:
        gates.append(Gate(name, passed, detail))

    if len(book) < MIN_TRADES:
        add("样本笔数", False, f"{len(book)} 笔，至少需要 {MIN_TRADES} 笔")
    else:
        add("样本笔数", True, f"{len(book)} 笔")

    if len(book) >= 2:
        span_days = (book[-1].exit_time - book[0].exit_time).days
    else:
        span_days = 0
    years = span_days / 365.25
    add("样本跨度", years >= MIN_YEARS, f"{years:.2f} 年，至少 {MIN_YEARS:.0f} 年")

    pnls = [t.pnl_usd for t in book]
    end_equity = start_equity + sum(pnls)
    if years > 0 and end_equity > 0:
        cagr = (end_equity / start_equity) ** (1 / years) - 1
    else:
        cagr = -1.0
    add("年化收益", cagr >= MIN_CAGR, f"{cagr:.1%}，门槛 {MIN_CAGR:.0%}")

    drawdown = _max_drawdown(pnls, start_equity) if book else 1.0
    add("最大回撤", drawdown <= MAX_DRAWDOWN, f"{drawdown:.1%}，门槛不超过 {MAX_DRAWDOWN:.0%}")

    pf = _profit_factor(pnls)
    add(
        "利润因子",
        pf is not None and pf >= MIN_PROFIT_FACTOR,
        "无亏损样本，无法计算" if pf is None else f"{pf:.2f}，门槛 {MIN_PROFIT_FACTOR:.2f}",
    )

    expectancy = sum(t.r_multiple for t in book) / len(book) if book else -1.0
    add("期望值", expectancy >= MIN_EXPECTANCY_R, f"{expectancy:.3f}R，门槛 {MIN_EXPECTANCY_R:.2f}R")

    month_pnl: dict[str, float] = {}
    month_start: dict[str, float] = {}
    year_pnl: dict[int, float] = {}
    year_start: dict[int, float] = {}
    equity = start_equity
    for trade in book:
        month = trade.exit_time.strftime("%Y-%m")
        year = trade.exit_time.year
        month_start.setdefault(month, equity)
        year_start.setdefault(year, equity)
        month_pnl[month] = month_pnl.get(month, 0.0) + trade.pnl_usd
        year_pnl[year] = year_pnl.get(year, 0.0) + trade.pnl_usd
        equity += trade.pnl_usd

    if month_pnl:
        worst_month_key = min(month_pnl, key=lambda key: month_pnl[key] / month_start[key])
        worst_month = month_pnl[worst_month_key] / month_start[worst_month_key]
    else:
        worst_month_key, worst_month = "-", -1.0
    add("最差月份", worst_month >= WORST_MONTH, f"{worst_month_key} {worst_month:.1%}，不得差于 {WORST_MONTH:.0%}")

    if year_pnl:
        worst_year_key = min(year_pnl, key=lambda key: year_pnl[key] / year_start[key])
        worst_year = year_pnl[worst_year_key] / year_start[worst_year_key]
    else:
        worst_year_key, worst_year = "-", -1.0
    add("最差年份", worst_year >= WORST_YEAR, f"{worst_year_key} {worst_year:.1%}，不得差于 {WORST_YEAR:.0%}")

    if side is None:
        for name in ("long", "short"):
            side_pnls = [t.pnl_usd for t in book if t.side == name]
            side_pf = _profit_factor(side_pnls)
            ok = len(side_pnls) >= MIN_SIDE_TRADES and side_pf is not None and side_pf >= MIN_SIDE_PF
            detail = f"{len(side_pnls)} 笔"
            if side_pf is None:
                detail += "，无亏损样本或没有交易"
            else:
                detail += f"，利润因子 {side_pf:.2f}"
            add(f"{name} 方向", ok, detail + f"，至少 {MIN_SIDE_TRADES} 笔且利润因子 ≥ {MIN_SIDE_PF:.2f}")
    else:
        add("方向限制", True, f"只验收 {side}，这一侧必须单独通过其余门槛")

    ranked = sorted(pnls, reverse=True)
    trimmed = ranked[DROP_WINNERS:] if len(ranked) > DROP_WINNERS else []
    trimmed_pf = _profit_factor(trimmed)
    add(
        "去掉最大的 5 笔盈利",
        trimmed_pf is not None and trimmed_pf >= MIN_PF_AFTER_DROP,
        "剩余样本不足以计算" if trimmed_pf is None else f"利润因子 {trimmed_pf:.2f}，门槛 {MIN_PF_AFTER_DROP:.2f}",
    )

    rng = random.Random(MONTE_CARLO_SEED)
    drawdowns = []
    for _ in range(MONTE_CARLO_PATHS):
        shuffled = pnls[:]
        rng.shuffle(shuffled)
        drawdowns.append(_max_drawdown(shuffled, start_equity))
    mc95 = _percentile_nearest_rank(drawdowns, 0.95) if drawdowns else 1.0
    add("蒙特卡洛回撤", mc95 <= MONTE_CARLO_DD, f"95 分位 {mc95:.1%}，门槛不超过 {MONTE_CARLO_DD:.0%}")

    if book:
        cutoff = book[-1].exit_time.timestamp() - OOS_DAYS * 86400
        oos = [t.pnl_usd for t in book if t.exit_time.timestamp() >= cutoff]
        ins = [t.pnl_usd for t in book if t.exit_time.timestamp() < cutoff]
    else:
        oos, ins = [], []
    oos_net = sum(oos)
    ins_net = sum(ins)
    add(
        "样本外最后 12 个月",
        bool(oos) and oos_net > 0 and bool(ins) and ins_net > 0,
        f"最近 12 个月 {oos_net:.0f} 美元（{len(oos)} 笔），此前 {ins_net:.0f} 美元（{len(ins)} 笔），两段都要为正",
    )

    return AcceptanceReport(
        passed=all(gate.passed for gate in gates),
        gates=gates,
        summary={
            "trades": len(book),
            "years": years,
            "net_usd": end_equity - start_equity,
            "cagr": cagr,
            "max_drawdown": drawdown,
            "profit_factor": pf,
            "expectancy_r": expectancy,
        },
    )
