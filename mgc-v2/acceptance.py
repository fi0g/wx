"""Score an exported MGC V2.0 trade list against the acceptance gates.

CSV header: exit_time,side,pnl_usd,risk_usd
PnL must already include commission and slippage.
"""

from __future__ import annotations

import argparse
import sys

from rules import evaluate, load_trades


def main() -> int:
    parser = argparse.ArgumentParser(description="验收 MGC V2.0 交易记录")
    parser.add_argument("csv_path", help="成交 CSV 路径")
    parser.add_argument("--equity", type=float, default=10_000.0, help="回测起始权益")
    parser.add_argument("--side", choices=["long", "short"], help="只验收一个方向")
    args = parser.parse_args()
    report = evaluate(load_trades(args.csv_path), start_equity=args.equity, side=args.side)
    print(report.text())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
