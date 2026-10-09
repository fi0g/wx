"""Lock the V2.0 sizing formula and acceptance gates."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from rules import (
    evaluate,
    plan_order,
    round_tick,
    stop_distance,
    Trade,
)


class StopTests(unittest.TestCase):
    def test_widens_noise_stop_to_minimum(self):
        self.assertEqual(stop_distance(bar_range=3.0, atr=3.0), 5.0)

    def test_uses_wider_of_bar_and_atr(self):
        # 1.2 * 7.4 = 8.88, rounds half-up to 8.9
        self.assertEqual(stop_distance(bar_range=7.6, atr=7.4), 8.9)

    def test_does_not_tighten_a_wide_bar(self):
        self.assertIsNone(stop_distance(bar_range=13.0, atr=6.0))

    def test_cap_is_inclusive_at_12(self):
        self.assertEqual(stop_distance(bar_range=12.0, atr=4.0), 12.0)
        self.assertIsNone(stop_distance(bar_range=12.1, atr=4.0))

    def test_tick_round(self):
        self.assertEqual(round_tick(8.84), 8.8)
        self.assertEqual(round_tick(8.85), 8.9)


class SizingTests(unittest.TestCase):
    def test_ten_thousand_one_contract_inside_target(self):
        plan = plan_order(10_000, bar_range=7.6, atr=7.4)
        self.assertEqual(plan.action, "trade")
        self.assertEqual(plan.contracts, 1)
        self.assertEqual(plan.stop_distance, 8.9)
        self.assertEqual(plan.risk_usd, 89.0)
        self.assertFalse(plan.elevated)

    def test_ten_thousand_elevated_but_allowed(self):
        plan = plan_order(10_000, bar_range=11.5, atr=4.0)
        self.assertEqual(plan.action, "trade")
        self.assertEqual(plan.contracts, 1)
        self.assertEqual(plan.risk_usd, 115.0)
        self.assertTrue(plan.elevated)
        self.assertLessEqual(plan.risk_pct, 0.012)

    def test_ten_thousand_never_two_contracts(self):
        plan = plan_order(10_000, bar_range=5.0, atr=4.0)
        self.assertEqual(plan.contracts, 1)
        self.assertEqual(plan.risk_usd, 50.0)

    def test_stop_above_cap_is_a_skip(self):
        plan = plan_order(10_000, bar_range=14.0, atr=8.0)
        self.assertEqual(plan.action, "skip")
        self.assertEqual(plan.contracts, 0)

    def test_kill_switch(self):
        plan = plan_order(8_500, bar_range=6.0, atr=5.0)
        self.assertEqual(plan.action, "skip")

    def test_margin_pause(self):
        plan = plan_order(10_000, bar_range=6.0, atr=5.0, maintenance=3_500)
        self.assertEqual(plan.action, "skip")

    def test_two_contracts_only_after_equity_and_risk_allow_it(self):
        plan = plan_order(20_000, bar_range=8.0, atr=4.0)
        self.assertEqual(plan.action, "trade")
        self.assertEqual(plan.contracts, 2)
        self.assertEqual(plan.risk_usd, 160.0)
        self.assertFalse(plan.elevated)

        still_one = plan_order(20_000, bar_range=11.0, atr=4.0)
        self.assertEqual(still_one.contracts, 1)
        self.assertEqual(still_one.risk_usd, 110.0)


def _book() -> list[Trade]:
    """200 closed trades over four years: +2R / -1R, 50% wins, both sides."""
    start = datetime(2022, 10, 3, 15, 45)
    trades: list[Trade] = []
    for i in range(200):
        win = i % 2 == 0
        trades.append(
            Trade(
                exit_time=start + timedelta(days=7 * i),
                side="long" if i % 3 else "short",
                pnl_usd=200.0 if win else -100.0,
                risk_usd=100.0,
            )
        )
    return trades


class AcceptanceTests(unittest.TestCase):
    def test_reference_book_passes(self):
        report = evaluate(_book())
        failed = [gate.name for gate in report.gates if not gate.passed]
        self.assertEqual(failed, [], report.text())
        self.assertTrue(report.passed)
        self.assertGreaterEqual(report.summary["cagr"], 0.15)
        self.assertGreaterEqual(report.summary["expectancy_r"], 0.20)

    def test_short_sample_fails(self):
        report = evaluate(_book()[:10])
        self.assertFalse(report.passed)
        names = {gate.name: gate.passed for gate in report.gates}
        self.assertFalse(names["样本笔数"])
        self.assertFalse(names["样本跨度"])

    def test_side_filter_retests_one_direction(self):
        trades = [
            Trade(t.exit_time, "long", t.pnl_usd, t.risk_usd) for t in _book()
        ]
        report = evaluate(trades, side="long")
        self.assertTrue(report.passed, report.text())


if __name__ == "__main__":
    unittest.main()
