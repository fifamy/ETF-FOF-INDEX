import unittest

import numpy as np
import pandas as pd

from etf_fof_index.report import summarize_period_metrics


class SummarizePeriodMetricsTest(unittest.TestCase):
    def test_monthly_uses_period_returns_and_reset_drawdown(self) -> None:
        index = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-02", "2024-02-29", "2024-03-01"])
        levels = pd.DataFrame(
            {
                "组合A": [1.0, 1.10, 1.045, 1.0659, 1.055241],
                "组合B": [1.0, 0.98, 0.9898, 0.960106, 0.99851024],
            },
            index=index,
        )

        actual = summarize_period_metrics(levels, freq="ME")
        feb_a = actual.loc[(actual["period"] == "2024-02") & (actual["strategy"] == "组合A")].iloc[0]

        expected_returns = np.array([0.10, -0.05, 0.02], dtype=float)
        expected_vol = float(expected_returns.std(ddof=0) * np.sqrt(252.0))

        self.assertEqual(feb_a["period_start"], "2024-02-01")
        self.assertEqual(feb_a["period_end"], "2024-02-29")
        self.assertEqual(int(feb_a["observations"]), 3)
        self.assertAlmostEqual(feb_a["period_return"], float((1.0 + expected_returns).prod() - 1.0))
        self.assertAlmostEqual(feb_a["annualized_volatility"], expected_vol)
        self.assertAlmostEqual(feb_a["max_drawdown"], -0.05)

    def test_yearly_splits_periods_correctly(self) -> None:
        index = pd.to_datetime(["2024-12-31", "2025-01-02", "2025-12-31"])
        levels = pd.DataFrame({"组合A": [1.0, 1.10, 1.045]}, index=index)

        actual = summarize_period_metrics(levels, freq="YE")

        self.assertEqual(actual["period"].tolist(), ["2024", "2025"])
        self.assertEqual(actual["strategy"].tolist(), ["组合A", "组合A"])
        self.assertAlmostEqual(float(actual.iloc[0]["period_return"]), 0.0)
        self.assertAlmostEqual(float(actual.iloc[1]["period_return"]), 0.045)


if __name__ == "__main__":
    unittest.main()
