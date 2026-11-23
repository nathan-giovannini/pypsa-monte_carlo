import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bundling_analysis import (
    build_cba_samples,
    calculate_joins,
    load_welfare_samples,
)
import scripts.bundling_analysis as bundling_analysis


class BundlingAnalysisTest(unittest.TestCase):
    def test_cba_pairs_bundle_and_sum_to_individual_sums(self):
        original = bundling_analysis.SCENARIOS
        bundling_analysis.SCENARIOS = {"A": "no-A", "B": "no-B"}
        self.addCleanup(setattr, bundling_analysis, "SCENARIOS", original)
        welfare = pd.DataFrame(
            [
                ("base", "BE", 0, 100),
                ("base", "BE", 1, 120),
                ("no-A", "BE", 0, 80),
                ("no-A", "BE", 1, 90),
                ("no-B", "BE", 0, 70),
                ("no-B", "BE", 1, 80),
                ("bundle", "BE", 0, 60),
                ("bundle", "BE", 1, 70),
            ],
            columns=["scenario", "country", "sample", "total"],
        )
        cba = build_cba_samples(welfare)
        be = cba[cba["country"] == "BE"]
        sums = be[be["counterfactual"] == "Sum of projects"].set_index("sample")["effect"]
        bundle = be[be["counterfactual"] == "Bundle"].set_index("sample")["effect"]
        np.testing.assert_allclose(sums.to_numpy(), [50, 70])
        np.testing.assert_allclose(bundle.to_numpy(), [40, 50])

    def test_framework1_compensation_is_signed_and_balanced(self):
        benefits = pd.DataFrame(
            [("BE", 8_000_000.0), ("GB", -2_000_000.0)],
            columns=["country", "expected_benefit"],
        )
        costs = pd.DataFrame(
            [("BE", 4.0), ("GB", 4.0)],
            columns=["country", "annualized_cost_2040_eur_m_per_year"],
        )
        table = calculate_joins("Test", benefits, costs, {"BE", "GB"}, "1")
        be = table.loc[table["country"] == "BE", "compensation_2040_eur_per_year"].iloc[0]
        gb = table.loc[table["country"] == "GB", "compensation_2040_eur_per_year"].iloc[0]
        self.assertLess(be, 0)
        self.assertGreater(gb, 0)

    def test_load_welfare_rejects_missing_scenario(self):
        with self.assertRaises(FileNotFoundError):
            load_welfare_samples(Path("/definitely/missing"))


if __name__ == "__main__":
    unittest.main()
