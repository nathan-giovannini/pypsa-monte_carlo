import json
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NOTEBOOK = Path(__file__).resolve().parents[1] / "post_proc_cba.ipynb"


class CompensationNotebookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text())
        cls.cells = {
            cell.get("id"): "".join(cell.get("source", []))
            for cell in notebook["cells"]
        }

    def setUp(self):
        self.namespace = {
            "np": np,
            "pd": pd,
            "plt": plt,
            "COST_UNIT": 1_000_000,
            "annualized_project_costs": pd.DataFrame(
                [
                    ("P1", "BE", 3.0),
                    ("P1", "GB", 1.0),
                    ("P2", "BE", 1.0),
                    ("P2", "NL", 3.0),
                ],
                columns=["counterfactual", "country", "annualized_cost_2040_eur_m_per_year"],
            ),
            "benefit_distribution": pd.DataFrame(
                [
                    ("P1", "BE", 1.0),
                    ("P1", "GB", 1.0),
                    ("P2", "BE", 1.0),
                    ("P2", "NL", 1.0),
                    ("Bundle", "BE", 1.0),
                    ("Bundle", "GB", 1.0),
                    ("Bundle", "NL", 1.0),
                    ("Bundle", "FR", 99.0),
                ],
                columns=["counterfactual", "country", "expected_benefit"],
            ),
        }
        exec(self.cells["refactored-section8-005"], self.namespace)

    def tearDown(self):
        plt.close("all")

    def test_receipts_and_payments_are_summed_before_netting(self):
        projects = self.namespace["project_compensation"]
        comparison = self.namespace["df_compensation"].set_index(["counterfactual", "country"])
        positive = "positive_compensation_2040_eur_per_year"
        negative = "negative_compensation_2040_eur_per_year"
        net = "compensation_2040_eur_per_year"

        self.assertEqual(comparison.loc[("Sum of projects", "BE"), positive], 1_000_000)
        self.assertEqual(comparison.loc[("Sum of projects", "BE"), negative], -1_000_000)
        self.assertEqual(comparison.loc[("Sum of projects", "BE"), net], 0)
        self.assertNotIn(("Bundle", "FR"), comparison.index)
        self.assertEqual(set(zip(projects.counterfactual, projects.country)), {
            ("P1", "BE"), ("P1", "GB"), ("P2", "BE"), ("P2", "NL")
        })
        np.testing.assert_allclose(comparison[positive] + comparison[negative], comparison[net])
        bundle = comparison.loc["Bundle"]
        np.testing.assert_allclose(bundle[positive], bundle[net].clip(lower=0))
        np.testing.assert_allclose(bundle[negative], bundle[net].clip(upper=0))
        for scenario in ("Sum of projects", "Bundle"):
            self.assertAlmostEqual(comparison.loc[scenario, net].sum(), 0)

    def test_project_and_comparison_plots(self):
        shown = []
        original_show = plt.show
        plt.show = lambda: shown.append(plt.gcf())
        try:
            exec(self.cells["6a8e843d4a3d55f4"], self.namespace)
            self.assertEqual(len(shown), 2)
            self.assertEqual({figure.axes[0].get_title() for figure in shown}, {"P1", "P2"})
            exec(self.cells["refactored-section8-007"], self.namespace)
            self.assertEqual(len(shown), 3)
            self.assertEqual(
                [axis.get_title() for axis in shown[-1].axes],
                ["Positive (received)", "Negative (paid)", "Net"],
            )
        finally:
            plt.show = original_show

    def test_sew_plot_uses_full_sample_distribution_and_signed_net_transfer(self):
        samples = []
        for scenario in ("Sum of projects", "Bundle"):
            for sample, multiplier in ((0, 1), (1, 2)):
                for country, value in (("BE", 10), ("OBZ_BE", 2), ("GB", 5), ("NL", 7), ("FR", 3)):
                    samples.append((country, scenario, sample, value * multiplier))
        self.namespace["df"] = pd.DataFrame(
            samples, columns=["country", "counterfactual", "sample", "effect"]
        )
        shown = []
        original_show = plt.show
        plt.show = lambda: shown.append(plt.gcf())
        try:
            exec(self.cells["sew-compensation-plot"], self.namespace)
        finally:
            plt.show = original_show
        positions = self.namespace["sew_expected_positions"].set_index(["country", "counterfactual"])
        expected = "expected_sew"
        adjusted = "expected_sew_after_compensation"
        net = "compensation_2040_eur_per_year"
        self.assertEqual(positions.loc[("BE", "Sum of projects"), expected], 18)
        self.assertEqual(positions.loc[("BE", "Sum of projects"), adjusted], 18)
        self.assertEqual(positions.loc[("GB", "Sum of projects"), adjusted], 7.5 - 1_000_000)
        self.assertEqual(positions.loc[("NL", "Sum of projects"), adjusted], 10.5 + 1_000_000)
        self.assertEqual(positions.loc[("FR", "Bundle"), adjusted], 4.5)
        np.testing.assert_allclose(positions[adjusted], positions[expected] + positions[net])
        self.assertEqual(len(shown), 1)
        self.assertIn("FR", [ax.get_title() for ax in shown[0].axes if ax.get_visible()])
        gb_axis = next(ax for ax in shown[0].axes if ax.get_title() == "GB")
        self.assertTrue(any(
            line.get_linestyle() == "--" and line.get_xdata()[0] == 7.5 - 1_000_000
            for line in gb_axis.lines
        ))

    def test_sew_plot_rejects_missing_participant_transfer(self):
        self.namespace["df"] = pd.DataFrame(
            [("BE", scenario, 0, 1.0) for scenario in ("Sum of projects", "Bundle")],
            columns=["country", "counterfactual", "sample", "effect"],
        )
        self.namespace["df_compensation"] = self.namespace["df_compensation"].query(
            "not (country == 'BE' and counterfactual == 'Bundle')"
        )
        with self.assertRaisesRegex(ValueError, "one transfer"):
            exec(self.cells["sew-compensation-plot"], self.namespace)

    def test_negative_sew_counts_samples_after_signed_compensation(self):
        samples = []
        for sample in range(3):
            for country in ("BE", "GB", "NL", "NO"):
                for scenario in ("Sum of projects", "Bundle"):
                    samples.append((country, scenario, sample, 0.0))
        # BE's offsetting project transfers have zero net. Offshore-zone
        # benefits make sample 0 exactly zero and sample 1 negative.
        samples.extend([
            ("OBZ_BE", "Sum of projects", 0, 0.0),
            ("OBZ_BE", "Sum of projects", 1, -2_000_000.0),
            ("OBZ_BE", "Sum of projects", 2, 1.0),
            ("OBZ_BE", "Bundle", 0, -2_000_000.0),
            ("OBZ_BE", "Bundle", 1, -2_000_000.0),
            ("OBZ_BE", "Bundle", 2, 1.0),
        ])
        self.namespace["df"] = pd.DataFrame(
            samples, columns=["country", "counterfactual", "sample", "effect"]
        )
        exec(self.cells["negative-sew-analysis"], self.namespace)
        comparison = self.namespace["negative_sew_comparison"]
        self.assertEqual(comparison.loc["BE", "negative_count_sum_of_projects"], 1)
        self.assertEqual(comparison.loc["BE", "negative_count_bundle"], 2)
        self.assertEqual(comparison.loc["BE", "difference_count_sum_minus_bundle"], -1)
        self.assertAlmostEqual(comparison.loc["BE", "difference_percentage_points_sum_minus_bundle"], -100 / 3)
        self.assertEqual(comparison.loc["NO", "negative_count_bundle"], 0)
        self.assertEqual(comparison.loc["GB", "negative_count_sum_of_projects"], 3)
        self.assertEqual(comparison.loc["NL", "negative_count_sum_of_projects"], 0)
        original_show = plt.show
        shown = []
        plt.show = lambda: shown.append(plt.gcf())
        try:
            exec(self.cells["negative-sew-plot"], self.namespace)
        finally:
            plt.show = original_show
        self.assertEqual(len(shown), 1)
        self.assertEqual(len(shown[0].axes), 2)
        self.assertEqual(len(shown[0].axes[0].patches), 2 * len(comparison))

    def test_negative_sew_requires_matching_samples(self):
        self.namespace["df"] = pd.DataFrame(
            [("BE", "Sum of projects", 0, -1.0), ("BE", "Bundle", 1, -1.0)],
            columns=["country", "counterfactual", "sample", "effect"],
        )
        with self.assertRaisesRegex(ValueError, "same samples"):
            exec(self.cells["negative-sew-analysis"], self.namespace)


if __name__ == "__main__":
    unittest.main()
