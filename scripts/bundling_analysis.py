#!/usr/bin/env python3
"""Ex-ante and ex-post analysis of standalone projects versus a project bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COST_UNIT = 1_000_000
SCENARIOS = {
    "Nautilus": "no-Nautilus",
    "TritonLink": "no-TritonLink",
    "LionLink": "no-LionLink",
    "DE-GB interconnector": "no-DE-GB_interconnector",
    "FR-GB interconnector": "no-FR-GB_interconnector",
}
PROJECT_NAME_MAP = {
    "Nautilus GB": "Nautilus",
    "Nautilus BE": "Nautilus",
    "TritonLink DK": "TritonLink",
    "TritonLink BE": "TritonLink",
    "FR-UK interconnector": "FR-GB interconnector",
    "DE-UK interconnector": "DE-GB interconnector",
    "LionLink": "LionLink",
}
NORTH_SEA = {"DE", "DK", "NL", "BE", "GB", "FR"}
PERCENTILES = (50, 75, 90, 95)


def normalize_country(country: object) -> str:
    return str(country).replace("OBZ_BE", "BE").replace("OBZ_NL", "NL")


def load_project_costs(
    path: Path,
    cost_base_year: int = 2024,
    benefit_year: int = 2040,
    project_start_year: int = 2035,
    lifespan_years: int = 25,
    interest_rate: float = 0.04,
) -> pd.DataFrame:
    workbook = pd.read_excel(path, sheet_name="ProjectsCost")
    required = {"Project name", "CAPEX", "OPEX", "country1", "share1"}
    missing = required - set(workbook.columns)
    if missing:
        raise ValueError(f"Missing project-cost columns: {sorted(missing)}")

    workbook["counterfactual"] = workbook["Project name"].map(PROJECT_NAME_MAP)
    if workbook["counterfactual"].isna().any():
        names = workbook.loc[workbook["counterfactual"].isna(), "Project name"].unique()
        raise ValueError(f"No CBA mapping for workbook projects: {list(names)}")

    cost_rows = []
    for _, row in workbook.iterrows():
        share_total = 0.0
        for country_col, share_col in (
            ("country1", "share1"),
            ("country2", "share2"),
            ("country3", "share3"),
        ):
            country = row.get(country_col)
            share = row.get(share_col)
            if pd.isna(country) and pd.isna(share):
                continue
            if pd.isna(country) or pd.isna(share):
                raise ValueError(
                    f"Incomplete country/share pair for {row['Project name']}: "
                    f"{country_col}, {share_col}"
                )
            share = float(share)
            if not np.isfinite(share) or share < 0 or share > 1:
                raise ValueError(f"Invalid cost share for {row['Project name']}: {share}")
            share_total += share
            cost_rows.append(
                {
                    "project_name": row["Project name"],
                    "counterfactual": row["counterfactual"],
                    "country": normalize_country(country),
                    "cost_share": share,
                    "capex_2024_eur_m": float(row["CAPEX"]),
                    "opex_2024_eur_m_per_year": float(row["OPEX"]),
                }
            )
        if not np.isclose(share_total, 1.0):
            raise ValueError(
                f"Country cost shares must sum to 1 for {row['Project name']}: {share_total}"
            )

    costs = pd.DataFrame(cost_rows)
    cost_to_benefit = (1 + interest_rate) ** (benefit_year - cost_base_year)
    crf = (
        interest_rate * (1 + interest_rate) ** lifespan_years
        / ((1 + interest_rate) ** lifespan_years - 1)
    )
    costs["annualized_cost_2040_eur_m_per_year"] = (
        costs["capex_2024_eur_m"] * cost_to_benefit * crf
        + costs["opex_2024_eur_m_per_year"] * cost_to_benefit
    ) * costs["cost_share"]

    annualized = (
        costs.groupby(["counterfactual", "country"], as_index=False)[
            "annualized_cost_2040_eur_m_per_year"
        ].sum()
    )
    annualized["cost_share"] = (
        annualized["annualized_cost_2040_eur_m_per_year"]
        / annualized.groupby("counterfactual")[
            "annualized_cost_2040_eur_m_per_year"
        ].transform("sum")
    )
    return annualized


def load_welfare_samples(results_dir: Path) -> pd.DataFrame:
    frames = []
    for scenario_name in ["base", "bundle", *SCENARIOS.values()]:
        path = results_dir / scenario_name / "welfare_by_country.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing welfare table: {path}")
        frame = pd.read_csv(path)
        required = {"country", "sample", "total"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} lacks columns {sorted(required)}")
        frame["scenario"] = scenario_name
        frames.append(frame)
    welfare = pd.concat(frames, ignore_index=True)
    welfare["country"] = welfare["country"].map(normalize_country)
    welfare = (
        welfare.groupby(["scenario", "country", "sample"], as_index=False)["total"].sum()
    )
    return welfare


def build_cba_samples(welfare: pd.DataFrame) -> pd.DataFrame:
    base = welfare[welfare["scenario"] == "base"][["country", "sample", "total"]]
    base = base.rename(columns={"total": "base_effect"})
    rows = []

    for project, counterfactual in SCENARIOS.items():
        scenario = welfare[welfare["scenario"] == counterfactual][
            ["country", "sample", "total"]
        ].rename(columns={"total": "scenario_effect"})
        effects = base.merge(
            scenario, on=["country", "sample"], how="outer", validate="one_to_one"
        )
        if effects.isna().any().any():
            raise ValueError(f"Base and {counterfactual} are not fully paired")
        effects["effect"] = effects["base_effect"] - effects["scenario_effect"]
        effects["counterfactual"] = project
        rows.append(effects[["country", "counterfactual", "sample", "effect"]])

    bundle = welfare[welfare["scenario"] == "bundle"][
        ["country", "sample", "total"]
    ].rename(columns={"total": "scenario_effect"})
    bundle_effect = base.merge(
        bundle, on=["country", "sample"], how="outer", validate="one_to_one"
    )
    if bundle_effect.isna().any().any():
        raise ValueError("Base and bundle are not fully paired")
    bundle_effect["effect"] = bundle_effect["base_effect"] - bundle_effect["scenario_effect"]
    bundle_effect["counterfactual"] = "Bundle"
    rows.append(bundle_effect[["country", "counterfactual", "sample", "effect"]])

    standalone = pd.concat(rows[:-1], ignore_index=True)
    standalone = (
        standalone.groupby(["country", "sample"], as_index=False)["effect"].sum()
    )
    standalone["counterfactual"] = "Sum of projects"
    rows.append(standalone)
    cba = pd.concat(rows, ignore_index=True)
    if cba.isna().any().any() or not np.isfinite(cba["effect"]).all():
        raise ValueError("CBA samples contain missing or non-finite values")
    return cba


def benefit_distribution(cba: pd.DataFrame) -> pd.DataFrame:
    return (
        cba.groupby(["country", "counterfactual"], as_index=False)["effect"]
        .agg(expected_benefit="mean", standard_deviation="std", samples="count")
    )


def bundle_costs(project_costs: pd.DataFrame) -> pd.DataFrame:
    cost_column = "annualized_cost_2040_eur_m_per_year"
    costs = (
        project_costs.groupby("country", as_index=False)[cost_column].sum()
        .assign(counterfactual="Bundle")
    )
    return costs


def country_cost_table(project_costs: pd.DataFrame) -> pd.DataFrame:
    cost_column = "annualized_cost_2040_eur_m_per_year"
    bundle = bundle_costs(project_costs)
    return pd.concat(
        [project_costs[["counterfactual", "country", cost_column]], bundle],
        ignore_index=True,
    )


def participant_set(
    scenario: str,
    framework: str,
    case: str,
    benefits: pd.DataFrame,
    project_costs: pd.DataFrame,
    europe: set[str],
) -> set[str]:
    scenario_benefits = benefits[benefits["counterfactual"] == scenario]
    benefit_by_country = scenario_benefits.set_index("country")["expected_benefit"]
    total_positive = benefit_by_country.abs().sum()
    above_threshold = set(
        benefit_by_country.index[
            benefit_by_country.abs() / total_positive > 0.10
        ].tolist()
    ) if total_positive else set()

    if framework == "1" and case == "1a":
        if scenario == "Sum of projects":
            return set(project_costs["country"])
        if scenario == "Bundle":
            return set(project_costs["country"])
        return set(
            project_costs.loc[
                project_costs["counterfactual"] == scenario, "country"
            ]
        )

    if framework == "1" and case in {"1b", "1c"}:
        involved = set(project_costs["country"])
        if case == "1c":
            involved &= europe
        return involved | (above_threshold & europe if case == "1c" else above_threshold)

    if framework == "2" and case == "2a":
        if scenario == "Sum of projects":
            return set(project_costs["country"])
        if scenario == "Bundle":
            return set(project_costs["country"])
        return set(
            project_costs.loc[
                project_costs["counterfactual"] == scenario, "country"
            ]
        )
    if framework == "2" and case == "2b":
        return NORTH_SEA & set(benefit_by_country.index)
    if framework == "2" and case == "2c":
        return europe & set(benefit_by_country.index)
    raise ValueError(f"Unknown framework/case: {framework}/{case}")


def exante_compensation(
    benefits: pd.DataFrame,
    project_costs: pd.DataFrame,
    europe: set[str],
) -> pd.DataFrame:
    cost_column = "annualized_cost_2040_eur_m_per_year"
    costs = country_cost_table(project_costs)
    scenarios = ["Sum of projects", "Bundle"]
    rows = []

    for framework in ("1", "2"):
        cases = ("1a", "1b", "1c") if framework == "1" else ("2a", "2b", "2c")
        for case in cases:
            for scenario in scenarios:
                if scenario == "Sum of projects":
                    individual_rows = []
                    for project in project_costs["counterfactual"].unique():
                        project_benefits = benefits[
                            benefits["counterfactual"] == project
                        ][["country", "expected_benefit"]]
                        project_cost = project_costs[
                            project_costs["counterfactual"] == project
                        ][["country", cost_column]]
                        participants = participant_set(
                            project,
                            framework,
                            case,
                            benefits,
                            project_costs,
                            europe,
                        )
                        individual_rows.append(
                            calculate_joins(
                                project,
                                project_benefits,
                                project_cost,
                                participants,
                                framework,
                            )
                        )
                    result = pd.concat(individual_rows, ignore_index=True)
                    result = (
                        result.groupby("country", as_index=False)[
                            ["compensation_2040_eur_per_year",
                             "net_position_2040_eur_per_year"]
                        ].sum()
                        .assign(counterfactual=scenario)
                    )
                else:
                    bundle_benefits = benefits[
                        benefits["counterfactual"] == "Bundle"
                    ][["country", "expected_benefit"]]
                    bundle_cost = costs[costs["counterfactual"] == "Bundle"][
                        ["country", cost_column]
                    ]
                    participants = participant_set(
                        "Bundle",
                        framework,
                        case,
                        benefits,
                        project_costs,
                        europe,
                    )
                    result = calculate_joins(
                        "Bundle",
                        bundle_benefits,
                        bundle_cost,
                        participants,
                        framework,
                    ).assign(counterfactual=scenario)

                result["framework"] = framework
                result["case"] = case
                rows.append(result)

    result = pd.concat(rows, ignore_index=True)
    result["positive_compensation_2040_eur_per_year"] = result[
        "compensation_2040_eur_per_year"
    ].clip(lower=0)
    result["negative_compensation_2040_eur_per_year"] = result[
        "compensation_2040_eur_per_year"
    ].clip(upper=0)
    return result


def calculate_joins(
    counterfactual: str,
    benefits: pd.DataFrame,
    costs: pd.DataFrame,
    participants: set[str],
    framework: str,
) -> pd.DataFrame:
    cost_column = "annualized_cost_2040_eur_m_per_year"
    table = (
        pd.DataFrame({"country": sorted(participants)})
        .merge(benefits, on="country", how="left", validate="one_to_one")
        .merge(costs, on="country", how="left", validate="one_to_one")
        .fillna({"expected_benefit": 0.0, cost_column: 0.0})
    )
    table["cost_2040_eur_per_year"] = table[cost_column] * COST_UNIT
    total_cost = table["cost_2040_eur_per_year"].sum()

    if framework == "1":
        table["net_position_2040_eur_per_year"] = (
            table["expected_benefit"] - table["cost_2040_eur_per_year"]
        )
        table["compensation_2040_eur_per_year"] = (
            table["cost_2040_eur_per_year"] - table["expected_benefit"]
        )
    else:
        benefit_total = table["expected_benefit"].sum()
        if abs(benefit_total) <= 1e-12:
            raise ValueError(f"Cannot calculate benefit shares for {counterfactual}")
        table["benefit_share"] = table["expected_benefit"] / benefit_total
        table["cost_share"] = (
            table["cost_2040_eur_per_year"] / total_cost
            if total_cost > 0
            else 0.0
        )
        table["benefit_based_allocation_2040_eur_per_year"] = (
            table["benefit_share"] * total_cost
        )
        table["net_position_2040_eur_per_year"] = (
            table["expected_benefit"]
            - table["benefit_based_allocation_2040_eur_per_year"]
        )
        table["compensation_2040_eur_per_year"] = (
            table["cost_2040_eur_per_year"]
            - table["benefit_based_allocation_2040_eur_per_year"]
        )

    if framework == "2" and abs(table["compensation_2040_eur_per_year"].sum()) > 1e-3:
        raise ValueError(
            f"Compensation does not balance for {counterfactual}: "
            f"{table['compensation_2040_eur_per_year'].sum()}"
        )
    table["counterfactual"] = counterfactual
    return table[
        [
            "country",
            "counterfactual",
            "expected_benefit",
            "cost_2040_eur_per_year",
            "net_position_2040_eur_per_year",
            "compensation_2040_eur_per_year",
        ]
    ]


def variability(cba: pd.DataFrame) -> pd.DataFrame:
    relevant = cba[cba["counterfactual"].isin(["Sum of projects", "Bundle"])]
    aggregate = (
        relevant.groupby(["counterfactual", "sample"], as_index=False)["effect"].sum()
        .rename(columns={"effect": "aggregate_effect"})
    )
    country = (
        relevant.groupby(["country", "counterfactual"], as_index=False)["effect"]
        .agg(
            mean="mean",
            std="std",
            min="min",
            p25=lambda x: x.quantile(0.25),
            median="median",
            p75=lambda x: x.quantile(0.75),
            p95=lambda x: x.quantile(0.95),
            max="max",
            samples="count",
        )
    )
    country["coefficient_of_variation"] = (
        country["std"] / country["mean"].abs().replace(0, np.nan)
    )
    aggregate_stats = (
        aggregate.groupby("counterfactual", as_index=False)["aggregate_effect"]
        .agg(
            mean="mean",
            std="std",
            min="min",
            p25=lambda x: x.quantile(0.25),
            median="median",
            p75=lambda x: x.quantile(0.75),
            p95=lambda x: x.quantile(0.95),
            max="max",
            samples="count",
        )
    )
    aggregate_stats["scope"] = "aggregate"
    country["scope"] = "country"
    return country, aggregate_stats


def net_loser_analysis(
    cba: pd.DataFrame,
    exante: pd.DataFrame,
    framework: str,
    case: str,
) -> pd.DataFrame:
    relevant_comp = exante[
        (exante["framework"] == framework)
        & (exante["case"] == case)
        & exante["counterfactual"].isin(["Sum of projects", "Bundle"])
    ][["country", "counterfactual", "compensation_2040_eur_per_year"]]
    samples = cba[cba["counterfactual"].isin(["Sum of projects", "Bundle"])][
        ["country", "counterfactual", "sample", "effect"]
    ].copy()
    paired = samples.merge(
        relevant_comp,
        on=["country", "counterfactual"],
        how="left",
        validate="many_to_one",
    )
    paired["compensation_2040_eur_per_year"] = paired[
        "compensation_2040_eur_per_year"
    ].fillna(0.0)
    paired["position_after_compensation"] = (
        paired["effect"] + paired["compensation_2040_eur_per_year"]
    )
    paired["is_net_loser"] = paired["position_after_compensation"] < 0

    pivot = paired.pivot(index=["country", "sample"], columns="counterfactual", values="is_net_loser")
    if pivot.isna().any().any():
        raise ValueError("The sum and bundle must have matching sample sets for every country")

    stats = (
        paired.groupby(["country", "counterfactual"], as_index=False)
        .agg(
            loser_count=("is_net_loser", "sum"),
            sample_count=("is_net_loser", "size"),
            loser_frequency=("is_net_loser", "mean"),
            adverse_mean=("position_after_compensation", lambda x: x[x < 0].mean()),
            adverse_median=("position_after_compensation", lambda x: x[x < 0].median()),
            adverse_p25=("position_after_compensation", lambda x: x[x < 0].quantile(0.25)),
            adverse_p75=("position_after_compensation", lambda x: x[x < 0].quantile(0.75)),
            adverse_min=("position_after_compensation", lambda x: x[x < 0].min()),
        )
    )
    stats["framework"] = framework
    stats["case"] = case
    stats["loser_frequency"] *= 100
    return stats


def perfect_expost(
    cba: pd.DataFrame,
    exante: pd.DataFrame,
    framework: str,
    case: str,
) -> pd.DataFrame:
    base = net_loser_samples(cba, exante, framework, case)
    base["perfect_adjustment"] = -base["position_after_compensation"].clip(upper=0)
    base["perfect_payment"] = base["position_after_compensation"].clip(lower=0)
    base["framework"] = framework
    base["case"] = case
    stats = (
        base.groupby(["country", "counterfactual"], as_index=False)
        .agg(
            adjustment_mean=("perfect_adjustment", "mean"),
            adjustment_median=("perfect_adjustment", "median"),
            adjustment_p95=("perfect_adjustment", lambda x: x.quantile(0.95)),
            adjustment_max=("perfect_adjustment", "max"),
            gross_receipts=("perfect_adjustment", "sum"),
            gross_surplus=("perfect_payment", "sum"),
            sample_count=("sample", "count"),
        )
    )
    stats["framework"] = framework
    stats["case"] = case
    return stats


def net_loser_samples(
    cba: pd.DataFrame,
    exante: pd.DataFrame,
    framework: str,
    case: str,
) -> pd.DataFrame:
    comp = exante[
        (exante["framework"] == framework)
        & (exante["case"] == case)
        & exante["counterfactual"].isin(["Sum of projects", "Bundle"])
    ][["country", "counterfactual", "compensation_2040_eur_per_year"]]
    samples = cba[cba["counterfactual"].isin(["Sum of projects", "Bundle"])][
        ["country", "counterfactual", "sample", "effect"]
    ]
    result = samples.merge(
        comp,
        on=["country", "counterfactual"],
        how="left",
        validate="many_to_one",
    )
    result["compensation_2040_eur_per_year"] = result[
        "compensation_2040_eur_per_year"
    ].fillna(0.0)
    result["position_after_compensation"] = (
        result["effect"] + result["compensation_2040_eur_per_year"]
    )
    return result


def percentile_design(
    cba: pd.DataFrame,
    exante: pd.DataFrame,
    framework: str,
    case: str,
) -> pd.DataFrame:
    base = net_loser_samples(cba, exante, framework, case)
    rows = []
    for percentile in PERCENTILES:
        q = percentile / 100
        additional = (
            base.groupby(["country", "counterfactual"], as_index=False)
            .agg(quantile=("position_after_compensation", lambda x: x.quantile(q)))
        )
        additional["additional_compensation"] = -additional["quantile"].clip(upper=0)
        additional["percentile"] = percentile

        total = additional.groupby("counterfactual")[
            ["additional_compensation"]
        ].sum().reset_index()
        for _, row in total.iterrows():
            risk = base[
                base["counterfactual"] == row["counterfactual"]
            ].merge(
                additional[
                    (additional["counterfactual"] == row["counterfactual"])
                    & (additional["percentile"] == percentile)
                ][["country", "additional_compensation"]],
                on="country",
                validate="many_to_one",
            )
            risk["designed_position"] = (
                risk["position_after_compensation"] + risk["additional_compensation"]
            )
            loser_rate = (risk["designed_position"] < 0).mean() * 100
            original_rate = (risk["position_after_compensation"] < 0).mean() * 100
            rows.append(
                {
                    "framework": framework,
                    "case": case,
                    "percentile": percentile,
                    "counterfactual": row["counterfactual"],
                    "additional_compensation_total": row["additional_compensation"],
                    "loser_rate_original_percent": original_rate,
                    "loser_rate_designed_percent": loser_rate,
                    "risk_reduction_percentage_points": original_rate - loser_rate,
                }
            )
    return pd.DataFrame(rows)


def write_plot(
    country_stats: pd.DataFrame,
    loser_stats: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot = country_stats[
        country_stats["counterfactual"].isin(["Sum of projects", "Bundle"])
    ].copy()
    if plot.empty:
        return
    top = plot.sort_values("std", ascending=False).head(20)["country"].unique()
    plot = plot[plot["country"].isin(top)]
    fig, ax = plt.subplots(figsize=(12, 7))
    pivot = plot.pivot(index="country", columns="counterfactual", values="std")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Standard deviation of country effect (EUR/year)")
    ax.set_xlabel("Country")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "variability_country_std.png", dpi=160)
    plt.close(fig)

    if not loser_stats.empty:
        pivot = loser_stats.pivot(
            index="country",
            columns=["framework", "case", "counterfactual"],
            values="loser_frequency",
        )
        fig, ax = plt.subplots(figsize=(14, 8))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Net-loser frequency (%)")
        ax.set_xlabel("Country")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "net_loser_frequency.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="analysis_v0")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--cost-file", type=Path, default=Path("inputs/ProjectsCost.xlsx"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    study_dir = args.results_dir / args.study
    output_dir = args.output_dir or study_dir / "bundling_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    project_costs = load_project_costs(args.cost_file)
    welfare = load_welfare_samples(study_dir)
    cba = build_cba_samples(welfare)
    benefits = benefit_distribution(cba)
    europe = set(welfare["country"])

    exante = exante_compensation(benefits, project_costs, europe)
    exante.to_csv(output_dir / "exante_country_compensation.csv", index=False)

    exante_summary = (
        exante.groupby(["framework", "case", "counterfactual"], as_index=False)
        .agg(
            total_receipts_2040_eur_per_year=("positive_compensation_2040_eur_per_year", "sum"),
            total_payments_2040_eur_per_year=("negative_compensation_2040_eur_per_year", "sum"),
            net_total_2040_eur_per_year=("compensation_2040_eur_per_year", "sum"),
        )
    )
    exante_summary.to_csv(output_dir / "exante_summary.csv", index=False)

    cba.to_csv(output_dir / "cba_country_sample_effects.csv", index=False)
    benefits.to_csv(output_dir / "expected_benefits.csv", index=False)

    country_variability, aggregate_variability = variability(cba)
    country_variability.to_csv(output_dir / "variability_country.csv", index=False)
    aggregate_variability.to_csv(output_dir / "variability_aggregate.csv", index=False)

    loser_tables = []
    perfect_tables = []
    percentile_tables = []
    for framework, case in (
        [("1", case) for case in ("1a", "1b", "1c")]
        + [("2", case) for case in ("2a", "2b", "2c")]
    ):
        loser_tables.append(net_loser_analysis(cba, exante, framework, case))
        perfect_tables.append(perfect_expost(cba, exante, framework, case))
        percentile_tables.append(percentile_design(cba, exante, framework, case))

    loser = pd.concat(loser_tables, ignore_index=True)
    perfect = pd.concat(perfect_tables, ignore_index=True)
    percentile = pd.concat(percentile_tables, ignore_index=True)
    loser.to_csv(output_dir / "expost_net_loser_risk.csv", index=False)
    perfect.to_csv(output_dir / "expost_perfect_adjustment.csv", index=False)
    percentile.to_csv(output_dir / "percentile_compensation.csv", index=False)

    write_plot(country_variability, loser, output_dir)
    metadata = {
        "north_sea_countries": sorted(NORTH_SEA),
        "europe_countries": sorted(europe),
        "percentiles": list(PERCENTILES),
        "standalone_scenarios": list(SCENARIOS.keys()),
        "bundle_scenario": "bundle",
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(output_dir)


if __name__ == "__main__":
    main()
