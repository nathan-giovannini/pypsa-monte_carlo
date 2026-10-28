"""
Post-processing of the cost-benefit analysis over Monte Carlo runs.

SEW(project, country, sample) = welfare_reference - welfare_counterfactual,
paired on the sample index (countries missing from a run count as 0).
Net benefit = SEW - cost * cost_share[country] when cost info is given.
"""

import warnings
from pathlib import Path

import pandas as pd
import yaml

KEYS = ["project", "sample", "country"]


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    projects = cfg.get("projects") or {}
    for name, p in projects.items():
        unknown = [c for c in p.get("children", []) if c not in projects]
        if unknown:
            raise ValueError(f"Project {name!r}: unknown children {unknown}")
        if not p.get("counterfactual") and not p.get("children"):
            raise ValueError(f"Project {name!r} needs a counterfactual or children")
        share = p.get("cost_share")
        if share and abs(sum(share.values()) - 1) > 1e-6:
            raise ValueError(f"Project {name!r}: cost_share must sum to 1")
        if share and p.get("cost") is None:
            raise ValueError(f"Project {name!r}: cost_share given without cost")
    _check_acyclic(projects)
    return cfg


def _check_acyclic(projects):
    state = {}

    def visit(n):
        if state.get(n) == 1:
            raise ValueError(f"Cycle in project hierarchy at {n!r}")
        if state.get(n) == 2:
            return
        state[n] = 1
        for c in projects[n].get("children", []):
            visit(c)
        state[n] = 2

    for n in projects:
        visit(n)


def _wide(path, root):
    df = pd.read_csv(Path(root) / path)
    return df.pivot_table(index="sample", columns="country", values="total", aggfunc="sum")


def sew_wide(reference, counterfactual):
    """Paired SEW as a sample x country frame."""
    if not reference.index.equals(counterfactual.index):
        warnings.warn("Sample indices differ between reference and counterfactual")
    return reference.sub(counterfactual, fill_value=0)


def project_sew(cfg, root="."):
    """Long table (project, sample, country, sew) for every project."""
    projects = cfg["projects"]
    ref = _wide(cfg["reference"], root)
    wide = {
        n: sew_wide(ref, _wide(p["counterfactual"], root))
        for n, p in projects.items()
        if p.get("counterfactual")
    }
    # Bundles without own counterfactual: sum of children (resolved recursively)
    def resolve(n):
        if n not in wide:
            wide[n] = sum(
                (resolve(c) for c in projects[n]["children"]),
                start=pd.DataFrame(0.0, index=ref.index, columns=ref.columns),
            ).fillna(0)
        return wide[n]

    for n in projects:
        resolve(n)

    df = pd.concat(
        {n: w.rename_axis(columns="country").stack().rename("sew") for n, w in wide.items()},
        names=["project"],
    ).reset_index()
    countries = cfg.get("countries")
    if countries:
        df = df[df.country.isin(countries)]
    return df[KEYS + ["sew"]].reset_index(drop=True)


def add_costs(sew, cfg):
    """Add cost borne per country and net_benefit (NaN where cost is unknown)."""
    cost = {}
    for n, p in cfg["projects"].items():
        for c, s in (p.get("cost_share") or {}).items():
            cost[(n, c)] = float(p["cost"]) * s
    out = sew.copy()
    key = pd.MultiIndex.from_frame(out[["project", "country"]])
    has_cost = {n for n, p in cfg["projects"].items() if p.get("cost_share")}
    out["cost"] = pd.Series(cost).reindex(key).fillna(0.0).values
    known = out["project"].isin(has_cost)
    out["net_benefit"] = (out["sew"] - out["cost"]).where(known)
    return out


def synergy(sew, cfg):
    """Per sample/country: SEW(bundle) - sum of SEW(children) for bundles."""
    p = sew.pivot_table(index=["sample", "country"], columns="project", values="sew")
    rows = {}
    for n, d in cfg["projects"].items():
        if d.get("children") and d.get("counterfactual"):
            rows[n] = p[n] - p[d["children"]].sum(axis=1)
    if not rows:
        return pd.DataFrame(columns=KEYS + ["synergy"])
    return (
        pd.concat(rows, names=["project"]).rename("synergy").reset_index()[KEYS + ["synergy"]]
    )


def summarize(df, value):
    g = df.groupby(["project", "country"])[value]
    out = g.agg(mean="mean", std="std", p5=lambda s: s.quantile(0.05),
                p50="median", p95=lambda s: s.quantile(0.95))
    out["prob_positive"] = g.apply(lambda s: (s > 0).mean())
    return out.reset_index()


def rank_per_country(summary):
    """Rank projects within each country: 1 = best expected value / lowest risk."""
    s = summary.copy()
    g = s.groupby("country")
    s["rank_mean"] = g["mean"].rank(ascending=False, method="min")
    s["rank_p5"] = g["p5"].rank(ascending=False, method="min")  # downside risk
    s["rank_std"] = g["std"].rank(ascending=True, method="min")
    return s.sort_values(["country", "rank_mean"]).reset_index(drop=True)


def run(config_path, root=".", outdir=None):
    cfg = load_config(config_path)
    sew = add_costs(project_sew(cfg, root), cfg)
    syn = synergy(sew, cfg)
    tables = {
        "cba_sew_samples": sew,
        "cba_summary_sew": rank_per_country(summarize(sew, "sew")),
        "cba_synergy_samples": syn,
        "cba_summary_synergy": summarize(syn.rename(columns={"synergy": "v"}), "v")
        if len(syn) else syn,
    }
    if sew["net_benefit"].notna().any():
        tables["cba_summary_net_benefit"] = rank_per_country(
            summarize(sew.dropna(subset=["net_benefit"]), "net_benefit")
        )
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        for k, t in tables.items():
            t.to_csv(Path(outdir) / f"{k}.csv", index=False)
    return tables


if __name__ == "__main__":
    import sys

    run(sys.argv[1], outdir=sys.argv[2] if len(sys.argv) > 2 else None)
