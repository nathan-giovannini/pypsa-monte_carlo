"""Workflow steps called from the Snakefile."""

import json
from pathlib import Path

import pandas as pd
import pypsa

from .climate import apply_climate_year
from .perturb import apply_sample
from .sampling import make_design
from .welfare import bus_country_map, welfare_by_country

EXTENDABLE = {
    "generators": "p_nom_extendable",
    "links": "p_nom_extendable",
    "storage_units": "p_nom_extendable",
    "stores": "e_nom_extendable",
    "lines": "s_nom_extendable",
    "transformers": "s_nom_extendable",
}


CLASS_NAMES = {
    "generators": "Generator", "links": "Link", "lines": "Line", "loads": "Load",
    "storage_units": "StorageUnit", "stores": "Store", "transformers": "Transformer",
    "buses": "Bus",
}


def step_prepare_reference(network_path, output, fix_expansion=True):
    """Load the reference network and forbid any capacity expansion."""
    n = pypsa.Network(network_path)
    if fix_expansion:
        for comp, flag in EXTENDABLE.items():
            df = getattr(n, comp)
            if flag in df.columns:
                df[flag] = False
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    n.export_to_netcdf(output)


CLIMATE_YEAR = "climate_year"


def _design_uncertainties(cfg):
    """Configured uncertainties, preceded by the climate-year choice when enabled."""
    cy = cfg.get("climate_years")
    if not cy:
        return cfg["uncertainties"]
    year_choice = {"name": CLIMATE_YEAR, "distribution": {"type": "choice", "values": cy["years"]}}
    return [year_choice] + cfg["uncertainties"]


def scenarios(cfg):
    """{name: {component: [names]}} of assets removed per scenario; 'base' (nothing removed) is implicit."""
    out = {"base": {}}
    for name, spec in (cfg.get("scenarios") or {}).items():
        if name == "base":
            raise ValueError("'base' is reserved for the reference network")
        out[name] = dict((spec or {}).get("remove") or {})
    return out


def step_derive_network(reference, remove, output):
    """Copy the reference network, dropping the listed projects (component -> names)."""
    n = pypsa.Network(reference)
    for comp, names in remove.items():
        df = getattr(n, comp, None) if comp in CLASS_NAMES else None
        if df is None:
            raise ValueError(f"Unknown component '{comp}' in scenario removal list")
        missing = [x for x in names if x not in df.index]
        if missing:
            raise ValueError(f"Cannot remove {comp} {missing}: not in the network")
        n.remove(CLASS_NAMES[comp], list(names))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    n.export_to_netcdf(output)


def step_compare(base_csv, scenario_csvs, output):
    """Per-sample total system cost of every scenario and its delta vs base."""
    base = pd.read_csv(base_csv, index_col="sample")["total_system_cost"]
    out = pd.DataFrame({"base": base})
    for name, path in scenario_csvs.items():
        c = pd.read_csv(path, index_col="sample")["total_system_cost"]
        out[name] = c
        out[f"{name}_delta"] = c - base
    out.to_csv(output)


def step_sample(cfg, output):
    mc = cfg["monte_carlo"]
    design = make_design(
        _design_uncertainties(cfg), mc["samples"], mc["seed"],
        strength=mc.get("lhs_strength", 1), optimization=mc.get("lhs_optimization"),
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(output)


def step_build_network(reference, design_csv, sample_id, cfg, output, skip_empty=False):
    n = pypsa.Network(reference)
    design = pd.read_csv(design_csv, index_col="sample")
    prefix = cfg.get("country_bus_prefix_length")
    bus_country = bus_country_map(n, prefix_length=prefix) if prefix or "country" in n.buses else None
    row = design.loc[int(sample_id)]
    cy = cfg.get("climate_years")
    if cy:
        # Applied first: the other timeseries uncertainties scale the swapped profiles.
        n = apply_climate_year(n, int(round(row[CLIMATE_YEAR])), cy["profiles_dir"])
    n = apply_sample(n, cfg["uncertainties"], row, bus_country, skip_empty)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    n.export_to_netcdf(output)


def step_optimize(network_path, cfg, out_network, out_welfare, out_metrics):
    """Solve one network. A failed solve does not stop the workflow: its outputs are written
    as empty placeholders and metrics.json gets status 'failed' (aggregate skips it)."""
    solver = cfg["solver"]
    n = pypsa.Network(network_path)
    for p in (out_network, out_welfare, out_metrics):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    try:
        status, condition = n.optimize(
            solver_name=solver["name"],
            **({"solver_options": solver["options"]} if solver.get("options") else {}),
        )
        error = None if status == "ok" else f"{status} / {condition}"
    except Exception as e:  # solver crash, license error, ...
        status, condition, error = "failed", "exception", f"{type(e).__name__}: {e}"

    if error is not None:
        Path(out_network).touch()
        Path(out_welfare).touch()
        with open(out_metrics, "w") as f:
            json.dump({"total_system_cost": None, "status": "failed", "condition": error}, f)
        return

    welfare = welfare_by_country(
        n, prefix_length=cfg.get("country_bus_prefix_length"), voll=cfg.get("voll", 3000)
    )
    n.export_to_netcdf(out_network)
    welfare.to_csv(out_welfare)
    with open(out_metrics, "w") as f:
        json.dump({"total_system_cost": float(n.objective), "status": "ok",
                   "condition": str(condition)}, f)


def step_aggregate(design_csv, sample_ids, res, out_metrics, out_welfare):
    design = pd.read_csv(design_csv, index_col="sample")
    rows, welfare, failed = [], [], []
    for i in sample_ids:
        with open(f"{res}/optimized/{i}/metrics.json") as f:
            m = json.load(f)
        rows.append({"sample": i, "status": m["status"], "total_system_cost": m["total_system_cost"]})
        if m["status"] != "ok":
            failed.append(i)
            continue
        w = pd.read_csv(f"{res}/optimized/{i}/welfare.csv", index_col=0)
        w.insert(0, "sample", i)
        welfare.append(w.reset_index(names="country"))
    if failed:
        print(f"{res}: {len(failed)} failed solve(s), samples {failed} (see status column)")
    metrics = pd.DataFrame(rows).set_index("sample").join(design)
    metrics.to_csv(out_metrics)
    if welfare:
        pd.concat(welfare).to_csv(out_welfare, index=False)
    else:
        pd.DataFrame(columns=["country", "sample"]).to_csv(out_welfare, index=False)
