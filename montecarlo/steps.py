"""Workflow steps called from the Snakefile."""

import json
from pathlib import Path

import pandas as pd
import pypsa

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


def step_sample(cfg, output):
    mc = cfg["monte_carlo"]
    design = make_design(
        cfg["uncertainties"], mc["samples"], mc["seed"],
        strength=mc.get("lhs_strength", 1), optimization=mc.get("lhs_optimization"),
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(output)


def step_build_network(reference, design_csv, sample_id, cfg, output):
    n = pypsa.Network(reference)
    design = pd.read_csv(design_csv, index_col="sample")
    prefix = cfg.get("country_bus_prefix_length")
    bus_country = bus_country_map(n, prefix_length=prefix) if prefix or "country" in n.buses else None
    n = apply_sample(n, cfg["uncertainties"], design.loc[int(sample_id)], bus_country)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    n.export_to_netcdf(output)


def step_optimize(network_path, cfg, out_network, out_welfare, out_metrics):
    solver = cfg["solver"]
    n = pypsa.Network(network_path)
    status, condition = n.optimize(
        solver_name=solver["name"],
        **({"solver_options": solver["options"]} if solver.get("options") else {}),
    )
    if status != "ok":
        raise RuntimeError(f"Optimization failed: {status} / {condition}")

    welfare = welfare_by_country(
        n, prefix_length=cfg.get("country_bus_prefix_length"), voll=cfg.get("voll", 3000)
    )
    for p in (out_network, out_welfare, out_metrics):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    n.export_to_netcdf(out_network)
    welfare.to_csv(out_welfare)
    with open(out_metrics, "w") as f:
        json.dump({"total_system_cost": float(n.objective), "status": status,
                   "condition": str(condition)}, f)


def step_aggregate(design_csv, sample_ids, res, out_metrics, out_welfare):
    design = pd.read_csv(design_csv, index_col="sample")
    rows, welfare = [], []
    for i in sample_ids:
        with open(f"{res}/optimized/{i}/metrics.json") as f:
            m = json.load(f)
        rows.append({"sample": i, "total_system_cost": m["total_system_cost"]})
        w = pd.read_csv(f"{res}/optimized/{i}/welfare.csv", index_col=0)
        w.insert(0, "sample", i)
        welfare.append(w.reset_index(names="country"))
    metrics = pd.DataFrame(rows).set_index("sample").join(design)
    metrics.to_csv(out_metrics)
    pd.concat(welfare).to_csv(out_welfare, index=False)
