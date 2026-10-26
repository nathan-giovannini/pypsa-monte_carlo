"""Experimental design: Latin Hypercube Sampling mapped to the configured distributions."""

import numpy as np
import pandas as pd
from scipy.stats import lognorm, norm, qmc, triang, uniform

DISTRIBUTIONS = {
    "uniform": ("min", "max"),
    "normal": ("mean", "std"),
    "triangular": ("min", "mode", "max"),
    "lognormal": ("mean", "std"),  # mean/std of the variable itself
}


def _ppf(dist, u):
    """Inverse CDF of the unit-interval sample `u` for a distribution spec."""
    kind = dist["type"]
    if kind == "uniform":
        return uniform.ppf(u, loc=dist["min"], scale=dist["max"] - dist["min"])
    if kind == "normal":
        return norm.ppf(u, loc=dist["mean"], scale=dist["std"])
    if kind == "triangular":
        span = dist["max"] - dist["min"]
        return triang.ppf(u, c=(dist["mode"] - dist["min"]) / span, loc=dist["min"], scale=span)
    if kind == "lognormal":
        m, s = dist["mean"], dist["std"]
        sigma2 = np.log(1 + (s / m) ** 2)
        return lognorm.ppf(u, s=np.sqrt(sigma2), scale=m / np.sqrt(1 + (s / m) ** 2))
    raise ValueError(f"Unknown distribution '{kind}'")


def validate_uncertainty(u):
    name = u.get("name", "<unnamed>")
    dist = u.get("distribution", {})
    kind = dist.get("type")
    if kind not in DISTRIBUTIONS:
        raise ValueError(f"'{name}': distribution.type must be one of {sorted(DISTRIBUTIONS)}")
    missing = [k for k in DISTRIBUTIONS[kind] if k not in dist]
    if missing:
        raise ValueError(f"'{name}': distribution '{kind}' needs {missing}")
    if kind == "uniform" and not dist["min"] < dist["max"]:
        raise ValueError(f"'{name}': uniform needs min < max")
    if kind == "triangular" and not dist["min"] <= dist["mode"] <= dist["max"]:
        raise ValueError(f"'{name}': triangular needs min <= mode <= max")
    if kind == "triangular" and dist["min"] == dist["max"]:
        raise ValueError(f"'{name}': triangular needs min < max")
    if kind in ("normal", "lognormal") and dist["std"] <= 0:
        raise ValueError(f"'{name}': std must be > 0")
    if kind == "lognormal" and dist["mean"] <= 0:
        raise ValueError(f"'{name}': lognormal mean must be > 0")
    if u.get("mode", "relative") not in ("relative", "absolute"):
        raise ValueError(f"'{name}': mode must be 'relative' or 'absolute'")


def make_design(uncertainties, samples, seed, strength=1, optimization=None):
    """Return a DataFrame (samples x uncertainties) of sampled values.

    For `relative` uncertainties the value is a multiplicative factor,
    for `absolute` ones it is an additive delta (same unit as the attribute).
    """
    for u in uncertainties:
        validate_uncertainty(u)
    sampler = qmc.LatinHypercube(d=len(uncertainties), strength=strength,
                                 optimization=optimization, seed=seed)
    unit = sampler.random(n=samples)
    cols = {u["name"]: _ppf(u["distribution"], unit[:, i]) for i, u in enumerate(uncertainties)}
    design = pd.DataFrame(cols)
    design.index.name = "sample"
    return design
