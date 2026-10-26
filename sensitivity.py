"""Simple one-at-a-time sensitivity analysis over a PyPSA network.

Usage: python sensitivity.py network.nc
"""
import sys

import pandas as pd
import pypsa

# (component, attribute, scaling factors applied to the base values)
PARAMS = [
    ("Generator", "marginal_cost", [0.8, 1.0, 1.2]),
    ("Generator", "p_nom", [0.8, 1.0, 1.2]),
    ("Load", "p_set", [0.9, 1.0, 1.1]),
]


def run(path):
    base = pypsa.Network(path)
    results = []
    for comp, attr, factors in PARAMS:
        for f in factors:
            n = base.copy()
            df = getattr(n, comp.lower() + "s")
            if attr in df:
                df[attr] = df[attr] * f
            elif attr in getattr(n, comp.lower() + "s_t"):
                getattr(n, comp.lower() + "s_t")[attr] *= f
            n.optimize()
            results.append(
                {"component": comp, "attribute": attr, "factor": f,
                 "objective": n.objective}
            )
    return pd.DataFrame(results)


if __name__ == "__main__":
    out = run(sys.argv[1])
    print(out)
    out.to_csv("sensitivity_results.csv", index=False)
