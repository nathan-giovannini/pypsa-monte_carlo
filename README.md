# pypsa-monte_carlo
a workflow to perform a Montecarlo simulation on a pypsa network

## Usage
1. Put the reference network (NetCDF) at the path given in `config/config.yaml` (`network`).
2. Edit `config/config.yaml`: number of samples, seed, solver + options, uncertainties
   (target/selector/mode/distribution).
3. `pixi run run` (= `snakemake -j 4 --configfile config/config.yaml`).

Set `per_asset: true` on an uncertainty to sample each selected asset independently.
By default, selected assets share one sample.

Optional climate years: set `climate_years` (`years`, `profiles_dir`) in the config. Each
`<profiles_dir>/<year>.nc` is a PyPSA network with the same snapshots as the reference; its
`loads_t.p_set` and `generators_t.p_max_pu` replace the reference profiles. The year is sampled
as a design column (`climate_year`) and applied before the other uncertainties.

Stages: reference (expansion disabled) -> LHS design -> one perturbed network per sample
-> optimize each (separate rule) -> `summary.csv` + `welfare_by_country.csv` in `results/<study>/`.

## Scenarios (derived networks)
List derived networks under `scenarios:` in `config/config.yaml`; each removes named projects
from the reference (`remove: {links: [...], generators: [...]}`). All scenarios and `base` share
one LHS design, so sample *i* is comparable across scenarios. Uncertainties that target removed
assets are skipped in that scenario.

- Everything: `pixi run run` -> `results/<study>/<scenario>/summary.csv` + `results/<study>/comparison.csv`
  (per-sample cost and delta vs base).
- One scenario only: `pixi run snakemake -j 4 results/<study>/<scenario>/summary.csv`.
- Preview: `pixi run snakemake -n`.

## Failed solves
A solve that fails (infeasible, solver error, crash) does not stop the workflow. Its sample keeps a
`status: failed` row in `summary.csv` (empty cost) and is left out of `welfare_by_country.csv` and
the cost deltas in `comparison.csv`. Rerun after fixing the cause; Snakemake only redoes unfinished samples
if you delete their `optimized/<i>/metrics.json` (or use `--rerun-incomplete`/`--forcerun optimize`).
