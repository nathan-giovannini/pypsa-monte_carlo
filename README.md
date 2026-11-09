# pypsa-monte_carlo
a workflow to perform a Montecarlo simulation on a pypsa network

## Usage
1. Put the reference network (NetCDF) at the path given in `config/config.yaml` (`network`).
2. Edit `config/config.yaml`: number of samples, seed, solver + options, uncertainties
   (target/selector/mode/distribution).
3. `pixi run run` (= `snakemake -j 4 --configfile config/config.yaml`).

Optional climate years: set `climate_years` (`years`, `profiles_dir`) in the config. Each
`<profiles_dir>/<year>.nc` is a PyPSA network with the same snapshots as the reference; its
`loads_t.p_set` and `generators_t.p_max_pu` replace the reference profiles. The year is sampled
as a design column (`climate_year`) and applied before the other uncertainties.

Stages: reference (expansion disabled) -> LHS design -> one perturbed network per sample
-> optimize each (separate rule) -> `summary.csv` + `welfare_by_country.csv` in `results/<study>/`.
