# pypsa-monte_carlo
a workflow to perform a Montecarlo simulation on a pypsa network

## Usage
1. Put the reference network (NetCDF) at the path given in `config/config.yaml` (`network`).
2. Edit `config/config.yaml`: number of samples, seed, solver + options, uncertainties
   (target/selector/mode/distribution).
3. `pixi run run` (= `snakemake -j 4 --configfile config/config.yaml`).

Stages: reference (expansion disabled) -> LHS design -> one perturbed network per sample
-> optimize each (separate rule) -> `summary.csv` + `welfare_by_country.csv` in `results/<study>/`.
