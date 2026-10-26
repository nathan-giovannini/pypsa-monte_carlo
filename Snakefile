"""
Monte Carlo workflow. Stages:
  1. prepare_reference  reference network with capacity expansion disabled
  2. sample             Latin Hypercube design from the configured uncertainties
  3. build_network      one perturbed network per sample (not optimized)
  4. optimize           separate stage: solve each network
  5. aggregate          cost + welfare summaries

    pixi run snakemake -j 4 --configfile config/config.yaml
"""

configfile: "config/config.yaml"

from montecarlo import steps

RES = f"results/{config['study']}"
IDS = list(range(config["monte_carlo"]["samples"]))


rule all:
    input:
        f"{RES}/summary.csv",
        f"{RES}/welfare_by_country.csv",


rule prepare_reference:
    input:
        config["network"],
    output:
        f"{RES}/reference.nc",
    run:
        steps.step_prepare_reference(input[0], output[0], config.get("fix_expansion", True))


rule sample:
    input:
        f"{RES}/reference.nc",  # only to order after the reference
    output:
        f"{RES}/design.csv",
    params:
        cfg=config,
    run:
        steps.step_sample(params.cfg, output[0])


rule build_network:
    input:
        reference=f"{RES}/reference.nc",
        design=f"{RES}/design.csv",
    output:
        f"{RES}/networks/{{i}}.nc",
    wildcard_constraints:
        i=r"\d+",
    params:
        cfg=config,
    run:
        steps.step_build_network(input.reference, input.design, wildcards.i, params.cfg, output[0])


rule optimize:
    input:
        f"{RES}/networks/{{i}}.nc",
    output:
        network=f"{RES}/optimized/{{i}}/network.nc",
        welfare=f"{RES}/optimized/{{i}}/welfare.csv",
        metrics=f"{RES}/optimized/{{i}}/metrics.json",
    wildcard_constraints:
        i=r"\d+",
    params:
        cfg=config,
    threads: config["solver"].get("threads", 1)
    run:
        steps.step_optimize(input[0], params.cfg, output.network, output.welfare, output.metrics)


rule aggregate:
    input:
        design=f"{RES}/design.csv",
        metrics=expand(f"{RES}/optimized/{{i}}/metrics.json", i=IDS),
        welfare=expand(f"{RES}/optimized/{{i}}/welfare.csv", i=IDS),
    output:
        summary=f"{RES}/summary.csv",
        welfare=f"{RES}/welfare_by_country.csv",
    run:
        steps.step_aggregate(input.design, IDS, RES, output.summary, output.welfare)
