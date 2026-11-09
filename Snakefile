"""
Monte Carlo workflow. Stages:
  1. prepare_reference  reference network with capacity expansion disabled
  2. sample             Latin Hypercube design from the configured uncertainties
  3. build_network      one perturbed network per sample (not optimized)
  4. optimize           separate stage: solve each network
  5. aggregate          cost + welfare summaries

Scenarios (config 'scenarios') are derived from the reference by removing projects; all share
the same design.csv. Results: results/<study>/<scenario>/ ('base' = unmodified reference).

    snakemake results/<study>/<scenario>/summary.csv   # a single scenario
    snakemake -j 4 compare                             # everything + comparison.csv

    pixi run snakemake -j 4 --configfile config/config.yaml
"""

configfile: "config/config.yaml"

from montecarlo import steps

STUDY = f"results/{config['study']}"
SCEN = steps.scenarios(config)
IDS = list(range(config["monte_carlo"]["samples"]))
SC = r"[^/]+"


rule all:
    input:
        expand(f"{STUDY}/{{s}}/summary.csv", s=SCEN),
        expand(f"{STUDY}/{{s}}/welfare_by_country.csv", s=SCEN),
        f"{STUDY}/comparison.csv",


rule compare:
    input:
        f"{STUDY}/comparison.csv",


rule prepare_reference:
    input:
        config["network"],
    output:
        f"{STUDY}/base/reference.nc",
    run:
        steps.step_prepare_reference(input[0], output[0], config.get("fix_expansion", True))


rule derive_network:
    input:
        f"{STUDY}/base/reference.nc",
    output:
        f"{STUDY}/{{scenario}}/reference.nc",
    wildcard_constraints:
        scenario=r"(?!base$)" + SC,
    params:
        removed=lambda wc: SCEN[wc.scenario],
    run:
        steps.step_derive_network(input[0], params.removed, output[0])


# one design for every scenario -> samples are paired across scenarios
rule sample:
    output:
        f"{STUDY}/design.csv",
    params:
        cfg=config,
    run:
        steps.step_sample(params.cfg, output[0])


rule build_network:
    input:
        reference=f"{STUDY}/{{scenario}}/reference.nc",
        design=f"{STUDY}/design.csv",
    output:
        f"{STUDY}/{{scenario}}/networks/{{i}}.nc",
    wildcard_constraints:
        i=r"\d+",
        scenario=SC,
    params:
        cfg=config,
        derived=lambda wc: bool(SCEN[wc.scenario]),
    run:
        steps.step_build_network(
            input.reference, input.design, wildcards.i, params.cfg, output[0], params.derived
        )


rule optimize:
    input:
        f"{STUDY}/{{scenario}}/networks/{{i}}.nc",
    output:
        network=f"{STUDY}/{{scenario}}/optimized/{{i}}/network.nc",
        welfare=f"{STUDY}/{{scenario}}/optimized/{{i}}/welfare.csv",
        metrics=f"{STUDY}/{{scenario}}/optimized/{{i}}/metrics.json",
    wildcard_constraints:
        i=r"\d+",
        scenario=SC,
    params:
        cfg=config,
    threads: config["solver"].get("threads", 1)
    run:
        steps.step_optimize(input[0], params.cfg, output.network, output.welfare, output.metrics)


rule aggregate:
    input:
        design=f"{STUDY}/design.csv",
        metrics=lambda wc: expand(f"{STUDY}/{wc.scenario}/optimized/{{i}}/metrics.json", i=IDS),
        welfare=lambda wc: expand(f"{STUDY}/{wc.scenario}/optimized/{{i}}/welfare.csv", i=IDS),
    output:
        summary=f"{STUDY}/{{scenario}}/summary.csv",
        welfare=f"{STUDY}/{{scenario}}/welfare_by_country.csv",
    wildcard_constraints:
        scenario=SC,
    run:
        steps.step_aggregate(
            input.design, IDS, f"{STUDY}/{wildcards.scenario}", output.summary, output.welfare
        )


rule comparison:
    input:
        base=f"{STUDY}/base/summary.csv",
        others=expand(f"{STUDY}/{{s}}/summary.csv", s=[s for s in SCEN if s != "base"]),
    output:
        f"{STUDY}/comparison.csv",
    run:
        steps.step_compare(
            input.base, {s: f"{STUDY}/{s}/summary.csv" for s in SCEN if s != "base"}, output[0]
        )
