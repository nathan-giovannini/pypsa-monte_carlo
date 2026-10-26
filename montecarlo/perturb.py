"""Apply sampled uncertainty values to a copy of the reference network.

An uncertainty states:
  target    component / attribute / kind (static | timeseries)
  selector  names, carrier, countries, query -- combined with AND
  mode      relative: value * sample
            absolute: value + sample, distributed over the selected assets
                      proportionally to their current value
"""

import pandas as pd


def _country_mask(df, bus_country, countries):
    if "bus" in df.columns:
        return df["bus"].map(bus_country).isin(countries)
    if "bus0" in df.columns and "bus1" in df.columns:
        return df["bus0"].map(bus_country).isin(countries) | df["bus1"].map(bus_country).isin(countries)
    raise ValueError("Cannot determine country: component has neither 'bus' nor 'bus0'/'bus1'.")


def select_assets(static_df, selector, bus_country=None):
    mask = pd.Series(True, index=static_df.index)
    selector = selector or {}
    if selector.get("names"):
        mask &= static_df.index.isin(selector["names"])
    if selector.get("carrier"):
        carriers = selector["carrier"]
        carriers = [carriers] if isinstance(carriers, str) else carriers
        if "carrier" not in static_df.columns:
            raise ValueError("selector.carrier used but component has no 'carrier' column")
        mask &= static_df["carrier"].isin(carriers)
    if selector.get("countries"):
        if bus_country is None:
            raise ValueError("selector.countries needs a bus->country map")
        mask &= _country_mask(static_df, bus_country, selector["countries"])
    if selector.get("query"):
        mask &= static_df.index.isin(static_df.query(selector["query"]).index)
    return static_df.index[mask.values]


def _apply_one(n, u, value, bus_country):
    component, attribute = u["target"]["component"], u["target"]["attribute"]
    kind = u["target"].get("kind", "static")
    relative = u.get("mode", "relative") == "relative"

    static_df = getattr(n, component)
    assets = select_assets(static_df, u.get("selector"), bus_country)
    if len(assets) == 0:
        raise ValueError(f"Uncertainty '{u['name']}' selected no {component}")

    if kind == "timeseries":
        ts = getattr(n, f"{component}_t")[attribute]
        cols = [a for a in assets if a in ts.columns]
        if not cols:
            raise ValueError(f"None of the selected {component} has a time series '{attribute}'")
        if relative:
            ts.loc[:, cols] = ts.loc[:, cols] * value
        else:
            weights = ts[cols].sum(axis=0)
            share = weights / weights.sum() if weights.sum() > 0 else pd.Series(1 / len(cols), index=cols)
            ts.loc[:, cols] = ts.loc[:, cols] + share * value
    else:
        if attribute not in static_df.columns:
            raise ValueError(f"n.{component} has no static column '{attribute}'")
        vals = static_df.loc[assets, attribute].astype(float)
        if relative:
            static_df.loc[assets, attribute] = vals * value
        else:
            share = vals / vals.sum() if vals.sum() > 0 else pd.Series(1 / len(assets), index=assets)
            static_df.loc[assets, attribute] = vals + share * value


def apply_sample(network, uncertainties, sample, bus_country=None):
    """Return a COPY of `network` with every uncertainty applied; `sample` maps name -> value."""
    n = network.copy()
    for u in uncertainties:
        _apply_one(n, u, float(sample[u["name"]]), bus_country)
    return n
