"""Switch the weather-dependent time series of a network to another climate year.

Each climate year is a PyPSA network (NetCDF) stored in `profiles_dir/<year>.nc`.
Only the time series that exist in the reference network are replaced:
  loads_t.p_set          demand
  generators_t.p_max_pu  renewable availability
Values are matched by position: the climate year must have the same number of snapshots as
the reference, but its timestamps may differ (e.g. 2008 vs 2009). The reference's timestamps
are kept. The switch must run before any other timeseries uncertainty multiplies the same attributes.
"""

from pathlib import Path

import pypsa

TIMESERIES = [("loads_t", "p_set"), ("generators_t", "p_max_pu")]


def climate_year_path(profiles_dir, year):
    return Path(profiles_dir) / f"{int(year)}.nc"


def _replace_timeseries(n, source, comp_t, attr):
    ref = getattr(n, comp_t)[attr]
    if ref.empty:
        return
    cols = list(ref.columns)
    src_ts = getattr(source, comp_t)[attr]
    missing = [c for c in cols if c not in src_ts.columns]
    if missing:
        raise ValueError(f"Climate year profile lacks {comp_t}.{attr} for {missing}")
    getattr(n, comp_t)[attr].loc[:, cols] = src_ts.loc[:, cols].values


def apply_climate_year(network, year, profiles_dir):
    """Return a COPY of `network` with demand and renewable profiles of `year`."""
    path = climate_year_path(profiles_dir, year)
    if not path.exists():
        raise FileNotFoundError(f"No profile file for climate year {int(year)}: {path}")
    source = pypsa.Network(str(path))
    if len(source.snapshots) != len(network.snapshots):
        raise ValueError(
            f"{path} has {len(source.snapshots)} snapshots, reference has "
            f"{len(network.snapshots)}; climate years must have the same length (check leap years)."
        )
    n = network.copy()
    for comp_t, attr in TIMESERIES:
        _replace_timeseries(n, source, comp_t, attr)
    return n
