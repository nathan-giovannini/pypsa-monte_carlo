"""
Computes market-welfare metrics by country from a solved PyPSA network.

Welfare is decomposed as:
  - producer surplus per generator: (price_at_bus - marginal_cost) * dispatch
  - producer surplus per storage unit: (price_at_bus * dispatch) -
    (marginal_cost * discharge), i.e. charging is only "paid for" via the
    price term (dispatch is negative while charging), discharge cost only
    applies to the positive (discharging) part of dispatch
  - consumer surplus per load: (VOLL - price_at_bus) * demand
  - congestion rent per line: (price_bus1 - price_bus0) * flow, split
    50/50 between the two connected countries
  - congestion rent per link (e.g. HVDC interconnectors): revenue from
    power delivered at bus1 minus cost of power withdrawn at bus0
    (accounts for link efficiency losses), split 50/50 between the two
    connected countries

Consumer surplus needs a value-of-lost-load (VOLL) assumption because
demand in PyPSA is usually modelled as fixed/inelastic (p_set), so there's
no explicit willingness-to-pay curve to integrate. If your model has
elastic demand or bids, replace consumer_surplus_by_country() accordingly.

All of this requires nodal marginal prices (network.buses_t.marginal_price),
so the network must already be solved before calling these functions.
"""

import pandas as pd

DEFAULT_VOLL = 3000  # EUR/MWh -- adjust to match your model's assumptions

WELFARE_COMPONENT_COLUMNS = [
    "producer_surplus",
    "storage_surplus",
    "consumer_surplus",
    "line_congestion_rent",
    "link_congestion_rent",
]


def bus_country_map(network, prefix_length=None, country_column=None):
    """
    Map every bus to a country code.

    Priority:
    1. `country_column`, if given and present on network.buses
    2. a 'country' column on network.buses, if present
    3. the first `prefix_length` characters of the bus name
       (e.g. 'DE1' -> 'DE' with prefix_length=2)
    """
    buses = network.buses
    col = country_column or ("country" if "country" in buses.columns else None)
    if col is not None:
        return buses[col]

    if prefix_length is None:
        raise ValueError(
            "No country column found on network.buses; pass prefix_length "
            "to derive country codes from bus names (e.g. 'DE1' -> 'DE')."
        )
    return buses.index.to_series().str[:prefix_length]


def producer_surplus_by_country(network, bus_country):
    gens = network.generators
    if gens.empty:
        return pd.Series(dtype=float)

    country = gens["bus"].map(bus_country)
    dispatch = network.generators_t.p  # snapshots x generators
    price_at_bus = network.buses_t.marginal_price[gens["bus"]]
    price_at_bus.columns = gens.index

    revenue = (price_at_bus * dispatch).sum()
    cost = (gens["marginal_cost"].reindex(dispatch.columns).values * dispatch).sum()
    surplus = revenue - cost
    return surplus.groupby(country).sum()

def storage_surplus_by_country(network, bus_country):
    """
    Producer surplus from storage_units. Revenue/cost of charging is
    captured through the price term (dispatch is negative while charging);
    marginal_cost is only applied to the discharging (positive) part of
    dispatch, matching PyPSA's convention that marginal_cost is a cost per
    MWh of discharge.
    """
    storage = network.storage_units
    if storage.empty:
        return pd.Series(dtype=float)

    country = storage["bus"].map(bus_country)
    dispatch = network.storage_units_t.p  # snapshots x storage units, +discharge/-charge
    price_at_bus = network.buses_t.marginal_price[storage["bus"]]
    price_at_bus.columns = storage.index

    revenue = (price_at_bus * dispatch).sum()
    discharge_cost = (
        storage["marginal_cost"].reindex(dispatch.columns).values * dispatch.clip(lower=0)
    ).sum()
    surplus = revenue - discharge_cost
    return surplus.groupby(country).sum()

def consumer_surplus_by_country(network, bus_country, voll=DEFAULT_VOLL):
    loads = network.loads
    if loads.empty:
        return pd.Series(dtype=float)

    country = loads["bus"].map(bus_country)
    demand = network.loads_t.p_set
    price_at_bus = network.buses_t.marginal_price[loads["bus"]]
    price_at_bus.columns = loads.index

    surplus = ((voll - price_at_bus) * demand).sum()
    return surplus.groupby(country).sum()


def congestion_rent_by_country(network, bus_country):
    """
    Rent on AC transmission lines, split 50/50 between the two connected
    countries. Adjust this allocation rule if your analysis needs a
    different convention (e.g. attribute fully to the importing country).
    """
    lines = network.lines
    if lines.empty:
        return pd.Series(dtype=float)

    flow = network.lines_t.p0  # snapshots x lines, flow leaving bus0
    # zone0/zone1 are set by zonal (FBMC) models; plain PyPSA uses bus0/bus1
    end0 = lines["zone0"] if "zone0" in lines.columns else lines["bus0"]
    end1 = lines["zone1"] if "zone1" in lines.columns else lines["bus1"]
    price0 = network.buses_t.marginal_price[end0]
    price1 = network.buses_t.marginal_price[end1]
    price0.columns = lines.index
    price1.columns = lines.index

    rent = ((price1 - price0) * flow).sum()  # per line, summed over snapshots


    country0 = end0.map(bus_country)
    country1 = end1.map(bus_country)


    half = rent / 2
    rent_by_country = pd.concat([
        half.groupby(country0).sum(),
        half.groupby(country1).sum(),
    ]).groupby(level=0).sum()

    return rent_by_country

def link_congestion_rent_by_country(network, bus_country):
    """
    Rent on links (e.g. HVDC interconnectors). Uses p0 (power withdrawn at
    bus0) and p1 (power withdrawn at bus1, negative of what's injected --
    PyPSA's convention, which already accounts for link efficiency losses
    since p1 = -efficiency * p0). Rent = revenue from power delivered at
    bus1 minus cost of power withdrawn at bus0, split 50/50 between the
    two connected countries. If the link has a marginal_cost, it's
    subtracted as an operating cost on the withdrawn (bus0) flow.
    """
    links = network.links
    if links.empty:
        return pd.Series(dtype=float)

    p0 = network.links_t.p0  # snapshots x links, withdrawn at bus0
    p1 = network.links_t.p1  # snapshots x links, withdrawn at bus1 (negative = injected)
    price0 = network.buses_t.marginal_price[links["bus0"]]
    price1 = network.buses_t.marginal_price[links["bus1"]]
    price0.columns = links.index
    price1.columns = links.index

    rent = (-(price0 * p0 + price1 * p1)).sum()  # per link, summed over snapshots

    if "marginal_cost" in links.columns:
        rent = rent - (links["marginal_cost"].reindex(p0.columns).values * p0).sum()

    country0 = links["bus0"].map(bus_country)
    country1 = links["bus1"].map(bus_country)

    half = rent / 2
    rent_by_country = pd.concat([
        half.groupby(country0).sum(),
        half.groupby(country1).sum(),
    ]).groupby(level=0).sum()

    return rent_by_country


def welfare_by_country(network, bus_country=None, prefix_length=None, voll=DEFAULT_VOLL):
    """
     Welfare by country, kept disaggregated by component so the breakdown
     can be analyzed later (not just the total). Returns a DataFrame
     indexed by country code with columns:

         producer_surplus, storage_surplus, consumer_surplus,
         line_congestion_rent, link_congestion_rent, total

     `total` is the row-wise sum of the component columns.
     """
    if bus_country is None:
        bus_country = bus_country_map(network, prefix_length=prefix_length)

    components = {
        "producer_surplus": producer_surplus_by_country(network, bus_country),
        "storage_surplus": storage_surplus_by_country(network, bus_country),
        "consumer_surplus": consumer_surplus_by_country(network, bus_country, voll=voll),
        "line_congestion_rent": congestion_rent_by_country(network, bus_country),
        "link_congestion_rent": link_congestion_rent_by_country(network, bus_country),
    }

    df = pd.DataFrame(components).reindex(columns=WELFARE_COMPONENT_COLUMNS).fillna(0)
    df["total"] = df.sum(axis=1)
    return df.sort_index()


def welfare_shares(welfare_series):
    total = welfare_series.sum()
    if total == 0:
        return welfare_series * 0
    return welfare_series / total


def welfare_shift(baseline_welfare, perturbed_welfare):
    """
    Sum over countries of the absolute change in welfare (EUR) between the
    baseline and the perturbed scenario.
    """
    countries = baseline_welfare.index.union(perturbed_welfare.index)
    base = baseline_welfare.reindex(countries, fill_value=0)
    pert = perturbed_welfare.reindex(countries, fill_value=0)
    return (pert - base).abs().sum()
