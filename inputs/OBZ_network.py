import pypsa
import pandas as pd

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]



def plot(n):
    import matplotlib.pyplot as plt

    plt.figure()
    n.plot()
    plt.show()
    # import networkx as nx
    # G = n.graph()
    # pos = nx.spring_layout(G)
    # plt.figure()
    # nx.draw_networkx_nodes(G, pos, node_size=5)
    # nx.draw_networkx_edges(G, pos, alpha=0.2)
    # plt.axis("off")
    # plt.show()


    import pypsa
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    # -----------------------------
    # Load network
    # -----------------------------


    snapshot = n.snapshots[0]

    # Results
    marginal_price = n.buses_t.marginal_price.loc[snapshot]
    flows = n.links_t.p0.loc[snapshot]

    # -----------------------------
    # Plot buses
    # -----------------------------
    fig, ax = plt.subplots(figsize=(15, 10))

    ax.scatter(
        n.buses.x,
        n.buses.y,
        s=1000,
        color="lightblue",
        edgecolors="k",
        zorder=3,
    )

    # Bus names + prices
    for bus in n.buses.index:

        x = n.buses.loc[bus, "x"]
        y = n.buses.loc[bus, "y"]

        price = marginal_price[bus]

        ax.text(
            x,
            y + 0.15,
            f"{bus}\n{price:.1f} €/MWh",
            ha="center",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.8),
            zorder=4,
        )

    # -----------------------------
    # Plot links
    # -----------------------------
    for link in n.links.index:

        bus0 = n.links.loc[link, "bus0"]
        bus1 = n.links.loc[link, "bus1"]

        x0, y0 = n.buses.loc[bus0, ["x", "y"]]
        x1, y1 = n.buses.loc[bus1, ["x", "y"]]

        flow = flows[link]
        capacity = n.links.loc[link, "p_nom"]

        loading = abs(flow) / capacity if capacity > 0 else 0

        # Congestion color
        color = "red" if loading >= 0.99 else "blue"

        # Actual flow direction
        if flow >= 0:
            start = (x0, y0)
            end = (x1, y1)
        else:
            start = (x1, y1)
            end = (x0, y0)

        # Arrow
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=80,
            color=color,
            lw=max(1, abs(flow)/abs(flows).max()*5),
            alpha=0.8,
            zorder=2,
        )

        ax.add_patch(arrow)

        # Link flow label
        xm = (x0 + x1) / 2
        ym = (y0 + y1) / 2

        ax.text(
            xm,
            ym,
            f"{flow:.1f} MW",
            color="darkred",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7),
            zorder=5,
        )

    # -----------------------------
    # Cosmetics
    # -----------------------------
    ax.set_title(f"Network state at {snapshot}")
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    plt.show()


n = pypsa.Network()
n.set_snapshots(['1', '2'])

LON = [-1.543051, 2.557000, 4.679751, 6.169059, 6.344649, 4.273000]
LAT = [52.674833, 51.499000, 50.784246, 49.700757, 52.982728, 54.276000]
n.add('Bus', ['GB', 'OBZ_BE', 'BE', 'LU', 'NL', "OBZ_NL"], x=LON, y=LAT)
n.buses.loc[:, 'country'] = ['GB', 'OBZ_BE', 'BE', 'LU', 'NL', "OBZ_NL"]

n.add('Link', "GB-OBZ_BE", bus0='GB', bus1='OBZ_BE', p_nom=1000, p_min_pu = -1.0)
n.add('Link', 'OBZ_BE-BE', bus0='OBZ_BE', bus1='BE', p_nom=1000, p_min_pu = -1.0)
n.add('Link', 'BE-NL', bus0='BE', bus1='NL', p_nom=1500, p_min_pu = -1.0)
n.add('Link', 'BE-LU', bus0='BE', bus1='LU', p_nom=500, p_min_pu = -1.0)
n.add('Link', 'LU-NL', bus0='LU', bus1='NL', p_nom=500, p_min_pu = -1.0)
n.add('Link', "GB-OBZ_NL", bus0='GB', bus1='OBZ_NL', p_nom=2000, p_min_pu = -1.0)
n.add('Link', 'OBZ_NL-NL', bus0='OBZ_NL', bus1='NL', p_nom=2000, p_min_pu = -1.0)

n.add('Generator', 'gen_GB_wind', bus='GB', p_nom=5000, marginal_cost=0.2, carrier="onwind")
n.add('Generator', 'gen_OBZ_BE', bus='OBZ_BE', p_nom=1000, marginal_cost=0.1, carrier="offwind", p_max_pu=[0.5, 0.5])
n.add('Generator', 'gen_LU_OCGT', bus='LU', p_nom=5000, marginal_cost=250, carrier="OCGT")
n.add('Generator', 'gen_NL_solar', bus='NL', p_nom=5500, marginal_cost=0.1, carrier="Solar")
n.add('Generator', 'gen_NL_CCGT', bus='NL', p_nom=6000, marginal_cost=100, carrier="CCGT")
n.add('Generator', 'gen_NL_OCGT', bus='NL', p_nom=10000, marginal_cost=150, carrier="OCGT")
n.add('Generator', 'gen_OBZ_NL', bus='OBZ_NL', p_nom=2000, marginal_cost=0.1, carrier="offwind", p_max_pu=[0.5, 0.5])
n.add('Generator', 'gen_BE_solar', bus='BE', p_nom=2500, marginal_cost=0.1, carrier="solar")
n.add('Generator', 'gen_BE_Nuke', bus='BE', p_nom=2000, marginal_cost=30, carrier="Nuclear")
n.add('Generator', 'gen_BE_OCGT', bus='BE', p_nom=2600, marginal_cost=200, carrier="OCGT")
n.add('Generator', 'loss_of_load_GB', bus='GB', p_nom=10000, marginal_cost=3000, carrier="LOL")
n.add('Generator', 'loss_of_load_BE', bus='BE', p_nom=10000, marginal_cost=3000, carrier="LOL")
n.add('Generator', 'loss_of_load_NL', bus='NL', p_nom=10000, marginal_cost=3000, carrier="LOL")
n.add('Generator', 'loss_of_load_LU', bus='LU', p_nom=10000, marginal_cost=3000, carrier="LOL")


n.add('Load', 'load_GB', bus='GB', p_set = [2000, 2000])
n.add('Load', 'load_LU', bus='LU', p_set = [1000, 1000])
n.add('Load', 'load_BE', bus='BE', p_set = [6000, 6000])
n.add('Load', 'load_NL', bus='NL', p_set = [6000, 6000])



n_cf_0_BE = n.copy()
n_cf_0_BE.remove("Bus", "OBZ_BE")
n_cf_0_BE.remove("Generator", "gen_OBZ_BE")
n_cf_0_BE.remove("Link", ["GB-OBZ_BE", "OBZ_BE-BE"])

n_cf_r_BE = n_cf_0_BE.copy()
n_cf_r_BE.add("Bus", "OBZ_BE", x=2.557000, y=51.499000, country="OBZ_BE")
n_cf_r_BE.add('Generator', 'gen_OBZ_BE', bus='OBZ_BE', p_nom=1000, marginal_cost=0.1, carrier="offwind", p_max_pu=[0.5, 0.5])
n_cf_r_BE.add('Link', 'OBZ_BE-BE', bus0='OBZ_BE', bus1='BE', p_nom=1000, p_min_pu = -1.0)

n_cf_r_i_BE = n_cf_r_BE.copy()
n_cf_r_i_BE.add('Link', 'GB-BE', bus0='GB', bus1='BE', p_nom=1000, p_min_pu = -1.0)


n_cf_0_NL = n.copy()
n_cf_0_NL.remove("Bus", "OBZ_NL")
n_cf_0_NL.remove("Generator", "gen_OBZ_NL")
n_cf_0_NL.remove("Link", ["GB-OBZ_NL", "OBZ_NL-NL"])

n_cf_r_NL = n_cf_0_NL.copy()
n_cf_r_NL.add("Bus", "OBZ_NL", x=4.273000, y=54.276000, country="OBZ_NL")
n_cf_r_NL.add('Generator', 'gen_OBZ_NL', bus='OBZ_NL', p_nom=2000, marginal_cost=0.1, carrier="offwind", p_max_pu=[0.5, 0.5])
n_cf_r_NL.add('Link', 'OBZ_NL-NL', bus0='OBZ_NL', bus1='NL', p_nom=2000, p_min_pu = -1.0)

n_cf_r_i_NL = n_cf_r_NL.copy()
n_cf_r_i_NL.add('Link', 'GB-NL', bus0='GB', bus1='NL', p_nom=2000, p_min_pu = -1.0)

n_cf_0_pool = n_cf_0_BE.copy()
n_cf_0_pool.remove("Bus", "OBZ_NL")
n_cf_0_pool.remove("Generator", "gen_OBZ_NL")
n_cf_0_pool.remove("Link", ["GB-OBZ_NL", "OBZ_NL-NL"])

n_cf_r_pool = n_cf_0_pool.copy()
n_cf_r_pool.add("Bus", "OBZ_BE", x=2.557000, y=51.499000, country="OBZ_BE")
n_cf_r_pool.add("Bus", "OBZ_NL", x=4.273000, y=54.276000, country="OBZ_NL")
n_cf_r_pool.add('Generator', 'gen_OBZ_BE', bus='OBZ_BE', p_nom=1000, marginal_cost=0.1, carrier="offwind", p_max_pu=[0.5, 0.5])
n_cf_r_pool.add('Generator', 'gen_OBZ_NL', bus='OBZ_NL', p_nom=2000, marginal_cost=0.1, carrier="offwind", p_max_pu=[0.5, 0.5])
n_cf_r_pool.add('Link', 'OBZ_BE-BE', bus0='OBZ_BE', bus1='BE', p_nom=1000, p_min_pu = -1.0)
n_cf_r_pool.add('Link', 'OBZ_NL-NL', bus0='OBZ_NL', bus1='NL', p_nom=2000, p_min_pu = -1.0)

n_cf_r_i_pool = n_cf_r_pool.copy()
n_cf_r_i_pool.add('Link', 'GB-BE', bus0='GB', bus1='BE', p_nom=1000, p_min_pu = -1.0)
n_cf_r_i_pool.add('Link', 'GB-NL', bus0='GB', bus1='NL', p_nom=2000, p_min_pu = -1.0)

n.components.carriers.add_missing_carriers()
n.optimize()

n.sanitize()
n.export_to_netcdf("/Users/ng-work/Models/pypsa-monte_carlo/inputs/solved_OBZ.nc")


network_list = [n_cf_0_BE, n_cf_r_BE,n_cf_r_i_BE,n_cf_0_NL,n_cf_r_NL,n_cf_r_i_NL,n_cf_0_pool,n_cf_r_pool,n_cf_r_i_pool]

for n in network_list:
    n.sanitize()
    n.optimize()
    name = namestr(n, globals())[1]
    n.export_to_netcdf(f"/Users/ng-work/Models/pypsa-monte_carlo/inputs/solved_OBZ_{name}.nc")

#n_cf_r_i_BE.optimize()
#plot(n_cf_r_i_BE)