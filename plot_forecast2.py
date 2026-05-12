import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd


thresholds = [40, 45, 50, 60, 70]


def assign_level(pn):
    pb = abs(pn)
    sign = np.sign(pn)
    if pb >= 70:
        return 5 * sign
    elif pb >= 60:
        return 4 * sign
    elif pb >= 50:
        return 3 * sign
    elif pb >= 45:
        return 2 * sign
    elif pb >= 40:
        return 1 * sign
    else:
        return 0


color_dict = {
    -5: "#6B1C00",  # dark brown
    -4: "#B84200",  # brown
    -3: "#D96B00",  # orange-brown
    -2: "#F0A500",  # orange
    -1: "#F5D000",  # yellow
    0: "#FFFFFF",  # white = climatological
    1: "#C8F0C8",  # very light green
    2: "#78C878",  # light green
    3: "#3DA63D",  # medium green
    4: "#1A7A1A",  # dark green
    5: "#0000CD",  # blue (70+)
    6: "#C0C0C0",  # gray = normal
}

year = 2026
month = 5

for var, ivar in [("T2M", "tref"), ("PRCP", "prec")]:
    for lead in range(3):
        ds = xr.open_dataset(
            f"viz_out/seasonal/{ivar}/live/{ivar}-KMMEH-3monthly-frequencies.nc"
        )
        time = ds["time"].values[lead]
        time_stamp = pd.Timestamp(time).strftime("%Y-%m")
        print(time_stamp)
        prob = ds["tercile_probability"].squeeze()[lead, :, :].values
        dprob = ds["decile_probability"].squeeze()[lead, :, :].values
        lats = ds["lat"].values
        lons = ds["lon"].values

        if np.nanmax(prob) <= 1.0:
            prob *= 100

        hatch_masked = np.ma.masked_where(dprob == 0, np.ones_like(dprob))

        level_map = np.vectorize(assign_level)(prob)

        levels_order = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
        colors = [color_dict[l] for l in levels_order]
        levels_order_rev = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, 6]
        colors_rev = [color_dict[l] for l in levels_order_rev]

        if var == "T2M":
            cmap = mcolors.ListedColormap(colors_rev)
        else:
            cmap = mcolors.ListedColormap(colors)
        bounds = [-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots(
            figsize=(12, 7), subplot_kw={"projection": ccrs.PlateCarree()}
        )

        p = ax.pcolormesh(
            lons, lats, level_map, cmap=cmap, norm=norm, transform=ccrs.PlateCarree()
        )

        ax.contourf(
            lons,
            lats,
            np.abs(dprob),
            levels=[0.3, 1.0],  # adjust to your data range where you want hatching
            hatches=["..."],
            colors="none",  # transparent fill
            transform=ccrs.PlateCarree(),
        )

        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")
        ax.add_feature(cfeature.OCEAN, facecolor="#DDEEFF", zorder=0)
        ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--", color="gray")

        ax.set_title(
            f"{var} Forecast for {time_stamp}; initialized at {month}/{year}",
            fontsize=13,
            fontweight="bold",
        )

        fig.subplots_adjust(bottom=0.22)

        cmap_red = mcolors.ListedColormap(
            ["#F5D000", "#F0A500", "#D96B00", "#B84200", "#6B1C00"]
        )
        cmap_blue = mcolors.ListedColormap(
            ["#C8F0C8", "#78C878", "#3DA63D", "#1A7A1A", "#0000CD"]
        )

        if var == "T2M":
            cmap_below = cmap_blue
            cmap_above = cmap_red
        else:
            cmap_below = cmap_red
            cmap_above = cmap_blue

        # --- Below Normal bar ---
        ax_below = fig.add_axes([0.08, 0.07, 0.28, 0.04])
        norm_below = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap_below.N)
        cb_below = matplotlib.colorbar.ColorbarBase(
            ax_below, cmap=cmap_below, norm=norm_below, orientation="horizontal"
        )
        cb_below.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
        cb_below.set_ticklabels(["40", "45", "50", "60", "70+"])
        ax_below.set_title("Below Normal", fontsize=9, fontweight="bold", pad=3)

        # --- Normal bar ---
        ax_normal = fig.add_axes([0.42, 0.07, 0.08, 0.04])
        cmap_normal = mcolors.ListedColormap(["#C0C0C0"])
        norm_normal = mcolors.BoundaryNorm([0, 1], cmap_normal.N)
        cb_normal = matplotlib.colorbar.ColorbarBase(
            ax_normal, cmap=cmap_normal, norm=norm_normal, orientation="horizontal"
        )
        cb_normal.set_ticks([0.5])
        cb_normal.set_ticklabels(["40+"])
        ax_normal.set_title("Normal", fontsize=9, fontweight="bold", pad=3)

        # --- Above Normal bar ---
        ax_above = fig.add_axes([0.60, 0.07, 0.28, 0.04])
        norm_above = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap_above.N)
        cb_above = matplotlib.colorbar.ColorbarBase(
            ax_above, cmap=cmap_above, norm=norm_above, orientation="horizontal"
        )
        cb_above.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
        cb_above.set_ticklabels(["40", "45", "50", "60", "70+"])
        ax_above.set_title("Above Normal", fontsize=9, fontweight="bold", pad=3)

        # Bottom label
        fig.text(
            0.5,
            0.02,
            "Probability (%) of Most Likely Category",
            ha="center",
            fontsize=10,
        )

        plt.savefig(
            f"forecast_{var}_{time_stamp}_{year}-{month:02d}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Saved: forecast_{var}_{time_stamp}_{year}-{month:02d}.png")
