import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature


var = "PRCP"
month = 5

for var in ["T2M", "PRCP"]:
    # Load datasets
    files = {
        "Lead 1–3": f"MME_skill_scores_{var}_{month}_l1-l3.nc",
        "Lead 2–4": f"MME_skill_scores_{var}_{month}_l2-l4.nc",
        "Lead 3–5": f"MME_skill_scores_{var}_{month}_l3-l5.nc",
    }

    datasets = {label: xr.open_dataset(f) for label, f in files.items()}

    # Plot
    fig, axes = plt.subplots(
        1, 3, figsize=(18, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    for ax, (label, ds) in zip(axes, datasets.items()):
        rpss = ds["rank_probability_skill_score"]
        X = ds["X"].values
        Y = ds["Y"].values

        p = ax.pcolormesh(
            X,
            Y,
            rpss.values,
            cmap="RdYlGn",
            vmin=-0.7,
            vmax=0.7,
            transform=ccrs.PlateCarree(),
        )

        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
        ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", color="gray")
        ax.set_title(label, fontsize=12, fontweight="bold")

    fig.suptitle(
        f"MME Skill Scores — {var}, Forecast Month - {month}",
        fontsize=14,
        fontweight="bold",
    )

    # Adjust layout first, then add colorbar axes manually
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom for cbar

    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(p, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Rank Probability Skill Score (RPSS)", fontsize=11)

    plt.savefig(
        f"MME_skill_scores_{var}_{month}_panels.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Saved.")
