from pathlib import Path
import cdsapi
import xarray as xr

SYSTEMS = {"ecmwf": ["51"], "ncep": ["2"]}


def _years(model: str, start_year: int, end_year: int) -> list[str]:
    return [str(year) for year in range(start_year, end_year + 1)]


def _renames(var: str):
    renames = {
        "latitude": "Y",
        "longitude": "X",
        "time": "T",
    }
    if var == "PRCP":
        renames["tprate"] = "PRCP"
    if var == "T2M":
        renames["2t"] = "T2M"

    return renames


var_name = {
    "T2M": "2m_temperature",
    "PRCP": "total_precipitation",
}


def download_data(
    model: str,
    varname: str,
    start_year: int,
    end_year: int,
    month: int,
    lead: int,
    area: tuple[float, float, float, float],
    system: str | None = None,
    odir: Path = Path("./"),
) -> tuple[Path, str]:
    if not system:
        system = SYSTEMS[model][0]

    dataset = "seasonal-monthly-single-levels"
    request = {
        "originating_centre": model,
        "system": system,
        "variable": [var_name[varname]],
        "product_type": ["monthly_mean"],
        "year": _years(model, start_year, end_year),
        "month": [str(month)],
        "leadtime_month": [str(lead)],
        "data_format": "grib",
        "area": area,
    }

    carea = "_".join(map(str, area))
    output = (
        odir / f"{model}_{varname}_{start_year}_{end_year}_{month}_{lead}_{carea}.nc"
    )
    tmp_output = output.with_suffix(".grib")
    if output.exists():
        return output, system

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(tmp_output)

    ds = xr.open_dataset(tmp_output)

    ds = ds.mean(dim="number")
    ds = ds.rename(_renames(varname))
    ds.to_netcdf(output)
    tmp_output.unlink()
    return output, system
