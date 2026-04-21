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
        "leadtime_month": [str(lead + 1)],
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


def ens_mean(input: list[Path], output: Path, ignore: list[str] = []):
    ds = xr.open_mfdataset(
        input,
        concat_dim="ensemble",
        combine="nested",
        compat="override",
        coords="minimal",
        join="override",
        preprocess=lambda ds: ds.drop_vars(ignore, errors="ignore"),
    ).mean(dim="ensemble")
    ds.to_netcdf(output)


def get_data(
    hstart_year,
    hend_year,
    fyear,
    month,
    area,
    varname,
    model,
):
    # download forecast
    dl_F = []
    dl_H = []
    for lead in range(1, 6):
        d, s = download_data(model, varname, fyear, fyear, month, lead, area)
        dl_F.append(d)
        d, s = download_data(
            model, varname, fyear, fyear, month, lead, area, system=SYSTEMS[model][1]
        )
        dl_H.append(d)

    for lead in range(1, 4):
        l1 = lead
        l2 = l1 + 2
        odir = Path(f"l{l1}-l{l2}")
        odir.mkdir(parents=True, exist_ok=True)
        ofile_F = odir / f"{model}.{varname}.F.nc"
        ofile_H = odir / f"{model}.{varname}.H.nc"
        ens_mean(dl_F[l1 - 1 : l2], ofile_F)
        ens_mean(dl_H[l1 - 1 : l2], ofile_H)
