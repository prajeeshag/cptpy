from pathlib import Path
import cdsapi
import xarray as xr

SYSTEMS = {
    "ecmwf": ["51"],
    "ncep": ["2"],
    "ukmo": ["12", "13", "14", "15", "600", "601", "602", "603", "604", "605", "610"],
    "jma": ["2", "3", "4"],
    "meteo_france": ["5", "6", "7", "8", "9"],
    "eccc": ["1", "2", "3", "4", "5"],
    "dwd": ["2", "21", "22"],
    "bom": ["2"],
    "cmcc": ["3", "35", "4"],
}


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


def _request(model, system, varname, start_year, end_year, month, lead, area):
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
    return request


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
    carea = "_".join(map(str, area))
    output = (
        odir / f"{model}_{varname}_{start_year}_{end_year}_{month}_{lead}_{carea}.nc"
    )
    if output.exists():
        print(f"{output} already exist... skipping download!")
        return output, system
    tmp_output = output.with_suffix(".grib")

    client = cdsapi.Client()
    request = _request(model, system, varname, start_year, end_year, month, lead, area)
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
    cache_dir: Path = Path(".cache"),
):
    # download forecast
    dl_F = []
    dl_H = []
    for lead in range(1, 6):
        d, s = download_data(
            model,
            varname,
            fyear,
            fyear,
            month,
            lead,
            area,
            odir=cache_dir,
        )
        dl_F.append(d)
        d, s = download_data(
            model,
            varname,
            hstart_year,
            hend_year,
            month,
            lead,
            area,
            system=s,
            odir=cache_dir,
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
