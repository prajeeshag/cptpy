import logging
from pathlib import Path

import xarray as xr

from .utils import season_to_tag

log = logging.getLogger(__name__)

# Precipitation-type variables — clipped at 0 after loading
_PRECIP_VARS = {"aprod", "prec", "PRCP", "precip", "precipitation"}
_TEMP_VARS = {"T2M", "tref", "t2m", "tas", "temp", "temperature", "tasmax", "tasmin"}
_KNOWN_VARS = list(_PRECIP_VARS) + list(_TEMP_VARS)

# When a configured model file is absent, try its partner before skipping.
# Both directions are listed so either can substitute for the other.
_MODEL_FALLBACKS = {
    "METEOFRANCE8": "METEOFRANCE9",
    "METEOFRANCE9": "METEOFRANCE8",
}


def _load_nc(path: Path) -> xr.DataArray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ds = xr.decode_cf(xr.open_dataset(path))
    for var in _KNOWN_VARS:
        if var in ds:
            da = ds[var].where(ds[var] != -999)
            if var in _PRECIP_VARS:
                da = da.clip(min=0)
            return da
    raise ValueError(
        f"No recognised variable in {path}. Found: {list(ds.data_vars)}. "
        f"Add the variable name to _KNOWN_VARS in data_io.py."
    )


def list_available_models(base_dir: Path, var: str) -> list[str]:
    """Return model names that have a hindcast file in base_dir (excludes KAUST obs)."""
    base_dir = Path(base_dir)
    models = []
    for f in sorted(base_dir.glob(f"*.{var}.nc")):
        name = f.stem.split(f".{var}")[0]
        if name != "KAUST":
            models.append(name)
    return models


def get_paths(config: dict) -> tuple[Path, Path, dict[str, Path]]:
    season_tag = season_to_tag(config["target"])
    year = config["fdate"].year
    var = config["variable"]
    base_dir = Path(f"data/{season_tag}{year}_{var}")
    obs_file = base_dir / f"KAUST.{var}.nc"
    model_files = {m: base_dir / f"{m}.{var}.nc" for m in config["models"]}
    return base_dir, obs_file, model_files


def load_obs(path: Path) -> xr.DataArray:
    return _load_nc(path)


def _resolve_model(m: str, base_dir: Path, suffix: str) -> tuple[str, Path] | None:
    """
    Return (name, path) for model m using suffix (e.g. '.PRCP.nc').
    If the primary file is missing, tries the fallback from _MODEL_FALLBACKS.
    Returns None if neither is available.
    """
    p = base_dir / f"{m}{suffix}"
    if p.exists():
        return m, p
    fallback = _MODEL_FALLBACKS.get(m)
    if fallback:
        pf = base_dir / f"{fallback}{suffix}"
        if pf.exists():
            log.warning("Model %s not found — using %s instead", m, fallback)
            return fallback, pf
    log.warning("Model %s not found and no fallback available — skipping", m)
    return None


def load_hindcast_models(
    config: dict, base_dir: Path, models: list[str], var: str
) -> dict[str, xr.DataArray]:
    """Load hindcast files; tries fallback model when primary is missing."""
    mods = models
    out = {}
    for m in mods:
        resolved = _resolve_model(m, base_dir, f".{var}.nc")
        if resolved:
            name, path = resolved
            out[name] = _load_nc(path)
    return out


def load_forecast_models(
    base_dir: Path, models: list[str], var: str
) -> dict[str, xr.DataArray]:
    """Load forecast files; tries fallback model when primary is missing, skips if neither found."""
    mods = models
    out = {}
    out = {}
    for m in mods:
        resolved = _resolve_model(m, base_dir, f".{var}_f2025.nc")
        if resolved:
            name, path = resolved
            out[name] = _load_nc(path)
        else:
            log.warning("No forecast file for %s — using climatology (skipped)", m)
    return out
