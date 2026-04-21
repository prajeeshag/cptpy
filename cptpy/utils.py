import time
import pandas as pd


def season_to_tag(target: str) -> str:
    """Convert 'Mar-May' → 'MAM'. Expects 'Mon-Mon' format."""
    parts = target.split("-")
    if len(parts) != 2:
        raise ValueError(
            f"target must be 'Mon-Mon' format (e.g. 'Mar-May'), got: {target!r}"
        )
    start, end = parts
    months = pd.date_range(f"2000-{start}-01", f"2000-{end}-01", freq="MS")
    return "".join(m.strftime("%b")[0] for m in months)


class Timer:
    def __init__(self):
        self.t0    = time.perf_counter()
        self._last = self.t0

    def step(self, label: str) -> None:
        now = time.perf_counter()
        print(f"  [{label}] {now - self._last:.2f}s")
        self._last = now

    def total(self) -> None:
        print(f"  [Total] {time.perf_counter() - self.t0:.2f}s")
