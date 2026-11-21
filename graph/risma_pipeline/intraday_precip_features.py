"""Past-only daily intraday precip features from raw 15-min RISMA CSVs.

Features are computed on calendar day t only (same day as Stage0 origin date).
No future tips; no oracle forecast precip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .met_bounds import MET_PHYSICAL_BOUNDS, STATIONS_MODEL_12
from .stage0_raw_audit import classify_raw_column, load_station_raw

INTRADAY_FEATURE_NAMES = [
    "precip_max_1h",
    "precip_max_3h",
    "precip_max_15min",
    "n_hours_rain",
    "precip_pm_share",
]

_PKG = Path(__file__).resolve().parent
_DEFAULT_RAW = _PKG.parent / "risma_data_full"
_DEFAULT_CACHE = (
    _PKG.parent / "risma_processed" / "step10_stage0" / "e3_followup" / "intraday_precip_daily.parquet"
)


def _find_precip_col(columns: list[str]) -> str | None:
    for c in columns:
        if classify_raw_column(c)["family"] == "precip":
            return c
    return None


def _station_daily_intraday(path: Path, station: str) -> pd.DataFrame:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    precip_col = _find_precip_col(cols)
    if precip_col is None:
        return pd.DataFrame(columns=["station", "date", *INTRADAY_FEATURE_NAMES])

    df = load_station_raw(path, usecols=["timestamp", precip_col])
    lo, hi = MET_PHYSICAL_BOUNDS["precip"]
    p = pd.to_numeric(df[precip_col], errors="coerce")
    p = p.where(p.between(lo, hi), np.nan).fillna(0.0).clip(lower=0.0)
    df = df.assign(p=p)
    df["date"] = df["timestamp"].dt.normalize()
    df["hour"] = df["timestamp"].dt.hour

    rows = []
    for day, g in df.groupby("date", sort=True):
        tips = g["p"].to_numpy(dtype=float)
        max_15 = float(tips.max()) if len(tips) else 0.0
        by_h = g.groupby("hour")["p"].sum().reindex(range(24), fill_value=0.0).to_numpy(dtype=float)
        daily = float(by_h.sum())
        max_1h = float(by_h.max()) if len(by_h) else 0.0
        if len(by_h) >= 3:
            max_3h = float(np.convolve(by_h, np.ones(3), mode="valid").max())
        else:
            max_3h = max_1h
        n_hours = int((by_h > 0.0).sum())
        pm = float(by_h[12:].sum())
        pm_share = (pm / daily) if daily > 0 else 0.0
        rows.append(
            {
                "station": station,
                "date": pd.Timestamp(day),
                "precip_max_1h": max_1h,
                "precip_max_3h": max_3h,
                "precip_max_15min": max_15,
                "n_hours_rain": float(n_hours),
                "precip_pm_share": pm_share,
            }
        )
    return pd.DataFrame(rows)


def build_intraday_precip_panel(
    raw_dir: Path | None = None,
    *,
    stations: list[str] | None = None,
    cache_path: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    raw_dir = raw_dir or _DEFAULT_RAW
    cache_path = cache_path or _DEFAULT_CACHE
    stations = stations or list(STATIONS_MODEL_12)

    if cache_path.exists() and not force:
        panel = pd.read_parquet(cache_path)
        panel["date"] = pd.to_datetime(panel["date"])
        return panel

    parts = []
    for st in stations:
        path = raw_dir / f"{st}.csv"
        if not path.exists():
            print(f"  warn: missing raw {path}", flush=True)
            continue
        print(f"  intraday precip: {st}", flush=True)
        parts.append(_station_daily_intraday(path, st))
    if not parts:
        raise FileNotFoundError(f"No raw precip found under {raw_dir}")
    panel = pd.concat(parts, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache_path, index=False)
    print(f"  wrote {cache_path} rows={len(panel)}", flush=True)
    return panel


if __name__ == "__main__":
    build_intraday_precip_panel(force=True)
