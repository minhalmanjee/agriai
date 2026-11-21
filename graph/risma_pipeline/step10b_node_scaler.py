"""Step 10b (node path): train-only scaler driven by node_schema.json.

Fits on Stage0 daily parquet rows whose dates fall in the train block.
Writes scaler.json + apply / inverse helpers for the loader and GRU smoke.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .depth_specs import schema_depth_specs
from .split_config import load_split_config

_PKG = Path(__file__).resolve().parent
_GRAPH = _PKG.parent
_STEP10 = _GRAPH / "risma_processed" / "step10_stage0"
_SCHEMA = _PKG / "node_schema.json"
_PARQUET = _STEP10 / "parquet_model12_v2"
_ORIGINS = _STEP10 / "origins_long.csv"
_OUT = _STEP10 / "step10b_scaled"
_SCALER_PATH = _OUT / "scaler.json"

DEPTHS = (5, 20, 50)
SENSORS = (1, 2, 3)


def load_schema(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or _SCHEMA).read_text())


def add_derived_columns(df: pd.DataFrame, *, steps_per_day: int = 1) -> pd.DataFrame:
    """Per-sensor ΔVWC (1d/3d) and precip rolling sums ending at t (incl. today).

    steps_per_day=1 → daily panels; 96 → 15-min tips (rolls/shifts in tip steps).
    """
    out = df.copy()
    spd = max(int(steps_per_day), 1)
    if "precip" in out.columns:
        p = pd.to_numeric(out["precip"], errors="coerce")
        out["precip_1d"] = p.rolling(spd, min_periods=1).sum() if spd > 1 else p
        out["precip_3d"] = p.rolling(3 * spd, min_periods=1).sum()
        out["precip_7d"] = p.rolling(7 * spd, min_periods=1).sum()
        out["precip_30d"] = p.rolling(30 * spd, min_periods=1).sum()
    labels = [lab for _, lab in schema_depth_specs()]
    # Prefer labels present in this panel (supports older 5/20/50-only exports).
    present_labels = []
    for lab in labels:
        if any(c.startswith(f"vwc_{lab}cm_s") for c in out.columns):
            present_labels.append(lab)
    if not present_labels:
        present_labels = [str(d) for d in DEPTHS]
    for lab in present_labels:
        for s in SENSORS:
            col = f"vwc_{lab}cm_s{s}"
            if col not in out.columns:
                continue
            v = pd.to_numeric(out[col], errors="coerce")
            out[f"dvwc_{lab}cm_s{s}_1d"] = v - v.shift(spd)
            out[f"dvwc_{lab}cm_s{s}_3d"] = v - v.shift(3 * spd)
    return out


def expand_feature_columns(schema: dict | None = None) -> dict[str, list[str]]:
    """Map scale group → concrete column names (parquet + derived)."""
    schema = schema or load_schema()
    z_vwc: list[str] = []
    z_dvwc: list[str] = []
    z_other: list[str] = []
    log1p: list[str] = []
    identity: list[str] = []

    for _, lab in schema_depth_specs(schema):
        for s in SENSORS:
            z_vwc.extend(
                [
                    f"vwc_{lab}cm_s{s}",
                    f"vwc_{lab}cm_s{s}_min",
                    f"vwc_{lab}cm_s{s}_max",
                ]
            )
            z_dvwc.extend(
                [
                    f"dvwc_{lab}cm_s{s}_1d",
                    f"dvwc_{lab}cm_s{s}_3d",
                ]
            )
            z_other.extend(
                [
                    f"soilT_{lab}cm_s{s}",
                    f"soilT_{lab}cm_s{s}_min",
                ]
            )
            identity.append(f"ice_frac_{lab}cm_s{s}")
            identity.append(f"obs_mask_vwc_{lab}cm_s{s}")
            identity.append(f"obs_mask_soilT_{lab}cm_s{s}")
            identity.append(f"structural_mask_vwc_{lab}cm_s{s}")
            identity.append(f"structural_mask_soilT_{lab}cm_s{s}")
        identity.append(f"frozen_frac_{lab}cm")

    log1p.extend(["precip", "precip_1d", "precip_3d", "precip_7d", "precip_30d"])
    z_other.extend(["air_temp", "rh", "wind_speed"])
    identity.extend(
        [
            "wind_dir_sin",
            "wind_dir_cos",
            "obs_mask_precip",
            "obs_mask_air_temp",
            "obs_mask_rh",
            "obs_mask_wind_dir",
            "obs_mask_wind_speed",
        ]
    )

    return {
        "vwc_z": z_vwc,
        "dvwc_z": z_dvwc,
        "z": z_other,
        "log1p_z": log1p,
        "identity": identity,
    }


def _mu_sig(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return 0.0, 1.0
    mu = float(np.mean(x))
    sig = float(np.std(x))
    if sig < 1e-12:
        sig = 1.0
    return mu, sig


def load_train_daily(
    parquet_dir: Path,
    split,
    *,
    steps_per_day: int = 1,
    subsample_stride: int | None = None,
) -> pd.DataFrame:
    frames = []
    time_col = None
    for path in sorted(parquet_dir.glob("RISMA_*.parquet")):
        df = pd.read_parquet(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            df["date"] = df["timestamp"].dt.floor("D")
            time_col = "timestamp"
        else:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            time_col = "date"
        if subsample_stride and subsample_stride > 1:
            df = df.iloc[:: int(subsample_stride)].reset_index(drop=True)
        df = add_derived_columns(df, steps_per_day=steps_per_day)
        df["station"] = path.stem
        df["block"] = split.assign_block(df["date"] if time_col == "date" else df["timestamp"])
        frames.append(df)
    daily = pd.concat(frames, ignore_index=True)
    return daily[daily["block"] == "train"].copy()


def fit_scaler(
    train_daily: pd.DataFrame,
    origins_train: pd.DataFrame,
    schema: dict | None = None,
) -> dict[str, Any]:
    schema = schema or load_schema()
    groups = expand_feature_columns(schema)
    params: dict[str, Any] = {
        "schema_id": schema.get("schema_id"),
        "fit_block": "train",
        "groups": groups,
        "features": {},
        "targets": {},
        "depth_scale": {"method": "divide_by", "divisor": 50.0},
        "missing_after_scale": "fill_zero_keep_mask",
    }

    present = set(train_daily.columns)
    for c in groups["vwc_z"] + groups.get("dvwc_z", []) + groups["z"]:
        if c not in present:
            continue
        mu, sig = _mu_sig(pd.to_numeric(train_daily[c], errors="coerce").to_numpy(float))
        params["features"][c] = {"method": "zscore", "mean": mu, "std": sig}

    for c in groups["log1p_z"]:
        if c not in present:
            continue
        x = np.log1p(
            np.clip(pd.to_numeric(train_daily[c], errors="coerce").to_numpy(float), 0, None)
        )
        mu, sig = _mu_sig(x)
        params["features"][c] = {"method": "log1p_zscore", "mean": mu, "std": sig}

    for c in groups["identity"]:
        if c not in present:
            continue
        params["features"][c] = {"method": "identity"}

    def _fit_target_sigmas(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for h, g in frame.groupby("horizon"):
            if int(h) not in schema.get("horizons_days", [1, 3, 7]):
                continue
            y = pd.to_numeric(g["delta_vwc"], errors="coerce").to_numpy(float)
            y = y[np.isfinite(y)]
            sig = float(np.std(y)) if len(y) > 1 else 1.0
            if sig < 1e-12:
                sig = 1.0
            out[str(int(h))] = {
                "method": "divide_by_std",
                "std": sig,
                "mean": 0.0,
                "n_train": int(len(y)),
            }
        return out

    # Headline targets: if multi-depth origins, default to depth=20 for backward-compat keys.
    if "depth" in origins_train.columns:
        by_depth: dict[str, Any] = {}
        for d, g in origins_train.groupby(origins_train["depth"].astype(str)):
            by_depth[str(d)] = _fit_target_sigmas(g)
        params["targets_by_depth"] = by_depth
        default_depth = str(
            schema.get("columns", {})
            .get("targets", {})
            .get("delta_vwc", {})
            .get("default_target_depth", "20")
        )
        params["default_target_depth"] = default_depth
        params["targets"] = by_depth.get(default_depth) or next(iter(by_depth.values()), {})
    else:
        params["targets"] = _fit_target_sigmas(origins_train)

    params["n_train_station_days"] = int(len(train_daily))
    params["n_train_origins"] = int(len(origins_train))
    params["train_date_min"] = str(train_daily["date"].min().date())
    params["train_date_max"] = str(train_daily["date"].max().date())
    return params


def transform_features(df: pd.DataFrame, params: dict, fill_missing: bool = True) -> pd.DataFrame:
    out = df.copy()
    for c, spec in params["features"].items():
        if c not in out.columns:
            continue
        method = spec["method"]
        if method == "identity":
            continue
        x = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float)
        if method == "log1p_zscore":
            x = np.log1p(np.clip(x, 0, None))
            x = (x - spec["mean"]) / spec["std"]
        elif method == "zscore":
            x = (x - spec["mean"]) / spec["std"]
        if fill_missing:
            x = np.where(np.isfinite(x), x, 0.0)
        out[c] = x
    return out


def _target_spec(params: dict, horizon: int, depth: str | None = None) -> dict[str, Any]:
    by_depth = params.get("targets_by_depth") or {}
    if depth is not None and str(depth) in by_depth:
        return by_depth[str(depth)][str(int(horizon))]
    return params["targets"][str(int(horizon))]


def transform_targets(
    origins: pd.DataFrame,
    params: dict,
    *,
    target_depth: str | None = None,
) -> pd.DataFrame:
    out = origins.copy()
    out["delta_vwc_scaled"] = np.nan
    out["delta_vwc_scale_std"] = np.nan
    by_depth = params.get("targets_by_depth") or {}
    if "depth" in out.columns and by_depth:
        for d, gidx in out.groupby(out["depth"].astype(str)).groups.items():
            specs = by_depth.get(str(d)) or params.get("targets", {})
            for h, spec in specs.items():
                mask = out.index.isin(gidx) & (out["horizon"].astype(int) == int(h))
                out.loc[mask, "delta_vwc_scaled"] = (
                    pd.to_numeric(out.loc[mask, "delta_vwc"], errors="coerce") / spec["std"]
                )
                out.loc[mask, "delta_vwc_scale_std"] = spec["std"]
    else:
        specs = params.get("targets", {})
        if target_depth is not None and str(target_depth) in by_depth:
            specs = by_depth[str(target_depth)]
        for h, spec in specs.items():
            mask = out["horizon"].astype(int) == int(h)
            out.loc[mask, "delta_vwc_scaled"] = (
                pd.to_numeric(out.loc[mask, "delta_vwc"], errors="coerce") / spec["std"]
            )
            out.loc[mask, "delta_vwc_scale_std"] = spec["std"]
    return out


def inverse_target(
    y_scaled: np.ndarray,
    horizon: int,
    params: dict,
    *,
    depth: str | None = None,
) -> np.ndarray:
    spec = _target_spec(params, horizon, depth=depth)
    return np.asarray(y_scaled, dtype=float) * spec["std"]


def scale_depth(depth_cm: float | np.ndarray, params: dict | None = None) -> float | np.ndarray:
    div = 50.0
    if params and "depth_scale" in params:
        div = float(params["depth_scale"]["divisor"])
    return np.asarray(depth_cm, dtype=float) / div


def run(
    schema_path: Path = _SCHEMA,
    parquet_dir: Path = _PARQUET,
    origins_path: Path = _ORIGINS,
    out_dir: Path = _OUT,
    *,
    steps_per_day: int | None = None,
    subsample_stride: int | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = load_schema(schema_path)
    split = load_split_config()
    res = str(schema.get("time", {}).get("resolution", "daily"))
    if steps_per_day is None:
        steps_per_day = 96 if res == "15min" else 1
    if subsample_stride is None and res == "15min":
        subsample_stride = 48  # tip volume control for scaler fit

    train_daily = load_train_daily(
        parquet_dir,
        split,
        steps_per_day=steps_per_day,
        subsample_stride=subsample_stride,
    )
    parse_cols = ["date"]
    # 15-min origins carry full timestamps
    peek = pd.read_csv(origins_path, nrows=0)
    if "timestamp" in peek.columns:
        parse_cols.append("timestamp")
    origins = pd.read_csv(origins_path, parse_dates=parse_cols)
    split_key = origins["timestamp"] if "timestamp" in origins.columns else origins["date"]
    origins["block"] = split.assign_block(split_key)
    origins["eval_eligible"] = split.eval_eligible(origins["station"], origins["block"])
    origins_train = origins[(origins["block"] == "train") & (origins["eval_eligible"])].copy()
    # headline horizons only for target σ
    origins_train = origins_train[origins_train["horizon"].isin(schema.get("horizons_days", [1, 3, 7]))]

    params = fit_scaler(train_daily, origins_train, schema)
    params["schema_path"] = str(schema_path)
    params["steps_per_day"] = int(steps_per_day)
    params["subsample_stride"] = subsample_stride
    scaler_out = out_dir / "scaler.json"
    scaler_out.write_text(json.dumps(params, indent=2) + "\n")

    # also write train-scaled origins for convenience
    scaled_origins = transform_targets(origins[origins["eval_eligible"]], params)
    scaled_origins.to_csv(out_dir / "origins_scaled_node.csv", index=False)

    return {
        "scaler_path": str(scaler_out),
        "n_train_station_days": params["n_train_station_days"],
        "n_train_origins": params["n_train_origins"],
        "n_features": len(params["features"]),
        "target_std": {h: params["targets"][h]["std"] for h in params["targets"]},
        "train_date_range": [params["train_date_min"], params["train_date_max"]],
        "steps_per_day": int(steps_per_day),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schema", type=Path, default=_SCHEMA)
    ap.add_argument("--parquet-dir", type=Path, default=_PARQUET)
    ap.add_argument("--origins", type=Path, default=_ORIGINS)
    ap.add_argument("--out-dir", type=Path, default=_OUT)
    ap.add_argument("--steps-per-day", type=int, default=None)
    ap.add_argument("--subsample-stride", type=int, default=None)
    args = ap.parse_args()
    print(
        json.dumps(
            run(
                args.schema,
                args.parquet_dir,
                args.origins,
                args.out_dir,
                steps_per_day=args.steps_per_day,
                subsample_stride=args.subsample_stride,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
