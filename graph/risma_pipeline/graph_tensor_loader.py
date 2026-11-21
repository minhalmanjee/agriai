"""Step 4: past-only graph tensor loader (schema + scaler driven).

For each valid origin (station, date t) returns:
  - node_x:      (N, T, F) last T days ending at t (scaled; NaN→0)
                 VWC channel 6 = anchored mean (vwc_d − vwc_t)
  - node_mask:   (N, T) observation masks
  - node_static: (N, S) depth/sensor_id, node-type one-hots
  - station_attrs: (A,) lat/lon/cluster + doy sin/cos at t
  - origin_feats: (F_origin,) LightGBM PAST_FEATURES at t (train-scaled)
  - edge_index:  (2, E) within-station adjacency from schema rules
  - y / y_scaled / y_mask / n_clean for horizons 1, 3, 7, 14

Reads column names and window from node_schema.json.
Scaler: step10b_scaled/scaler.json (train-only). No future rain / solar.

Node presence: ``structural_mask_*=1`` means permanently absent (never tier-A
in the station record). Loader zeros that node's obs_mask. Sensors with any
tier-A history have structural=0.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .depth_specs import schema_depth_specs
from .split_config import load_split_config
from .step10_lightgbm_baselines import build_origin_feat_lookup
from .step10b_node_scaler import add_derived_columns, transform_targets

_PKG = Path(__file__).resolve().parent
_GRAPH = _PKG.parent
_STEP10 = _GRAPH / "risma_processed" / "step10_stage0"
_DEFAULT_SCHEMA = _PKG / "node_schema.json"
_DEFAULT_PARQUET = _STEP10 / "parquet_model12_v2"
_DEFAULT_ORIGINS = _STEP10 / "origins_long.csv"
_DEFAULT_ORIGINS_SCALED = _STEP10 / "step10b_scaled" / "origins_scaled_node.csv"
_DEFAULT_SCALER = _STEP10 / "step10b_scaled" / "scaler.json"
_DEFAULT_ATTR = _PKG / "station_attributes_model12.csv"

DEPTHS = (5, 20, 50)
SENSORS = (1, 2, 3)
NODE_TYPE_VWC = 0
NODE_TYPE_SOILT = 1
NODE_TYPE_CLIMATE = 2

# Must never feed these parquet column name tokens into node_x / weather inputs.
FORBIDDEN_INPUT_TOKENS = (
    "delta_vwc",
    "origin_valid",
    "n_clean_sensors",
    "disagreement",
    "grade21_span",
    "clean_subset_thaw",
)

# Dynamic channel layout (pad shorter nodes with 0)
# Headline / legacy VWC: mean, min, max, ice, dvwc_1d, dvwc_3d  → F_DYN=10
# Ablation adds anchored_mean at index 6 → F_DYN=11
# soilT:   mean, min
# climate: precip, precip_3d, precip_7d, precip_30d, air_temp, rh, wind_sin, wind_cos, wind_speed
F_DYN_LEGACY = 10
F_DYN_ANCHORED = 11
F_DYN = F_DYN_LEGACY  # module default = headline
CH_VWC_MEAN = 0
CH_VWC_ANCHORED = 6
CH_VWC = ("mean", "min", "max", "ice", "dvwc_1d", "dvwc_3d", "anchored")
CH_SOILT = ("mean", "min")
CH_CLIMATE = (
    "precip",
    "precip_3d",
    "precip_7d",
    "precip_30d",
    "air_temp",
    "rh",
    "wind_dir_sin",
    "wind_dir_cos",
    "wind_speed",
)
CLIMATE_MASKS = (
    "obs_mask_precip",
    "obs_mask_precip",
    "obs_mask_precip",
    "obs_mask_precip",
    "obs_mask_air_temp",
    "obs_mask_rh",
    "obs_mask_wind_dir",
    "obs_mask_wind_dir",
    "obs_mask_wind_speed",
)


def load_node_schema(path: Path | None = None) -> dict[str, Any]:
    p = path or _DEFAULT_SCHEMA
    return json.loads(Path(p).read_text())


def _assert_no_forbidden(colnames: list[str], context: str) -> None:
    bad = [c for c in colnames if any(tok in c for tok in FORBIDDEN_INPUT_TOKENS)]
    if bad:
        raise AssertionError(f"Forbidden input columns in {context}: {bad}")


@dataclass(frozen=True)
class GraphSample:
    station: str
    date: pd.Timestamp
    block: str
    node_x: torch.Tensor  # (N, T, F)
    node_mask: torch.Tensor  # (N, T)
    node_static: torch.Tensor  # (N, S)
    station_attrs: torch.Tensor  # (A,) incl. doy sin/cos
    origin_feats: torch.Tensor  # (F_origin,)
    edge_index: torch.Tensor  # (2, E)
    y: torch.Tensor  # (H,) physical Δ
    y_scaled: torch.Tensor  # (H,)
    y_mask: torch.Tensor  # (H,)
    horizons: tuple[int, ...]
    n_clean_sensors: torch.Tensor  # (H,)


def _build_node_table(schema: dict) -> list[dict]:
    """Ordered nodes from schema columns.per_sensor_per_day (+ climate)."""
    cols = schema.get("columns", {}).get("per_sensor_per_day", {})
    depth_specs = schema_depth_specs(schema)
    sensors = cols.get("sensors", list(SENSORS))
    nodes: list[dict] = []
    for depth_cm, label in depth_specs:
        for sid in sensors:
            nodes.append(
                {
                    "node_type": NODE_TYPE_VWC,
                    "depth_cm": float(depth_cm),
                    "depth_label": label,
                    "sensor_id": int(sid),
                    "mean_col": f"vwc_{label}cm_s{sid}",
                    "min_col": f"vwc_{label}cm_s{sid}_min",
                    "max_col": f"vwc_{label}cm_s{sid}_max",
                    "ice_col": f"ice_frac_{label}cm_s{sid}",
                    "dvwc_1d_col": f"dvwc_{label}cm_s{sid}_1d",
                    "dvwc_3d_col": f"dvwc_{label}cm_s{sid}_3d",
                    "mask_col": f"obs_mask_vwc_{label}cm_s{sid}",
                    "struct_col": f"structural_mask_vwc_{label}cm_s{sid}",
                }
            )
    for depth_cm, label in depth_specs:
        for sid in sensors:
            nodes.append(
                {
                    "node_type": NODE_TYPE_SOILT,
                    "depth_cm": float(depth_cm),
                    "depth_label": label,
                    "sensor_id": int(sid),
                    "mean_col": f"soilT_{label}cm_s{sid}",
                    "min_col": f"soilT_{label}cm_s{sid}_min",
                    "mask_col": f"obs_mask_soilT_{label}cm_s{sid}",
                    "struct_col": f"structural_mask_soilT_{label}cm_s{sid}",
                }
            )
    nodes.append(
        {
            "node_type": NODE_TYPE_CLIMATE,
            "depth_cm": -1.0,
            "depth_label": "",
            "sensor_id": -1,
        }
    )
    return nodes


def _within_station_edges(nodes: list[dict], directed_vertical: bool = True) -> np.ndarray:
    """Adjacency from schema: vertical (optionally directed), replicates, VWC↔soilT, climate."""
    edges: list[tuple[int, int]] = []
    vwc_idx = [i for i, n in enumerate(nodes) if n["node_type"] == NODE_TYPE_VWC]
    soilt_idx = [i for i, n in enumerate(nodes) if n["node_type"] == NODE_TYPE_SOILT]
    climate_i = next(i for i, n in enumerate(nodes) if n["node_type"] == NODE_TYPE_CLIMATE)

    by_depth: dict[int, list[int]] = {}
    for i in vwc_idx:
        by_depth.setdefault(nodes[i]["depth_cm"], []).append(i)

    for idxs in by_depth.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                edges.append((idxs[a], idxs[b]))
                edges.append((idxs[b], idxs[a]))

    depth_order = sorted(by_depth)
    for d0, d1 in zip(depth_order, depth_order[1:]):
        for i in by_depth[d0]:
            for j in by_depth[d1]:
                if nodes[i]["sensor_id"] != nodes[j]["sensor_id"]:
                    continue
                edges.append((i, j))
                if not directed_vertical:
                    edges.append((j, i))

    soilt_map = {(nodes[i]["depth_cm"], nodes[i]["sensor_id"]): i for i in soilt_idx}
    for i in vwc_idx:
        key = (nodes[i]["depth_cm"], nodes[i]["sensor_id"])
        if key in soilt_map:
            j = soilt_map[key]
            edges.append((i, j))
            edges.append((j, i))

    for i in vwc_idx + soilt_idx:
        edges.append((climate_i, i))
        edges.append((i, climate_i))

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


def _apply_scaler_col(vals: np.ndarray, col: str, scaler: dict) -> np.ndarray:
    spec = scaler.get("features", {}).get(col)
    out = vals.astype(float).copy()
    if not spec:
        return out
    method = spec.get("method", "identity")
    if method == "identity":
        return out
    if method == "log1p_zscore":
        out = np.log1p(np.clip(out, 0, None))
        out = (out - float(spec["mean"])) / float(spec["std"])
    elif method == "zscore":
        out = (out - float(spec["mean"])) / float(spec["std"])
    return out


def _load_station_attrs(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """Return station → attr vector and feature names (lat, lon, cluster one-hots)."""
    if not path.exists():
        return {}, []
    df = pd.read_csv(path)
    clusters = sorted(df["cluster"].dropna().unique().tolist())
    names = ["lat", "lon"] + [f"cluster_{c}" for c in clusters]
    out: dict[str, np.ndarray] = {}
    for _, r in df.iterrows():
        vec = np.zeros(len(names), dtype=np.float32)
        vec[0] = float(r["lat"]) if pd.notna(r["lat"]) else 0.0
        vec[1] = float(r["lon"]) if pd.notna(r["lon"]) else 0.0
        if pd.notna(r["cluster"]) and r["cluster"] in clusters:
            vec[2 + clusters.index(r["cluster"])] = 1.0
        out[str(r["station"])] = vec
    return out, names


def _doy_sin_cos(date: pd.Timestamp) -> tuple[float, float]:
    doy = float(date.dayofyear)
    ang = 2.0 * math.pi * doy / 365.25
    return math.sin(ang), math.cos(ang)


class RismaGraphDataset(Dataset):
    """One item = one origin (station, date) with multi-horizon targets."""

    def __init__(
        self,
        block: str | None = None,
        schema_path: Path | None = None,
        parquet_dir: Path = _DEFAULT_PARQUET,
        origins_path: Path = _DEFAULT_ORIGINS,
        origins_scaled_path: Path = _DEFAULT_ORIGINS_SCALED,
        scaler_path: Path = _DEFAULT_SCALER,
        attr_path: Path = _DEFAULT_ATTR,
        window_days: int | None = None,
        use_scaled_inputs: bool = True,
        directed_vertical: bool = True,
        include_origin_feats: bool = True,
        include_station_code: bool = True,
        # Headline / legacy defaults (bit-match v3_h14_thaw hybrid).
        include_anchored_vwc: bool = False,
        include_doy: bool = False,
        target_depth: str | None = None,
    ) -> None:
        self.schema = load_node_schema(schema_path)
        self.nodes = _build_node_table(self.schema)
        tgt_default = (
            self.schema.get("columns", {})
            .get("targets", {})
            .get("delta_vwc", {})
            .get("default_target_depth", "20")
        )
        self.target_depth = str(target_depth) if target_depth is not None else str(tgt_default)
        self.edge_index = torch.as_tensor(
            _within_station_edges(self.nodes, directed_vertical=directed_vertical),
            dtype=torch.long,
        )
        self.horizons = tuple(int(h) for h in self.schema.get("horizons_days", [1, 3, 7]))
        time_cfg = self.schema.get("time", {})
        self.resolution = str(time_cfg.get("resolution", "daily"))
        self.is_15min = self.resolution == "15min"
        self.steps_per_day = 96 if self.is_15min else 1
        if window_days is not None:
            self.window = int(window_days)
        elif self.is_15min and "input_window_steps" in time_cfg:
            self.window = int(time_cfg["input_window_steps"])
        else:
            self.window = int(time_cfg.get("input_window_days", 30))
        self.use_scaled_inputs = use_scaled_inputs
        self.include_anchored_vwc = include_anchored_vwc
        self.include_doy = include_doy
        self.split = load_split_config()
        self.f_dyn = F_DYN_ANCHORED if include_anchored_vwc else F_DYN_LEGACY
        self.parquet_dir = Path(parquet_dir)
        # Origin time key: full tip timestamp for 15-min; calendar date for daily.
        self.origin_time_col = "timestamp" if self.is_15min else "date"

        if not Path(scaler_path).exists():
            raise FileNotFoundError(
                f"Missing {scaler_path} — run: python -m graph.risma_pipeline.step10b_node_scaler"
            )
        self.scaler = json.loads(Path(scaler_path).read_text())

        self._x_cache: dict[str, np.ndarray] = {}
        self._m_cache: dict[str, np.ndarray] = {}
        self._date_pos: dict[str, dict[pd.Timestamp, int]] = {}
        self.panels: dict[str, pd.DataFrame] = {}
        for path in sorted(Path(parquet_dir).glob("RISMA_*.parquet")):
            df = pd.read_parquet(path)
            if self.is_15min or "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
                df["date"] = df["timestamp"].dt.floor("D")
                pos_keys = df["timestamp"]
            else:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                pos_keys = df["date"]
            df = add_derived_columns(df, steps_per_day=self.steps_per_day)
            st = path.stem
            self.panels[st] = df
            x_arr, m_arr = self._materialize_station(df)
            self._x_cache[st] = x_arr
            self._m_cache[st] = m_arr
            self._date_pos[st] = {pd.Timestamp(d): i for i, d in enumerate(pos_keys)}

        src_origins = Path(origins_scaled_path) if Path(origins_scaled_path).exists() else Path(origins_path)
        peek_cols = pd.read_csv(src_origins, nrows=0).columns
        parse_dates = [c for c in ("date", "timestamp") if c in peek_cols]
        o = pd.read_csv(src_origins, parse_dates=parse_dates, low_memory=False)
        if "depth" in o.columns:
            o["depth"] = o["depth"].astype(str)
            o = o[o["depth"] == self.target_depth].copy()
        if "delta_vwc_scaled" not in o.columns:
            o = transform_targets(o, self.scaler, target_depth=self.target_depth)
        if self.origin_time_col not in o.columns:
            # daily scaled origins only have date
            self.origin_time_col = "date"
        o = o[o["horizon"].isin(self.horizons)].copy()
        split_key = o["timestamp"] if "timestamp" in o.columns else o["date"]
        o["block"] = self.split.assign_block(split_key)
        o["eval_eligible"] = self.split.eval_eligible(o["station"], o["block"])
        o = o[o["block"].isin(["train", "val", "test"]) & o["eval_eligible"]].copy()
        if block is not None:
            o = o[o["block"] == block].copy()

        tcol = self.origin_time_col
        base = o[["station", tcol, "block"]].drop_duplicates()
        if tcol != "date" and "date" in o.columns:
            # keep calendar date for reporting convenience
            base = base.merge(
                o[["station", tcol, "date"]].drop_duplicates(),
                on=["station", tcol],
                how="left",
            )
        wide = base
        for h in self.horizons:
            hs = o.loc[
                o["horizon"] == h,
                ["station", tcol, "delta_vwc", "delta_vwc_scaled", "n_clean_sensors"],
            ].copy()
            hs = hs.rename(
                columns={
                    "delta_vwc": f"y_h{h}",
                    "delta_vwc_scaled": f"y_scaled_h{h}",
                    "n_clean_sensors": f"n_clean_h{h}",
                }
            )
            hs[f"y_mask_h{h}"] = 1.0
            wide = wide.merge(hs, on=["station", tcol], how="left")
        for h in self.horizons:
            if f"y_mask_h{h}" not in wide.columns:
                wide[f"y_mask_h{h}"] = 0.0
            wide[f"y_mask_h{h}"] = wide[f"y_mask_h{h}"].fillna(0.0)
            if f"n_clean_h{h}" not in wide.columns:
                wide[f"n_clean_h{h}"] = np.nan
        self.index = wide.reset_index(drop=True)

        self.station_attrs, self.attr_names = _load_station_attrs(attr_path)
        if self.include_doy:
            self.attr_names = list(self.attr_names) + ["doy_sin", "doy_cos"]
        self.n_attrs = len(self.attr_names) if self.attr_names else (2 + (2 if self.include_doy else 0))

        static = []
        for n in self.nodes:
            depth_n = 0.0 if n["depth_cm"] < 0 else n["depth_cm"] / 50.0
            sid_n = 0.0 if n["sensor_id"] < 0 else n["sensor_id"] / 3.0
            t = n["node_type"]
            static.append(
                [depth_n, sid_n, float(t == 0), float(t == 1), float(t == 2)]
            )
        self.node_static = torch.tensor(static, dtype=torch.float32)

        self._stations = self.index["station"].astype(str).to_numpy()
        # Window lookup key: tip timestamp (15-min) or calendar date (daily).
        _tkey = self.origin_time_col if self.origin_time_col in self.index.columns else "date"
        self._dates = pd.to_datetime(self.index[_tkey]).to_numpy()
        self._blocks = self.index["block"].astype(str).to_numpy()
        self._y = np.stack(
            [self.index[f"y_h{h}"].to_numpy(dtype=np.float32) for h in self.horizons],
            axis=1,
        )
        self._y_s = np.stack(
            [self.index[f"y_scaled_h{h}"].to_numpy(dtype=np.float32) for h in self.horizons],
            axis=1,
        )
        self._y_m = np.stack(
            [self.index[f"y_mask_h{h}"].to_numpy(dtype=np.float32) for h in self.horizons],
            axis=1,
        )
        self._n_clean = np.stack(
            [self.index[f"n_clean_h{h}"].to_numpy(dtype=np.float32) for h in self.horizons],
            axis=1,
        )
        tgt_specs = self.scaler.get("targets", {})
        by_depth = self.scaler.get("targets_by_depth") or {}
        if self.target_depth in by_depth:
            tgt_specs = by_depth[self.target_depth]
        for hi, h in enumerate(self.horizons):
            need = (self._y_m[:, hi] > 0) & (~np.isfinite(self._y_s[:, hi])) & np.isfinite(
                self._y[:, hi]
            )
            if need.any():
                self._y_s[need, hi] = self._y[need, hi] / float(tgt_specs[str(h)]["std"])
        self._y = np.nan_to_num(self._y, nan=0.0)
        self._y_s = np.nan_to_num(self._y_s, nan=0.0)
        self._n_clean = np.nan_to_num(self._n_clean, nan=0.0)
        self._y_m = np.nan_to_num(self._y_m, nan=0.0)

        # Origin engineered features (LightGBM PAST_FEATURES, train-scaled)
        self.include_origin_feats = include_origin_feats
        self.origin_feat_names: list[str] = []
        self.origin_feat_stats: dict[str, tuple[float, float]] = {}
        self._origin_lookup: dict[tuple[str, pd.Timestamp], np.ndarray] = {}
        self.n_origin_feats = 0
        if include_origin_feats:
            train_mask = self._blocks == "train"
            # If this instance is val/test-only, fit stats from full train origins.
            if train_mask.any():
                train_keys = pd.DataFrame(
                    {
                        "station": self._stations[train_mask],
                        "date": self._dates[train_mask],
                    }
                )
            else:
                o_all = pd.read_csv(origins_path, parse_dates=["date"])
                o_all["block"] = self.split.assign_block(o_all["date"])
                o_all["eval_eligible"] = self.split.eval_eligible(
                    o_all["station"], o_all["block"]
                )
                tr = o_all[
                    (o_all["block"] == "train")
                    & o_all["eval_eligible"]
                    & o_all["horizon"].isin(self.horizons)
                ][["station", "date"]].drop_duplicates()
                train_keys = tr
            lookup, names, stats = build_origin_feat_lookup(
                self.parquet_dir,
                train_keys,
                include_station_code=include_station_code,
            )
            self._origin_lookup = lookup
            self.origin_feat_names = names
            self.origin_feat_stats = stats
            self.n_origin_feats = len(names)

    def input_column_allowlist(self) -> list[str]:
        """Explicit columns that may enter node_x / masks."""
        allowed: set[str] = set()
        for n in self.nodes:
            for k, v in n.items():
                if isinstance(v, str) and k.endswith("_col"):
                    allowed.add(v)
        allowed.update(CH_CLIMATE)
        allowed.update(CLIMATE_MASKS)
        allowed.update({f"precip_{k}" for k in ("1d", "3d", "7d", "30d")})
        for _, label in schema_depth_specs(self.schema):
            for s in SENSORS:
                allowed.add(f"dvwc_{label}cm_s{s}_1d")
                allowed.add(f"dvwc_{label}cm_s{s}_3d")
        _assert_no_forbidden(sorted(allowed), "input allowlist")
        return sorted(allowed)

    def _materialize_station(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Precompute scaled node tensors (N, D, F) and masks (N, D) for one station."""
        allow = set(self.input_column_allowlist())
        n_nodes = len(self.nodes)
        d = len(df)
        x = np.zeros((n_nodes, d, self.f_dyn), dtype=np.float32)
        m = np.zeros((n_nodes, d), dtype=np.float32)

        def col(name: str) -> np.ndarray:
            if name not in allow:
                raise AssertionError(f"Attempted to read non-allowlisted column {name!r}")
            if name not in df.columns:
                return np.full(d, np.nan, dtype=float)
            return df[name].to_numpy(dtype=float)

        for ni, node in enumerate(self.nodes):
            if node["node_type"] == NODE_TYPE_CLIMATE:
                for fi, (ch, mk) in enumerate(zip(CH_CLIMATE, CLIMATE_MASKS)):
                    vals = col(ch)
                    if self.use_scaled_inputs:
                        vals = _apply_scaler_col(vals, ch, self.scaler)
                    x[ni, :, fi] = np.nan_to_num(vals, nan=0.0)
                    mask_v = np.nan_to_num(col(mk), nan=0.0)
                    if fi == 0:
                        m[ni, :] = mask_v
                    else:
                        m[ni, :] = np.maximum(m[ni, :], mask_v)
            elif node["node_type"] == NODE_TYPE_VWC:
                channels = [
                    (node["mean_col"], 0),
                    (node["min_col"], 1),
                    (node["max_col"], 2),
                    (node["ice_col"], 3),
                    (node["dvwc_1d_col"], 4),
                    (node["dvwc_3d_col"], 5),
                ]
                mask = col(node["mask_col"])
                # structural_mask==1 ⇒ permanently absent
                if node.get("struct_col"):
                    absent = col(node["struct_col"])
                    mask = np.where(np.nan_to_num(absent, nan=0.0) >= 1.0, 0.0, mask)
                for cname, fi in channels:
                    vals = col(cname)
                    if self.use_scaled_inputs:
                        vals = _apply_scaler_col(vals, cname, self.scaler)
                    x[ni, :, fi] = np.nan_to_num(vals, nan=0.0)
                m[ni, :] = np.nan_to_num(mask, nan=0.0)
            else:
                channels = [(node["mean_col"], 0), (node["min_col"], 1)]
                mask = col(node["mask_col"])
                if node.get("struct_col"):
                    absent = col(node["struct_col"])
                    mask = np.where(np.nan_to_num(absent, nan=0.0) >= 1.0, 0.0, mask)
                for cname, fi in channels:
                    vals = col(cname)
                    if self.use_scaled_inputs:
                        vals = _apply_scaler_col(vals, cname, self.scaler)
                    x[ni, :, fi] = np.nan_to_num(vals, nan=0.0)
                m[ni, :] = np.nan_to_num(mask, nan=0.0)
        return x, m

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> GraphSample:
        station = self._stations[i]
        date = pd.Timestamp(self._dates[i])
        T = self.window
        N = len(self.nodes)
        x = np.zeros((N, T, self.f_dyn), dtype=np.float32)
        m = np.zeros((N, T), dtype=np.float32)

        pos = self._date_pos.get(station, {}).get(date)
        if pos is not None:
            start = pos - T + 1
            src_x = self._x_cache[station]
            src_m = self._m_cache[station]
            for t_dst, t_src in enumerate(range(start, pos + 1)):
                if 0 <= t_src < src_x.shape[1]:
                    x[:, t_dst, :] = src_x[:, t_src, :]
                    m[:, t_dst] = src_m[:, t_src]

        # Anchored VWC: mean_d − mean_t (channel 6); zero if missing at t
        if self.include_anchored_vwc:
            for ni, node in enumerate(self.nodes):
                if node["node_type"] != NODE_TYPE_VWC:
                    continue
                level = x[ni, :, CH_VWC_MEAN]
                m_t = m[ni, -1]
                if m_t < 0.5:
                    x[ni, :, CH_VWC_ANCHORED] = 0.0
                else:
                    x[ni, :, CH_VWC_ANCHORED] = level - level[-1]

        base_attrs = self.station_attrs.get(
            station, np.zeros(max(self.n_attrs - (2 if self.include_doy else 0), 1), dtype=np.float32)
        )
        if self.include_doy:
            s, c = _doy_sin_cos(date)
            attrs = np.concatenate(
                [base_attrs.astype(np.float32), np.asarray([s, c], dtype=np.float32)]
            )
        else:
            attrs = base_attrs.astype(np.float32)

        if self.include_origin_feats and self.n_origin_feats > 0:
            of = self._origin_lookup.get((station, date))
            if of is None:
                of = np.zeros(self.n_origin_feats, dtype=np.float32)
            origin_feats = torch.from_numpy(of.astype(np.float32))
        else:
            origin_feats = torch.zeros(0, dtype=torch.float32)

        return GraphSample(
            station=station,
            date=date,
            block=str(self._blocks[i]),
            node_x=torch.from_numpy(x),
            node_mask=torch.from_numpy(m),
            node_static=self.node_static.clone(),
            station_attrs=torch.from_numpy(attrs),
            origin_feats=origin_feats,
            edge_index=self.edge_index.clone(),
            y=torch.from_numpy(self._y[i].copy()),
            y_scaled=torch.from_numpy(self._y_s[i].copy()),
            y_mask=torch.from_numpy(self._y_m[i].copy()),
            horizons=self.horizons,
            n_clean_sensors=torch.from_numpy(self._n_clean[i].copy()),
        )


def collate_graphs(batch: list[GraphSample]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "station": [b.station for b in batch],
        "date": [b.date for b in batch],
        "block": [b.block for b in batch],
        "node_x": torch.stack([b.node_x for b in batch], dim=0),
        "node_mask": torch.stack([b.node_mask for b in batch], dim=0),
        "node_static": torch.stack([b.node_static for b in batch], dim=0),
        "station_attrs": torch.stack([b.station_attrs for b in batch], dim=0),
        "edge_index": batch[0].edge_index,
        "y": torch.stack([b.y for b in batch], dim=0),
        "y_scaled": torch.stack([b.y_scaled for b in batch], dim=0),
        "y_mask": torch.stack([b.y_mask for b in batch], dim=0),
        "n_clean_sensors": torch.stack([b.n_clean_sensors for b in batch], dim=0),
        "horizons": batch[0].horizons,
    }
    if batch[0].origin_feats.numel() > 0:
        out["origin_feats"] = torch.stack([b.origin_feats for b in batch], dim=0)
    return out


def smoke_shapes(block: str = "val", n: int = 3) -> dict:
    ds = RismaGraphDataset(block=block)
    out: dict[str, Any] = {
        "n_origins": len(ds),
        "n_nodes": len(ds.nodes),
        "window": ds.window,
        "f_dyn": ds.f_dyn,
        "horizons": list(ds.horizons),
        "edge_index_shape": list(ds.edge_index.shape),
        "attr_names": ds.attr_names,
        "origin_feat_names": ds.origin_feat_names,
        "n_origin_feats": ds.n_origin_feats,
        "scaler_path": str(_DEFAULT_SCALER),
        "schema_id": ds.schema.get("schema_id"),
        "samples": [],
    }
    for i in range(min(n, len(ds))):
        s = ds[i]
        out["samples"].append(
            {
                "station": s.station,
                "date": str(s.date.date()),
                "node_x": list(s.node_x.shape),
                "node_mask": list(s.node_mask.shape),
                "station_attrs": list(s.station_attrs.shape),
                "origin_feats": list(s.origin_feats.shape),
                "y_physical": [float(v) for v in s.y.tolist()],
                "y_scaled": [float(v) for v in s.y_scaled.tolist()],
                "y_mask": [float(v) for v in s.y_mask.tolist()],
                "n_clean": [float(v) for v in s.n_clean_sensors.tolist()],
                "mask_frac": float(s.node_mask.mean()),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block", default="val", choices=["train", "val", "test", "all"])
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    block = None if args.block == "all" else args.block
    info = smoke_shapes(block or "val", args.n)
    if args.block == "all":
        info = {
            "train": smoke_shapes("train", 1),
            "val": smoke_shapes("val", 1),
            "test": smoke_shapes("test", 1),
        }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
