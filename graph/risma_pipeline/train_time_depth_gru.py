#!/usr/bin/env python3
"""Priority-3: Temporal GRU × Depth GRU (or depth MLP) for 20 cm ΔVWC.

- Temporal GRU: shared over nodes (same as SharedGRU encoder).
- Depth path: pool VWC node hiddens by depth {5,20,50} → ordered depth GRU (or MLP).
- Fusion → multi-horizon head (same 20 cm targets as Stage0).
- Optional Rain C state features at origin (fair vs S1b Rain C).

Past-only inputs only. Seed 0 ablation vs SharedGRU / report.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from . import train_shared_gru_s1_wetting as s1
from .depth_specs import schema_depth_specs
from .graph_tensor_loader import RismaGraphDataset, collate_graphs
from .intraday_precip_features import INTRADAY_FEATURE_NAMES
from .step10b_node_scaler import _target_spec, inverse_target
from .train_shared_gru import SharedGRU, masked_huber, set_seed, skill_vs_persistence
from .train_shared_gru_hybrid import HybridGraphDataset

_PKG = Path(__file__).resolve().parent
_STEP10 = _PKG.parent / "risma_processed" / "step10_stage0"
_STEP10_15 = _PKG.parent / "risma_processed" / "step10_stage0_15min"
_OUT = _STEP10 / "e3_followup" / "time_depth_gru"
_ORIGINS = _STEP10 / "origins_long.csv"
_PARQUET = _STEP10 / "parquet_model12_v2"
_E1_META = _STEP10 / "e3_followup" / "phase1_e1" / "run_meta.json"
_S0_PREDS = _STEP10 / "e3_followup" / "s1_wetting" / "preds" / "s0_seed0_test.csv"
_RAIN_C_PREDS = _STEP10 / "e3_followup" / "s1_rain_ablation" / "preds" / "s1b_C_seed0_test.csv"
_SCHEMA_DAILY = _PKG / "node_schema.json"
_SCHEMA_DAILY_MD = _PKG / "node_schema_daily_multidepth.json"
_SCHEMA_15MIN = _PKG / "node_schema_15min.json"
_ORIGINS_DAILY_MD = _STEP10 / "origins_long_multidepth.csv"
_SCALED_DAILY_MD = _STEP10 / "step10b_scaled_multidepth"
_FOLLOWUP_OUT = _PKG.parent / "risma_processed" / "step10" / "e3_followup"


def _ddp_env() -> tuple[bool, int, int, int]:
    """Return (enabled, rank, world_size, local_rank) from torchrun env."""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        return False, 0, 1, 0
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return True, rank, ws, local_rank


def _unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module
    return model


def _is_main(rank: int) -> bool:
    return int(rank) == 0


RAIN_C_STATE = [
    "precip_1d",
    "precip_3d",
    "precip_7d",
    "precip_max_1h",
    "precip_max_3h",
    "n_hours_rain",
    "days_since_rain",
    "n_rain_days_3d",
    "dvwc_5cm_1d",
    "dvwc_20cm_1d",
    "dvwc_50cm_1d",
    "grad_5_20",
    "grad_20_50",
    "vwc_20cm",
    "soilT_thaw",
    "frozen_frac_20cm",
]

# node_static: [depth_n, sid_n, is_vwc, is_soilt, is_climate]; depth_n = depth_cm/50
DEPTHS_CM = (5, 20, 50)


def target_depth_to_cm(depth_key: str) -> float:
    """Map origin depth key → representative cm used in node_static / depth path."""
    key = str(depth_key).strip().lower()
    if key in {"0_to_5", "0to5"}:
        return 2.5
    return float(key)


class TimeDepthGRU(nn.Module):
    def __init__(
        self,
        f_dyn: int,
        n_static: int,
        n_station_attrs: int,
        n_state: int = 0,
        hidden: int = 64,
        station_emb_dim: int = 4,
        n_horizons: int = 4,
        dropout: float = 0.3,
        depth_mode: str = "gru",  # gru | mlp | none
        depths_cm: tuple[float, ...] = DEPTHS_CM,
        target_depth_cm: float | None = None,
        depth_gate_temp: float = 15.0,
    ) -> None:
        super().__init__()
        self.depth_mode = depth_mode
        self.depths_cm = tuple(float(d) for d in depths_cm)
        self.target_depth_cm = (
            float(target_depth_cm) if target_depth_cm is not None else float(self.depths_cm[0])
        )
        self.depth_gate_temp = float(depth_gate_temp)
        n_depth = max(1, len(self.depths_cm))
        in_dim = f_dyn + 1 + n_static
        self.temporal_gru = nn.GRU(in_dim, hidden, batch_first=True)
        # Temporal encoder is 1-layer (no inter-layer GRU dropout); regularize pooled reps.
        self.temp_dropout = nn.Dropout(dropout)
        self.station_proj = nn.Linear(n_station_attrs, station_emb_dim)
        self.n_state = n_state
        if n_state > 0:
            self.state_proj = nn.Sequential(nn.Linear(n_state, hidden), nn.ReLU(), nn.Dropout(dropout))
            state_dim = hidden
        else:
            self.state_proj = None
            state_dim = 0

        if depth_mode == "gru":
            self.depth_enc = nn.GRU(hidden, hidden, batch_first=True)
            depth_out = hidden
        elif depth_mode == "mlp":
            self.depth_enc = nn.Sequential(
                nn.Linear(hidden * n_depth, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            depth_out = hidden
        else:
            self.depth_enc = None
            depth_out = 0

        # head_in unchanged vs prior checkpoints (gate is parameter-free).
        head_in = hidden + station_emb_dim + depth_out + state_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, n_horizons))
        self.hidden = hidden

    def _pool_temporal(
        self,
        node_x: torch.Tensor,
        node_mask: torch.Tensor,
        node_static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pooled_all, per_node_h) with shapes (B,H), (B,N,H)."""
        b, n, t, _ = node_x.shape
        mask = node_mask.unsqueeze(-1)
        static = node_static.unsqueeze(2).expand(-1, -1, t, -1)
        x = torch.cat([node_x, mask, static], dim=-1) * mask
        x = x.reshape(b * n, t, -1)
        _, h = self.temporal_gru(x)
        h = h[-1].reshape(b, n, -1)
        w = node_mask.mean(dim=-1).clamp_min(0.0)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)
        return pooled, h

    def _depth_gate_weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Softmax over depth slots; peaks at target_depth_cm (temp in cm)."""
        depths = torch.tensor(self.depths_cm, device=device, dtype=dtype)
        dist = torch.abs(depths - float(self.target_depth_cm))
        # Larger temp → wider neighborhood (nearby depths still contribute).
        logits = -dist / max(float(self.depth_gate_temp), 1e-3)
        return torch.softmax(logits, dim=0)

    def _depth_context(self, h_nodes: torch.Tensor, node_static: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """Build ordered depth embeddings from VWC nodes; gate toward target depth."""
        if self.depth_mode == "none" or self.depth_enc is None:
            return h_nodes.new_zeros(h_nodes.size(0), 0)

        is_vwc = node_static[..., 2] > 0.5  # (B,N)
        depth_n = node_static[..., 0]  # depth_cm/50
        b, n, hid = h_nodes.shape
        # validity: any time observed
        node_obs = (node_mask.mean(dim=-1) > 0) & is_vwc  # (B,N)

        depth_hs = []
        for d_cm in self.depths_cm:
            target = float(d_cm) / 50.0
            # match depth within tolerance
            match = node_obs & (torch.abs(depth_n - target) < 1e-3)
            # masked mean of node hiddens at this depth
            m = match.unsqueeze(-1).float()
            denom = m.sum(dim=1).clamp_min(1e-6)
            pooled_d = (h_nodes * m).sum(dim=1) / denom
            # if no nodes, zeros
            depth_hs.append(pooled_d)
        # (B, n_depth, H)
        seq = torch.stack(depth_hs, dim=1)
        gate = self._depth_gate_weights(seq.device, seq.dtype)  # (n_depth,)
        seq = seq * gate.view(1, -1, 1)

        if self.depth_mode == "gru":
            _, dh = self.depth_enc(seq)
            return dh[-1]
        # mlp on concat
        return self.depth_enc(seq.reshape(b, -1))

    def forward(
        self,
        node_x: torch.Tensor,
        node_mask: torch.Tensor,
        node_static: torch.Tensor,
        station_attrs: torch.Tensor,
        state_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled, h_nodes = self._pool_temporal(node_x, node_mask, node_static)
        pooled = self.temp_dropout(pooled)
        emb = self.station_proj(station_attrs)
        depth_ctx = self._depth_context(h_nodes, node_static, node_mask)
        parts = [pooled, emb, depth_ctx]
        if self.state_proj is not None and state_feats is not None:
            parts.append(self.state_proj(state_feats))
        return self.head(torch.cat(parts, dim=-1))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    scaler: dict,
    horizons: tuple[int, ...],
    device: torch.device,
    *,
    use_state: bool,
    target_depth: str | None = None,
    target_kind: str = "delta",
    abs_scaler: dict[str, Any] | None = None,
    vt_lookup: dict[tuple[str, str], float] | None = None,
) -> dict[str, Any]:
    model.eval()
    rows = []
    huber_sum = 0.0
    huber_w = 0.0
    target_kind = str(target_kind).lower()
    for batch in loader:
        kwargs = dict(
            node_x=batch["node_x"].to(device),
            node_mask=batch["node_mask"].to(device),
            node_static=batch["node_static"].to(device),
            station_attrs=batch["station_attrs"].to(device),
        )
        if use_state:
            pred = model(**kwargs, state_feats=batch["origin_feats"].to(device))
        else:
            pred = model(**kwargs)
        y_s = batch["y_scaled"].to(device)
        ym_t = batch["y_mask"].to(device)
        w = float(ym_t.sum().item())
        if w > 0:
            huber_sum += float(masked_huber(pred, y_s, ym_t).item()) * w
            huber_w += w
        y = batch["y"].cpu().numpy()
        mask = batch["y_mask"].cpu().numpy()
        pred_np = pred.cpu().numpy()
        for i in range(len(batch["station"])):
            for hi, h in enumerate(horizons):
                if mask[i, hi] <= 0:
                    continue
                ts = pd.Timestamp(batch["date"][i])
                # Keep full tip timestamp for 15-min; calendar date for daily.
                date_str = ts.isoformat(sep=" ") if (ts.hour or ts.minute or ts.second) else str(ts.date())
                if target_kind == "absolute":
                    # Residual decoder: train on Δ; report absolute SM = Vt + Δ̂.
                    st = str(batch["station"][i])
                    vt = float("nan")
                    if vt_lookup is not None:
                        # Tip key for 15-min; calendar date for daily.
                        k_tip = (st, _origin_time_key(ts))
                        k_day = (st, str(pd.Timestamp(ts).normalize().date()))
                        vt = float(vt_lookup.get(k_tip, vt_lookup.get(k_day, float("nan"))))
                    dhat = float(
                        inverse_target(
                            pred_np[i, hi], h, scaler, depth=target_depth
                        )
                    )
                    # y stored as Δ under residual-absolute; convert both sides.
                    y_abs = float(vt) + float(y[i, hi]) if np.isfinite(vt) else float("nan")
                    yhat = float(vt) + dhat if np.isfinite(vt) else float("nan")
                    rows.append(
                        {
                            "station": st,
                            "date": date_str,
                            "horizon": int(h),
                            "y": y_abs,
                            "yhat": yhat,
                            "vt": vt,
                        }
                    )
                else:
                    rows.append(
                        {
                            "station": batch["station"][i],
                            "date": date_str,
                            "horizon": int(h),
                            "y": float(y[i, hi]),
                            "yhat": float(
                                inverse_target(
                                    pred_np[i, hi], h, scaler, depth=target_depth
                                )
                            ),
                        }
                    )
    pred_df = pd.DataFrame(rows)
    if target_kind == "absolute":
        scored = _score_pred_df_absolute(pred_df, horizons)
    else:
        scored = s1.score_pred_df(pred_df, horizons)
    out = {**scored, "pred_df": pred_df}
    out["val_huber"] = float(huber_sum / huber_w) if huber_w > 0 else float("nan")
    return out


def normalize_station_id(name: str) -> str:
    """Accept MB9 / RISMA_MB9 / mb9 → RISMA_MB9."""
    s = str(name).strip()
    if not s:
        raise ValueError("empty station id")
    if not s.upper().startswith("RISMA_"):
        s = f"RISMA_{s}"
    # Preserve MB/SK numeric casing used in parquet stems.
    prefix, _, rest = s.partition("_")
    return f"{prefix.upper()}_{rest.upper()}"


# Hasan & Mahfuz (2026) SK4 chronological fractions ≈ 60/20/20 on 2014–2021.
_HASAN_SK4_SPLITS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train": (pd.Timestamp("2014-01-01"), pd.Timestamp("2017-12-31")),
    "val": (pd.Timestamp("2018-01-01"), pd.Timestamp("2019-12-31")),
    "test": (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
}

# Our calendar: SK4 100cm (through 2023-09) and 150cm (from 2023+) both appear in test.
_COVER100_SPLITS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train": (pd.Timestamp("2013-01-01"), pd.Timestamp("2019-12-31")),
    "val": (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
    "test": (pd.Timestamp("2022-01-01"), pd.Timestamp("2025-12-31")),
}

_CHRONO_SPLIT_PRESETS: dict[str, dict[str, tuple[pd.Timestamp, pd.Timestamp]]] = {
    "hasan_sk4": _HASAN_SK4_SPLITS,
    "cover100": _COVER100_SPLITS,
}


def _vwc_mean_cols(columns: list[str] | pd.Index, depth_key: str) -> list[str]:
    """Mean VWC sensor columns for a target depth key (exclude min/max/tier)."""
    cols = [str(c) for c in columns]
    if depth_key in {"0_to_5", "0to5"}:
        pref = "vwc_0to5cm_s"
    else:
        pref = f"vwc_{depth_key}cm_s"
    out = []
    for c in cols:
        if not c.startswith(pref):
            continue
        if any(tok in c for tok in ("_min", "_max", "tier", "ice")):
            continue
        out.append(c)
    return sorted(out)


def _origin_time_key(ts: pd.Timestamp) -> str:
    """String key for Vt lookup: tip iso for 15-min, calendar date for daily."""
    t = pd.Timestamp(ts)
    if t.hour or t.minute or t.second or t.microsecond:
        return t.isoformat(sep=" ")
    return str(t.normalize().date())


def load_daily_vwc_lookup(
    parquet_dir: Path,
    stations: list[str],
    depth_key: str,
) -> dict[tuple[str, pd.Timestamp], float]:
    """(station, calendar date) → day-mean fused VWC at depth."""
    lookup: dict[tuple[str, pd.Timestamp], float] = {}
    for st in stations:
        path = Path(parquet_dir) / f"{st}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        elif "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        else:
            raise RuntimeError(f"{path} missing date/timestamp")
        cols = _vwc_mean_cols(df.columns, depth_key)
        if not cols:
            continue
        lev = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        g = (
            pd.DataFrame({"date": df["date"], "vwc": lev})
            .groupby("date", as_index=True)["vwc"]
            .mean()
        )
        for d, v in g.items():
            if np.isfinite(v):
                lookup[(st, pd.Timestamp(d).normalize())] = float(v)
    return lookup


def load_tip_vwc_lookup(
    parquet_dir: Path,
    stations: list[str],
    depth_key: str,
) -> dict[tuple[str, pd.Timestamp], float]:
    """(station, tip timestamp) → fused tip VWC at depth (15-min tip→tip)."""
    lookup: dict[tuple[str, pd.Timestamp], float] = {}
    for st in stations:
        path = Path(parquet_dir) / f"{st}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "timestamp" not in df.columns:
            raise RuntimeError(f"{path} missing timestamp for tip lookup")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cols = _vwc_mean_cols(df.columns, depth_key)
        if not cols:
            continue
        lev = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        for ts, v in zip(df["timestamp"], lev):
            if np.isfinite(v):
                lookup[(st, pd.Timestamp(ts))] = float(v)
    return lookup


def attach_vt_for_residual_absolute(
    full: RismaGraphDataset,
    *,
    depth_key: str,
    parquet_dir: Path,
    resolution: str = "daily",
) -> dict[str, Any]:
    """Residual-absolute: train Δ = VWC(t+h)−VWC(t); score absolute = Vt+Δ̂.

    Daily: day-mean VWC. 15-min: tip→tip VWC at exact origin/future tips.
    Rebuilds Δ from the same lookup used for Vt so absolute labels are true V(t+h).
    """
    resolution = str(resolution).lower()
    stations = sorted({str(full._stations[i]) for i in range(len(full))})
    if resolution == "15min":
        raw_lookup = load_tip_vwc_lookup(parquet_dir, stations, depth_key)
    else:
        raw_lookup = load_daily_vwc_lookup(parquet_dir, stations, depth_key)
    n = len(full)
    horizons = tuple(int(h) for h in full.horizons)
    vt = np.full(n, np.nan, dtype=np.float64)
    y = full._y.astype(np.float64).copy()
    y_m = full._y_m.astype(np.float64).copy()
    n_rebuilt = 0
    n_masked_fut = 0
    for i in range(n):
        st = str(full._stations[i])
        t0 = pd.Timestamp(full._dates[i])
        key0 = t0 if resolution == "15min" else t0.normalize()
        v0 = raw_lookup.get((st, key0))
        if v0 is None or not np.isfinite(v0):
            y_m[i, :] = 0.0
            continue
        vt[i] = float(v0)
        for hi, h in enumerate(horizons):
            if float(y_m[i, hi]) <= 0:
                continue
            t1 = t0 + pd.Timedelta(days=int(h))
            key1 = t1 if resolution == "15min" else t1.normalize()
            v1 = raw_lookup.get((st, key1))
            if v1 is None or not np.isfinite(v1):
                y_m[i, hi] = 0.0
                n_masked_fut += 1
                continue
            y[i, hi] = float(v1) - float(v0)
            n_rebuilt += 1
    y_s = np.zeros_like(y, dtype=np.float64)
    for hi, h in enumerate(horizons):
        std = float(_target_spec(full.scaler, int(h), str(depth_key))["std"])
        if not np.isfinite(std) or std <= 0:
            raise RuntimeError(f"bad target std depth={depth_key} h={h}: {std}")
        y_s[:, hi] = y[:, hi] / std
    y = np.where(y_m > 0, y, 0.0)
    y_s = np.where(y_m > 0, y_s, 0.0)
    full._y = y.astype(np.float32)
    full._y_s = y_s.astype(np.float32)
    full._y_m = y_m.astype(np.float32)
    full._vt = vt.astype(np.float32)
    meta = {
        "target_kind": "absolute",
        "absolute_decoder": "residual_vt_plus_delta",
        "absolute_labels": "true_vwc_t_plus_h",
        "delta_source": "vwc_lookup_minus_vt",
        "resolution": resolution,
        "depth": str(depth_key),
        "n_vt": int(np.isfinite(vt).sum()),
        "horizons": list(horizons),
        "n_masked_no_vt": int((~np.isfinite(vt)).sum()),
        "n_delta_rebuilt": int(n_rebuilt),
        "n_masked_missing_future": int(n_masked_fut),
    }
    print(
        f"  target_kind=absolute residual_vt+delta true_VWC(t+h) res={resolution} "
        f"depth={depth_key} vt_ok={meta['n_vt']}/{n} rebuilt_delta={n_rebuilt} "
        f"masked_no_vt={meta['n_masked_no_vt']} masked_no_fut={n_masked_fut}",
        flush=True,
    )
    return meta


# Back-compat alias used by QA / older call sites.
def convert_targets_to_absolute(
    full: RismaGraphDataset,
    *,
    train_idx: list[int],
    depth_key: str,
    parquet_dir: Path,
    resolution: str = "daily",
) -> dict[str, Any]:
    del train_idx  # residual path does not refit an absolute z-scaler
    return attach_vt_for_residual_absolute(
        full,
        depth_key=depth_key,
        parquet_dir=parquet_dir,
        resolution=resolution,
    )


def _score_pred_df_absolute(
    pred_df: pd.DataFrame, horizons: tuple[int, ...]
) -> dict[str, Any]:
    """RMSE on absolute SM; skill/regimes on Δ from Vt (persistence = hold Vt)."""
    pred_df = pred_df.copy()
    if "vt" not in pred_df.columns:
        raise RuntimeError("absolute scoring requires vt column")
    pred_df["dy"] = pred_df["y"] - pred_df["vt"]
    pred_df["dyhat"] = pred_df["yhat"] - pred_df["vt"]
    pred_df["regime"] = "quiet"
    pred_df.loc[pred_df["dy"] > 0.01, "regime"] = "wetting"
    pred_df.loc[pred_df["dy"] < -0.01, "regime"] = "drying"
    out: dict[str, Any] = {"per_horizon": {}, "regimes": {}}
    skills: list[float] = []
    for h in horizons:
        sub = pred_df[pred_df.horizon == h]
        sub = sub[np.isfinite(sub.y) & np.isfinite(sub.yhat) & np.isfinite(sub.vt)]
        y = sub.y.to_numpy(float)
        p = sub.yhat.to_numpy(float)
        dy = sub.dy.to_numpy(float)
        dp = sub.dyhat.to_numpy(float)
        rmse = float(np.sqrt(np.mean((p - y) ** 2))) if len(sub) else float("nan")
        sk = skill_vs_persistence(dy, dp) if len(sub) else float("nan")
        out["per_horizon"][str(h)] = {"n": int(len(sub)), "rmse": rmse, "skill": sk}
        skills.append(sk)
        for reg in ("wetting", "drying", "quiet"):
            g = sub[sub.regime == reg]
            key = f"h{h}_{reg}"
            if len(g) < 5:
                out["regimes"][key] = {"n": int(len(g)), "rmse": float("nan")}
            else:
                yy, pp = g.y.to_numpy(float), g.yhat.to_numpy(float)
                out["regimes"][key] = {
                    "n": int(len(g)),
                    "rmse": float(np.sqrt(np.mean((pp - yy) ** 2))),
                    "bias": float(np.mean(pp - yy)),
                }
    out["mean_horizon_skill"] = float(np.nanmean(skills)) if skills else float("nan")
    return out


def train_one(
    *,
    variant: str,
    depth_mode: str,
    use_rain_state: bool,
    seeds: tuple[int, ...],
    max_epochs: int,
    patience: int,
    device: torch.device,
    out_dir: Path,
    holdout_stations: tuple[str, ...] = (),
    allow_stations: tuple[str, ...] = (),
    split_preset: str = "stage0",
    target_kind: str = "delta",
    resolution: str = "daily",
    shuffle_train: bool = True,
    data_parallel: bool = False,
    ddp: bool = False,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    ddp_train_stations: tuple[str, ...] = (),
    num_workers: int = 4,
    target_depth: str = "20",
    hidden: int = 64,
    dropout: float = 0.3,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    scheduler_patience: int = 8,
    es_metric: str = "huber",
    window_days: int | None = None,
    batch_size: int | None = None,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    resolution = str(resolution).lower()
    target_depth = str(target_depth)
    use_ddp = bool(ddp) and int(ddp_world_size) > 1
    rank = int(ddp_rank)
    world_size = int(ddp_world_size) if use_ddp else 1
    main = _is_main(rank)
    if resolution == "15min":
        schema_path = _SCHEMA_15MIN
        parquet_dir = _STEP10_15 / "parquet_model12_15min"
        origins_path = _STEP10_15 / "origins_long_15min.csv"
        scaler_path = _STEP10_15 / "step10b_scaled" / "scaler.json"
        origins_scaled = _STEP10_15 / "step10b_scaled" / "origins_scaled_node.csv"
        schema = json.loads(schema_path.read_text())
        steps_per_day = 96
        if window_days is not None:
            window_steps = int(window_days) * steps_per_day
        else:
            window_steps = int(schema.get("time", {}).get("input_window_steps", 192))
            window_days = int(schema.get("time", {}).get("input_window_days", window_steps // steps_per_day))
        meta = {
            "scaler_path": str(scaler_path),
            "horizons": list(schema.get("horizons_days", [1, 3, 7, 14])),
            # loader uses this as tip-step count for 15min
            "window_days": window_steps,
            "window_calendar_days": int(window_days),
            "schema_path": str(schema_path),
            "resolution": "15min",
        }
        print(
            f"  resolution=15min window_days={window_days} tips={window_steps} "
            f"target_depth={target_depth} scaler={scaler_path}",
            flush=True,
        )
        from .split_config import load_split_config

        split = load_split_config()
        gap_days = int((split.val_start.normalize() - split.train_end.normalize()).days)
        max_h = max(int(h) for h in meta["horizons"])
        need_gap = int(window_days) + max_h
        if gap_days < need_gap:
            print(
                f"  [warn] split gap={gap_days}d < window({window_days})+maxH({max_h})={need_gap}d; "
                f"train targets can overlap val input tips near the boundary",
                flush=True,
            )
    else:
        # Prefer multi-depth daily origins/schema when present (all-depth matrix).
        use_md = (
            _ORIGINS_DAILY_MD.exists()
            and _SCHEMA_DAILY_MD.exists()
            and (_SCALED_DAILY_MD / "scaler.json").exists()
        )
        if use_md:
            schema_path = _SCHEMA_DAILY_MD
            scaler_path = _SCALED_DAILY_MD / "scaler.json"
            origins_scaled = _SCALED_DAILY_MD / "origins_scaled_node.csv"
            origins_path = _ORIGINS_DAILY_MD
        elif _E1_META.exists():
            schema_path = _SCHEMA_DAILY
            meta_e1 = json.loads(_E1_META.read_text())
            scaler_path = Path(meta_e1.get("scaler_path", _STEP10 / "step10b_scaled" / "scaler.json"))
            origins_path = _ORIGINS
            origins_scaled = _STEP10 / "step10b_scaled" / "origins_scaled_node.csv"
        else:
            schema_path = _SCHEMA_DAILY
            scaler_path = _STEP10 / "step10b_scaled" / "scaler.json"
            origins_path = _ORIGINS
            origins_scaled = _STEP10 / "step10b_scaled" / "origins_scaled_node.csv"
            print(f"  [warn] missing {_E1_META}; using schema/scaler fallback", flush=True)
        schema = json.loads(schema_path.read_text())
        win = int(
            window_days
            if window_days is not None
            else schema.get("time", {}).get("input_window_days", 30)
        )
        meta = {
            "scaler_path": str(scaler_path),
            "horizons": list(schema.get("horizons_days", [1, 3, 7, 14])),
            "window_days": win,
            "window_calendar_days": win,
            "schema_path": str(schema_path),
            "resolution": "daily",
            "multidepth": bool(use_md),
        }
        parquet_dir = _PARQUET
        print(
            f"  resolution=daily window_days={win} target_depth={target_depth} "
            f"multidepth={use_md} scaler={scaler_path}",
            flush=True,
        )
        from .split_config import load_split_config

        split = load_split_config()
        gap_days = int((split.val_start.normalize() - split.train_end.normalize()).days)
        max_h = max(int(h) for h in meta["horizons"])
        need_gap = int(win) + max_h
        if gap_days < need_gap:
            print(
                f"  [warn] split gap={gap_days}d < window({win})+maxH({max_h})={need_gap}d; "
                f"train targets can overlap val input days near the boundary",
                flush=True,
            )

    scaler = json.loads(Path(meta["scaler_path"]).read_text())
    horizons = tuple(int(h) for h in meta["horizons"])
    schema_for_depths = json.loads(Path(schema_path).read_text())
    depths_cm = tuple(cm for cm, _ in schema_depth_specs(schema_for_depths))
    holdout = tuple(normalize_station_id(s) for s in holdout_stations)
    holdout_set = set(holdout)
    allow = tuple(normalize_station_id(s) for s in allow_stations)
    allow_set = set(allow)
    split_preset = str(split_preset or "stage0").lower()
    if split_preset not in {"stage0", *_CHRONO_SPLIT_PRESETS}:
        raise ValueError(f"unknown split_preset={split_preset!r}")
    if allow_set and holdout_set and (allow_set & holdout_set):
        raise ValueError(
            f"--stations and --holdout-station overlap: {sorted(allow_set & holdout_set)}"
        )
    # Plain td_* does not consume LGBM origin feats; rain-C path builds its own state lookup.
    full = RismaGraphDataset(
        origins_path=origins_path,
        origins_scaled_path=origins_scaled,
        parquet_dir=parquet_dir,
        scaler_path=meta["scaler_path"],
        schema_path=schema_path,
        block=None,
        window_days=int(meta["window_days"]),
        include_origin_feats=False,
        target_depth=target_depth,
    )

    def _station_ok(i: int) -> bool:
        if allow_set and full._stations[i] not in allow_set:
            return False
        return True

    def _in_range(i: int, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        d = pd.Timestamp(full._dates[i])
        return bool(start <= d <= end)

    # LOFO / fault-tolerance: drop holdouts from train+val only; keep them in test.
    # Chrono presets override Stage0 block labels (Hasan years or cover100 calendar).
    if split_preset in _CHRONO_SPLIT_PRESETS:
        ranges = _CHRONO_SPLIT_PRESETS[split_preset]
        tr0, tr1 = ranges["train"]
        va0, va1 = ranges["val"]
        te0, te1 = ranges["test"]
        train_idx = [
            i
            for i in range(len(full))
            if _station_ok(i) and full._stations[i] not in holdout_set and _in_range(i, tr0, tr1)
        ]
        val_idx = [
            i
            for i in range(len(full))
            if _station_ok(i) and full._stations[i] not in holdout_set and _in_range(i, va0, va1)
        ]
        test_idx = [
            i for i in range(len(full)) if _station_ok(i) and _in_range(i, te0, te1)
        ]
        if main:
            print(
                f"  split_preset={split_preset} stations={sorted(allow_set) or 'ALL'} "
                f"train={tr0.date()}..{tr1.date()} val={va0.date()}..{va1.date()} "
                f"test={te0.date()}..{te1.date()} "
                f"n={len(train_idx)}/{len(val_idx)}/{len(test_idx)}",
                flush=True,
            )
            (out_dir / "split_preset_meta.json").write_text(
                json.dumps(
                    {
                        "split_preset": split_preset,
                        "allow_stations": sorted(allow_set),
                        "ranges": {
                            k: [str(a.date()), str(b.date())] for k, (a, b) in ranges.items()
                        },
                        "n_train": len(train_idx),
                        "n_val": len(val_idx),
                        "n_test": len(test_idx),
                        "target_depth": target_depth,
                    },
                    indent=2,
                )
                + "\n"
            )
    else:
        train_idx = [
            i
            for i in range(len(full))
            if _station_ok(i)
            and full._blocks[i] == "train"
            and full._stations[i] not in holdout_set
        ]
        val_idx = [
            i
            for i in range(len(full))
            if _station_ok(i)
            and full._blocks[i] == "val"
            and full._stations[i] not in holdout_set
        ]
        test_idx = [
            i
            for i in range(len(full))
            if _station_ok(i) and full._blocks[i] == "test"
        ]
        if allow_set:
            print(
                f"  split_preset=stage0 allow_stations={sorted(allow_set)} "
                f"n={len(train_idx)}/{len(val_idx)}/{len(test_idx)}",
                flush=True,
            )
    if holdout and split_preset == "stage0":
        present = set(full._stations.tolist())
        missing = sorted(holdout_set - present)
        # Multi-depth: some stations have no origins at this depth (e.g. MB @ 150 cm).
        # Keep requested holdouts that exist; warn and skip the rest.
        if missing:
            print(
                f"  [warn] holdout stations absent at depth={target_depth}: {missing} "
                f"(no origins for this depth; ignored)",
                flush=True,
            )
        active_holdout = sorted(holdout_set & present)
        holdout_set = set(active_holdout)
        # Rebuild train/val with the active holdout set only.
        train_idx = [
            i
            for i in range(len(full))
            if _station_ok(i)
            and full._blocks[i] == "train"
            and full._stations[i] not in holdout_set
        ]
        val_idx = [
            i
            for i in range(len(full))
            if _station_ok(i)
            and full._blocks[i] == "val"
            and full._stations[i] not in holdout_set
        ]
        test_idx = [
            i
            for i in range(len(full))
            if _station_ok(i) and full._blocks[i] == "test"
        ]
        n_hold_test = sum(1 for i in test_idx if full._stations[i] in holdout_set)
        print(
            f"  holdout_requested={list(holdout)} holdout_active={active_holdout} "
            f"splits train/val/test="
            f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)} "
            f"(holdout_in_test={n_hold_test})",
            flush=True,
        )
        (out_dir / "holdout_meta.json").write_text(
            json.dumps(
                {
                    "holdout_stations_requested": list(holdout),
                    "holdout_stations": active_holdout,
                    "holdout_stations_absent_at_depth": missing,
                    "target_depth": target_depth,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "n_test": len(test_idx),
                    "n_holdout_in_test": n_hold_test,
                },
                indent=2,
            )
            + "\n"
        )
    if not train_idx or not val_idx or not test_idx:
        raise RuntimeError(
            f"empty split after holdout={list(holdout)}: "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )

    # Region-per-rank train shard (peer all-reduce). Test stays global for scoring.
    # Val used for early-stopping is restricted to the union of training regions
    # (no held-out cluster leakage into model selection).
    region_train = tuple(normalize_station_id(s) for s in ddp_train_stations)
    region_train_set = set(region_train)
    region_union: set[str] = set(region_train_set)
    region_stations_by_rank: list[list[str]] = []
    if region_train_set:
        train_idx = [i for i in train_idx if full._stations[i] in region_train_set]
        if not train_idx:
            raise RuntimeError(
                f"DDP rank {rank} has empty train after region filter {sorted(region_train_set)}"
            )
        if use_ddp:
            gathered: list[list[str] | None] = [None] * world_size
            dist.all_gather_object(gathered, sorted(region_train_set))
            region_stations_by_rank = [list(g or []) for g in gathered]
            region_union = {s for g in region_stations_by_rank for s in g}
        else:
            region_stations_by_rank = [sorted(region_train_set)]
            region_union = set(region_train_set)
        n_val_before = len(val_idx)
        val_idx = [i for i in val_idx if full._stations[i] in region_union]
        if not val_idx:
            raise RuntimeError(
                f"DDP rank {rank} has empty val after region-union filter {sorted(region_union)}"
            )
        if main:
            print(
                f"  ddp region-partition rank={rank} train_stations={sorted(region_train_set)} "
                f"n_train={len(train_idx)} region_union={sorted(region_union)} "
                f"val_filtered={n_val_before}->{len(val_idx)} (test remains all stations)",
                flush=True,
            )

    target_kind = str(target_kind or "delta").lower()
    if target_kind not in {"delta", "absolute"}:
        raise ValueError(f"unknown target_kind={target_kind!r}")
    abs_meta: dict[str, Any] | None = None
    vt_lookup: dict[tuple[str, str], float] | None = None
    if target_kind == "absolute":
        abs_meta = attach_vt_for_residual_absolute(
            full,
            depth_key=str(target_depth),
            parquet_dir=parquet_dir,
            resolution=resolution,
        )
        # Drop samples that lost all horizon masks after Vt join.
        def _any_mask(idxs: list[int]) -> list[int]:
            return [i for i in idxs if float(full._y_m[i].sum()) > 0]

        train_idx = _any_mask(train_idx)
        val_idx = _any_mask(val_idx)
        test_idx = _any_mask(test_idx)
        if not train_idx or not val_idx or not test_idx:
            raise RuntimeError(
                f"empty split after absolute Vt join: "
                f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
            )
        vt_lookup = {
            (
                str(full._stations[i]),
                _origin_time_key(pd.Timestamp(full._dates[i])),
            ): float(full._vt[i])
            for i in range(len(full))
            if np.isfinite(full._vt[i])
        }
        (out_dir / "abs_target_scaler.json").write_text(json.dumps(abs_meta, indent=2) + "\n")

    lookup = None
    n_state = 0
    if use_rain_state:
        s1.STATE_FEATURE_NAMES = list(RAIN_C_STATE)
        train_keys = pd.DataFrame(
            {"station": [full._stations[i] for i in train_idx], "date": [full._dates[i] for i in train_idx]}
        )
        lookup, names, stats = s1.build_state_lookup(_PARQUET, train_keys, use_intraday=True)
        n_state = len(names)
        (out_dir / "state_feature_names.json").write_text(json.dumps(names, indent=2))
        print(f"  rain state feats={names}", flush=True)

    sample = full[train_idx[0]]
    metric_rows = []
    test_preds: dict[int, pd.DataFrame] = {}

    for seed in seeds:
        set_seed(seed)
        pin = device.type == "cuda"
        nw = max(0, int(num_workers))
        dl_common: dict[str, Any] = dict(num_workers=nw, pin_memory=pin)
        if nw > 0:
            dl_common["persistent_workers"] = True
            dl_common["prefetch_factor"] = 2
        # Keep shuffle order reproducible with worker processes.
        g = torch.Generator()
        g.manual_seed(int(seed))
        # Longer tip windows need smaller batches to fit GPU memory.
        bs_train = int(batch_size) if batch_size is not None else 128
        if resolution == "15min" and window_days is not None and batch_size is None:
            wd = int(window_days)
            if wd >= 30:
                bs_train = 8
            elif wd >= 14:
                bs_train = 32
            elif wd >= 7:
                bs_train = 64
        elif resolution == "daily" and window_days is not None and batch_size is None:
            # Daily windows are short in steps; keep default unless overridden.
            bs_train = 128
        # Sharded DDP: keep global batch ≈ single-process default.
        # Region partition: each rank already has exclusive stations → keep full per-rank batch.
        if use_ddp and batch_size is None and not region_train_set:
            bs_train = max(1, int(bs_train) // world_size)
        bs_eval = max(bs_train, min(256, bs_train * 2))
        if main:
            if region_train_set:
                gbatch_msg = f"per_rank_batch={bs_train} (region partition; not bs×world)"
            else:
                gbatch_msg = f"global_batch≈{bs_train * world_size}"
            print(
                f"  dataloader num_workers={nw} pin_memory={pin} "
                f"batch_train={bs_train} batch_eval={bs_eval} "
                f"ddp={use_ddp} world_size={world_size} {gbatch_msg}",
                flush=True,
            )
        train_sampler = None
        # Region partition already assigns exclusive stations per rank → no DistributedSampler.
        use_dist_sampler = use_ddp and not region_train_set
        if use_rain_state:
            tr_ds = HybridGraphDataset(full, train_idx, feat_lookup=lookup, n_feats=n_state)
            va_ds = HybridGraphDataset(full, val_idx, feat_lookup=lookup, n_feats=n_state)
            te_ds = HybridGraphDataset(full, test_idx, feat_lookup=lookup, n_feats=n_state)

            def collate(batch):
                samples = [b[0] for b in batch]
                out = collate_graphs(samples)
                out["origin_feats"] = torch.stack([b[1] for b in batch], dim=0)
                return out

            if use_dist_sampler:
                train_sampler = DistributedSampler(
                    tr_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=int(seed)
                )
            tr = DataLoader(
                tr_ds,
                batch_size=bs_train,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                collate_fn=collate,
                generator=g if train_sampler is None else None,
                **dl_common,
            )
            va = DataLoader(va_ds, batch_size=bs_eval, shuffle=False, collate_fn=collate, **dl_common)
            te = DataLoader(te_ds, batch_size=bs_eval, shuffle=False, collate_fn=collate, **dl_common)
        else:
            tr_subset = Subset(full, train_idx)
            if use_dist_sampler:
                train_sampler = DistributedSampler(
                    tr_subset, num_replicas=world_size, rank=rank, shuffle=True, seed=int(seed)
                )
            tr = DataLoader(
                tr_subset,
                batch_size=bs_train,
                shuffle=(train_sampler is None and shuffle_train),
                sampler=train_sampler,
                collate_fn=collate_graphs,
                generator=g if train_sampler is None else None,
                **dl_common,
            )
            va = DataLoader(
                Subset(full, val_idx), batch_size=bs_eval, shuffle=False, collate_fn=collate_graphs, **dl_common
            )
            te = DataLoader(
                Subset(full, test_idx), batch_size=bs_eval, shuffle=False, collate_fn=collate_graphs, **dl_common
            )

        model = TimeDepthGRU(
            f_dyn=int(sample.node_x.shape[-1]),
            n_static=int(sample.node_static.shape[-1]),
            n_station_attrs=int(sample.station_attrs.shape[-1]),
            n_state=n_state,
            hidden=hidden,
            dropout=dropout,
            n_horizons=len(horizons),
            depth_mode=depth_mode,
            depths_cm=depths_cm,
            target_depth_cm=target_depth_to_cm(target_depth),
        ).to(device)
        # DataParallel: same DataLoader order / batch_size; each batch is split across GPUs.
        # Keeps seed-0 sample sequence identical to single-GPU (unlike DistributedSampler).
        n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
        if use_ddp:
            if device.type == "cuda":
                model = DDP(model, device_ids=[device.index], output_device=device.index)
            else:
                model = DDP(model)  # gloo CPU peer all-reduce
            if main:
                print(
                    f"  DDP peer all-reduce world_size={world_size} rank={rank} device={device}",
                    flush=True,
                )
        elif data_parallel and n_gpus > 1:
            model = nn.DataParallel(model)
            if main:
                print(
                    f"  DataParallel gpus={n_gpus} batch_size=128 (sequence preserved)",
                    flush=True,
                )

        def _raw_state_dict() -> dict[str, torch.Tensor]:
            raw = _unwrap_model(model)
            return {k: v.detach().cpu().clone() for k, v in raw.state_dict().items()}

        def _load_raw_state(state: dict[str, torch.Tensor]) -> None:
            _unwrap_model(model).load_state_dict(state)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = None
        if scheduler_patience > 0:
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            scheduler = ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=scheduler_patience, min_lr=1e-6
            )
        es_huber = es_metric == "huber"
        best_score = 1e9 if es_huber else -1e9
        best_skill = float("nan")
        best_val_huber = float("nan")
        best_state = None
        best_epoch = 0
        bad = 0
        history: list[dict[str, Any]] = []
        t0 = time.time()
        hold_tag = ",".join(holdout) if holdout else "none"
        if main:
            print(
                f"=== {variant} seed={seed} depth={depth_mode} state={use_rain_state} "
                f"holdout={hold_tag} dropout={dropout} es={es_metric} "
                f"ddp={use_ddp} ws={world_size} ===",
                flush=True,
            )
        for epoch in range(1, max_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            losses = []
            # Region partition: unequal n → lockstep on MAX batches; smaller ranks cycle.
            n_steps = len(tr)
            if use_ddp and region_train_set:
                t_steps = torch.tensor([n_steps], dtype=torch.long, device=device)
                dist.all_reduce(t_steps, op=dist.ReduceOp.MAX)
                n_steps = int(t_steps.item())
            batch_iter = iter(tr)

            def _next_batch():
                nonlocal batch_iter
                try:
                    return next(batch_iter)
                except StopIteration:
                    batch_iter = iter(tr)
                    return next(batch_iter)

            for _ in range(n_steps):
                batch = _next_batch()
                opt.zero_grad(set_to_none=True)
                if use_rain_state:
                    pred = model(
                        batch["node_x"].to(device),
                        batch["node_mask"].to(device),
                        batch["node_static"].to(device),
                        batch["station_attrs"].to(device),
                        batch["origin_feats"].to(device),
                    )
                else:
                    pred = model(
                        batch["node_x"].to(device),
                        batch["node_mask"].to(device),
                        batch["node_static"].to(device),
                        batch["station_attrs"].to(device),
                    )
                loss = masked_huber(pred, batch["y_scaled"].to(device), batch["y_mask"].to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.item()))

            val_m = evaluate(
                model,
                va,
                scaler,
                horizons,
                device,
                use_state=use_rain_state,
                target_depth=target_depth,
                target_kind=target_kind,
                abs_scaler=None,
                vt_lookup=vt_lookup,
            )
            skill = val_m["mean_horizon_skill"]
            val_huber = float(val_m.get("val_huber", float("nan")))
            train_loss = float(np.mean(losses)) if losses else float("nan")
            if scheduler is not None and np.isfinite(val_huber):
                scheduler.step(val_huber)
            lr_now = float(opt.param_groups[0]["lr"])
            # Rank 0 owns early-stopping; broadcast improved/stop so peers exit together.
            if es_huber:
                improved_local = np.isfinite(val_huber) and val_huber < best_score - 1e-5
            else:
                improved_local = np.isfinite(skill) and skill > best_score + 1e-4
            if main:
                improved = bool(improved_local)
                if improved:
                    best_score = float(val_huber if es_huber else skill)
                    best_epoch = epoch
                    best_skill = float(skill) if np.isfinite(skill) else best_skill
                    best_val_huber = float(val_huber) if np.isfinite(val_huber) else best_val_huber
                    best_state = _raw_state_dict()
                    bad = 0
                else:
                    bad += 1
                stop_now = bad >= patience
            else:
                improved = False
                stop_now = False
            if use_ddp:
                flags = torch.tensor(
                    [1 if improved else 0, 1 if stop_now else 0],
                    dtype=torch.int32,
                    device=device,
                )
                dist.broadcast(flags, src=0)
                improved = bool(flags[0].item())
                stop_now = bool(flags[1].item())
                if not main and improved:
                    best_epoch = epoch
                    best_skill = float(skill) if np.isfinite(skill) else best_skill
                    best_val_huber = float(val_huber) if np.isfinite(val_huber) else best_val_huber
                    best_state = _raw_state_dict()
            ph = val_m["per_horizon"]
            h1 = ph["1"]["rmse"]
            h3 = ph["3"]["rmse"]
            h7 = ph.get("7", {}).get("rmse", float("nan"))
            h14 = ph.get("14", {}).get("rmse", float("nan"))
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_huber": val_huber,
                    "val_skill": float(skill) if np.isfinite(skill) else float("nan"),
                    "val_h1_rmse": h1,
                    "val_h3_rmse": h3,
                    "val_h7_rmse": h7,
                    "val_h14_rmse": h14,
                    "lr": lr_now,
                    "improved": bool(improved),
                }
            )
            if main:
                print(
                    f"  {variant} ep={epoch:03d} loss={train_loss:.4f} val_huber={val_huber:.5f} "
                    f"val_skill={skill:.4f} h1={h1:.5f} h3={h3:.5f} h7={h7:.5f} h14={h14:.5f} "
                    f"lr={lr_now:.2e} {'*' if improved else ''} bad={bad}",
                    flush=True,
                )
            if stop_now:
                break

        if best_state is not None:
            _load_raw_state(best_state)
        if use_ddp:
            # Ensure every rank loads the same best checkpoint weights.
            obj = [best_state if main else None]
            dist.broadcast_object_list(obj, src=0)
            best_state = obj[0]
            if best_state is not None:
                _load_raw_state(best_state)
            dist.barrier()
        if main:
            ckpt = out_dir / "checkpoints" / f"{variant}_seed{seed}.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, ckpt)
            pd.DataFrame(history).to_csv(out_dir / f"history_{variant}_seed{seed}.csv", index=False)
            (out_dir / f"train_meta_{variant}_seed{seed}.json").write_text(
                json.dumps(
                    {
                        "variant": variant,
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "best_val_skill": best_skill,
                        "best_val_huber": best_val_huber,
                        "es_metric": es_metric,
                        "dropout": dropout,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "hidden": hidden,
                        "patience": patience,
                        "scheduler_patience": scheduler_patience,
                        "target_depth": target_depth,
                        "resolution": resolution,
                        "target_kind": target_kind,
                        "absolute_decoder": (abs_meta or {}).get("absolute_decoder"),
                        "target_depth_cm": target_depth_to_cm(target_depth),
                        "split_preset": split_preset,
                        "allow_stations": sorted(allow_set),
                        "window_days": meta.get("window_calendar_days", meta.get("window_days")),
                        "ddp": use_ddp,
                        "ddp_world_size": world_size,
                        "ddp_backend": (
                            dist.get_backend() if use_ddp and dist.is_initialized() else None
                        ),
                        "ddp_train_stations": sorted(region_union) if region_union else None,
                        "ddp_train_stations_by_rank": region_stations_by_rank or None,
                        "ddp_region_partition": bool(region_train_set),
                        "ddp_val_stations": sorted(region_union) if region_union else None,
                        "ddp_es_synced": True,
                        "ddp_region_lockstep": "max_cycle" if region_train_set else None,
                        "train_batch_size_per_rank": bs_train,
                        "approx_global_batch": (
                            bs_train if region_train_set else bs_train * world_size
                        ),
                    },
                    indent=2,
                )
                + "\n"
            )

        # Rank 0 evaluates + writes full val/test (shared weights after DDP sync).
        if main:
            for block, loader in (("val", va), ("test", te)):
                m = evaluate(
                    model,
                    loader,
                    scaler,
                    horizons,
                    device,
                    use_state=use_rain_state,
                    target_depth=target_depth,
                    target_kind=target_kind,
                    abs_scaler=None,
                    vt_lookup=vt_lookup,
                )
                pred_df = m.pop("pred_df")
                pred_df["variant"] = variant
                pred_df["seed"] = seed
                pred_df["block"] = block
                pred_path = out_dir / "preds" / f"{variant}_seed{seed}_{block}.csv"
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                pred_df.to_csv(pred_path, index=False)
                if block == "test":
                    test_preds[seed] = pred_df
                for h in horizons:
                    ph = m["per_horizon"][str(h)]
                    metric_rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "block": block,
                            "horizon": h,
                            "slice": "all",
                            "rmse": ph["rmse"],
                            "skill": ph["skill"],
                            "n": ph["n"],
                            "best_epoch": best_epoch,
                            "wall_s": time.time() - t0,
                        }
                    )
                for key, regm in m["regimes"].items():
                    hs, reg = key.split("_", 1)
                    metric_rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "block": block,
                            "horizon": int(hs[1:]),
                            "slice": reg,
                            "rmse": regm["rmse"],
                            "skill": float("nan"),
                            "n": regm["n"],
                            "best_epoch": best_epoch,
                            "wall_s": time.time() - t0,
                        }
                    )
                print(
                    f"  {variant} {block}: h1={m['per_horizon']['1']['rmse']:.5f} "
                    f"h3={m['per_horizon']['3']['rmse']:.5f} "
                    f"wet_h1={m['regimes'].get('h1_wetting', {}).get('rmse', float('nan')):.5f}",
                    flush=True,
                )
        if use_ddp:
            dist.barrier()
    return pd.DataFrame(metric_rows), test_preds


def _load_pred(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    d = pd.read_csv(path)
    d["regime"] = "quiet"
    d.loc[d.y > 0.01, "regime"] = "wetting"
    d.loc[d.y < -0.01, "regime"] = "drying"
    return d


def write_report(out_dir: Path, metrics: pd.DataFrame, paired: pd.DataFrame) -> None:
    lines = [
        "# Time × Depth GRU ablation (priority 3)",
        "",
        "Same 20 cm ΔVWC targets / identical-row Stage0 test. Seed 0.",
        "",
        "| variant | meaning |",
        "|--|--|",
        "| td_gru | temporal GRU + depth GRU over {5,20,50} |",
        "| td_mlp | temporal GRU + depth MLP (no depth recurrence) |",
        "| td_gru_rainC | td_gru + Rain C state features |",
        "",
        "## Test RMSE",
        "",
        "| variant | h | all | wetting | quiet | drying |",
        "|--|--:|--:|--:|--:|--:|",
    ]
    te = metrics[metrics.block == "test"]
    for variant in te.variant.unique():
        for h in (1, 3, 7, 14):
            cells = []
            for sl in ("all", "wetting", "quiet", "drying"):
                sub = te[(te.variant == variant) & (te.horizon == h) & (te.slice == sl)]
                cells.append(f"{sub.rmse.mean():.5f}" if len(sub) and np.isfinite(sub.rmse).any() else "—")
            lines.append(f"| {variant} | {h} | " + " | ".join(cells) + " |")
    if not paired.empty:
        lines += [
            "",
            "## Paired ΔRMSE (negative ⇒ row model better than baseline)",
            "",
            "| model | baseline | h | slice | ΔRMSE [95% CI] | n |",
            "|--|--|--:|--|--|--:|",
        ]
        for _, r in paired.iterrows():
            lines.append(
                f"| {r.model} | {r.baseline} | {int(r.horizon)} | {r.slice} | "
                f"{r.delta:.6f} [{r.lo:.6f}, {r.hi:.6f}] | {int(r.n)} |"
            )
    lines += ["", "## Decision", "", "- Keep depth path only if it beats SharedGRU / Rain C on h1/h3 wetting.", ""]
    (out_dir / "TIME_DEPTH_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=_OUT)
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["td_gru", "td_mlp", "td_gru_rainC"],
        help="td_gru | td_mlp | td_gru_rainC",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--max-epochs", type=int, default=80)
    ap.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stop patience (thesis-aligned default 20 on val Huber)",
    )
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--scheduler-patience",
        type=int,
        default=8,
        help="ReduceLROnPlateau patience on val Huber (0 disables)",
    )
    ap.add_argument(
        "--es-metric",
        choices=("huber", "skill"),
        default="huber",
        help="Early-stop / checkpoint on val Huber (default) or val skill (legacy)",
    )
    ap.add_argument(
        "--holdout-station",
        nargs="*",
        default=[],
        help="Stations excluded from train+val but kept in test (e.g. MB9 RISMA_MB11 SK1)",
    )
    ap.add_argument(
        "--stations",
        nargs="*",
        default=[],
        help="If set, keep only these stations in train/val/test (e.g. SK4 for Hasan comparison)",
    )
    ap.add_argument(
        "--split-preset",
        choices=["stage0", "hasan_sk4", "cover100"],
        default="stage0",
        help=(
            "stage0=locked Stage0 calendar blocks; "
            "hasan_sk4=2014–2017 train / 2018–2019 val / 2020–2021 test (Hasan & Mahfuz SK4); "
            "cover100=≤2019 train / 2020–2021 val / 2022–2025 test (SK4 100cm+150cm in test)"
        ),
    )
    ap.add_argument(
        "--resolution",
        choices=["daily", "15min"],
        default="daily",
        help="daily Stage0 or 15-min tip panels (T=192)",
    )
    ap.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable train DataLoader shuffle (station→time order)",
    )
    ap.add_argument(
        "--data-parallel",
        action="store_true",
        help="Wrap model in nn.DataParallel (same batch order/size; split each batch across GPUs)",
    )
    ap.add_argument(
        "--ddp",
        action="store_true",
        help="Enable DistributedDataParallel peer all-reduce (launch with torchrun)",
    )
    ap.add_argument(
        "--ddp-region-layout",
        default="",
        help=(
            "Comma-separated province/cluster codes per rank, e.g. MB,SK or "
            "Carman_Elm_Creek,Sturgeon_Creek,Kenaston. "
            "Rank i trains only on that region's stations; val/test stay global."
        ),
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (pin_memory on when CUDA)",
    )
    ap.add_argument(
        "--target-depth",
        default="20",
        help="Origin depth key to train/eval (0_to_5|5|20|50|100|150)",
    )
    ap.add_argument(
        "--window-days",
        type=int,
        default=None,
        help="Historical input length in days (15min: tips=days×96). A=2, B=7, C=14",
    )
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument(
        "--target-kind",
        choices=["delta", "absolute"],
        default="delta",
        help="delta=ΔVWC (default); absolute=true VWC(t+h) via residual Vt+Δ̂ (daily or 15min tip→tip)",
    )
    args = ap.parse_args()
    ddp_on, rank, world_size, local_rank = _ddp_env()
    if args.ddp and not ddp_on:
        raise SystemExit("--ddp requires launch via torchrun (WORLD_SIZE>1)")
    if ddp_on:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        if _is_main(rank):
            print(
                f"  DDP init backend={backend} world_size={world_size} device={device}",
                flush=True,
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = args.out_dir
    wtag = f"w{args.window_days}d" if args.window_days is not None else (
        "w192" if args.resolution == "15min" else "w30d"
    )
    if out == _OUT:
        if args.resolution == "15min":
            out = (
                _FOLLOWUP_OUT
                / f"time_depth_gru_15min_{wtag}_tiptip_d{args.target_depth}"
            )
        else:
            out = (
                _FOLLOWUP_OUT
                / f"time_depth_gru_daily_{wtag}_d{args.target_depth}"
            )
    if _is_main(rank):
        out.mkdir(parents=True, exist_ok=True)
    if ddp_on:
        dist.barrier()
    holdout = tuple(args.holdout_station)
    stations = tuple(args.stations)
    ddp_train_stations: tuple[str, ...] = ()
    layout = str(args.ddp_region_layout or "").strip()
    if layout:
        parts = [p.strip().upper() for p in layout.split(",") if p.strip()]
        if ddp_on and len(parts) != world_size:
            raise SystemExit(
                f"--ddp-region-layout has {len(parts)} entries but world_size={world_size}"
            )
        region_code = parts[rank] if ddp_on else parts[0]
        attr_path = _PKG / "station_attributes_model12.csv"
        attr = pd.read_csv(attr_path)
        # Aliases for thesis clusters (Carman / Sturgeon / Kenaston).
        cluster_alias = {
            "CARMAN": "Carman_Elm_Creek",
            "CARMAN_ELM_CREEK": "Carman_Elm_Creek",
            "STURGEON": "Sturgeon_Creek",
            "STURGEON_CREEK": "Sturgeon_Creek",
            "KENASTON": "Kenaston",
        }
        cluster_names = {
            str(c).upper(): str(c) for c in attr.cluster.astype(str).unique()
        }
        # Allow province codes (MB/SK), cluster names, or exact station names.
        if region_code in set(attr.province.astype(str).str.upper()):
            ddp_train_stations = tuple(
                attr.loc[attr.province.astype(str).str.upper() == region_code, "station"]
                .map(normalize_station_id)
                .tolist()
            )
        elif region_code in cluster_alias or region_code in cluster_names:
            cluster_key = cluster_alias.get(region_code) or cluster_names[region_code]
            ddp_train_stations = tuple(
                attr.loc[attr.cluster.astype(str) == cluster_key, "station"]
                .map(normalize_station_id)
                .tolist()
            )
        elif region_code in {"ON", "ONTARIO"}:
            ddp_train_stations = ()
        else:
            ddp_train_stations = (normalize_station_id(region_code),)
        if not ddp_train_stations:
            raise SystemExit(
                f"rank {rank} region={region_code!r} has 0 stations in Stage0 "
                f"(Ontario is not in the 12-station corpus). "
                f"Use layout MB,SK or Carman,Sturgeon,Kenaston."
            )
        if _is_main(rank):
            print(
                f"  region layout={parts} → rank{rank}={region_code} "
                f"n_stations={len(ddp_train_stations)} {sorted(ddp_train_stations)}",
                flush=True,
            )
    if args.data_parallel and device.type == "cuda" and _is_main(rank):
        print(
            f"  multi-GPU DataParallel requested device_count={torch.cuda.device_count()}",
            flush=True,
        )

    all_m = []
    all_preds: dict[str, dict[int, pd.DataFrame]] = {}
    for variant in args.variants:
        if variant == "td_gru":
            depth_mode, use_state = "gru", False
        elif variant == "td_mlp":
            depth_mode, use_state = "mlp", False
        elif variant == "td_gru_rainC":
            depth_mode, use_state = "gru", True
        else:
            raise ValueError(variant)
        m, preds = train_one(
            variant=variant,
            depth_mode=depth_mode,
            use_rain_state=use_state,
            seeds=tuple(args.seeds),
            max_epochs=args.max_epochs,
            patience=args.patience,
            device=device,
            out_dir=out,
            holdout_stations=holdout,
            allow_stations=stations,
            split_preset=args.split_preset,
            target_kind=args.target_kind,
            resolution=args.resolution,
            shuffle_train=not args.no_shuffle,
            data_parallel=args.data_parallel and not ddp_on,
            ddp=ddp_on,
            ddp_rank=rank,
            ddp_world_size=world_size,
            ddp_train_stations=ddp_train_stations,
            num_workers=args.num_workers,
            target_depth=args.target_depth,
            hidden=args.hidden,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            scheduler_patience=args.scheduler_patience,
            es_metric=args.es_metric,
            window_days=args.window_days,
            batch_size=args.batch_size,
        )
        all_m.append(m)
        all_preds[variant] = preds
        if _is_main(rank):
            m.to_csv(out / f"metrics_{variant}.csv", index=False)

    if not _is_main(rank):
        if ddp_on:
            dist.barrier()
            dist.destroy_process_group()
        return

    metrics = pd.concat(all_m, ignore_index=True)
    metrics.to_csv(out / "metrics_all.csv", index=False)

    baselines = {
        "SharedGRU": _load_pred(_S0_PREDS),
        "RainC": _load_pred(_RAIN_C_PREDS),
    }
    paired_rows = []
    for variant, pmap in all_preds.items():
        other = s1._seed_avg_preds(pmap)
        for bname, base in baselines.items():
            if base is None:
                continue
            for h in (1, 3):
                for sl in ("all", "wetting", "quiet"):
                    d, lo, hi, n = s1.station_bootstrap_rmse_delta(
                        base, other, horizon=h, regime=None if sl == "all" else sl
                    )
                    paired_rows.append(
                        {
                            "model": variant,
                            "baseline": bname,
                            "horizon": h,
                            "slice": sl,
                            "delta": d,
                            "lo": lo,
                            "hi": hi,
                            "n": n,
                        }
                    )
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired.to_csv(out / "paired_delta.csv", index=False)
        print(paired.to_string(index=False), flush=True)

    write_report(out, metrics, paired)
    print("Wrote", out / "TIME_DEPTH_REPORT.md", flush=True)
    print((out / "TIME_DEPTH_REPORT.md").read_text())
    if ddp_on:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
