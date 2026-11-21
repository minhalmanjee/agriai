#!/usr/bin/env python3
"""S1 ablation: SharedGRU + past-only wetting-response mixture.

Variants
  S0  — plain SharedGRU (baseline)
  S1a — quiet head + jump head + gate from past state features (no loss weights)
  S1b — S1a + upweight origins with past wet-process cues
  S1c — S1b + quiet-preservation aux loss (pull ŷ→0 on past-quiet cues)

Gate / state inputs are past-only at origin t (no future rain, no future Δ labels).
Eval regimes (wetting/drying/quiet) use observed y only for reporting.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .graph_tensor_loader import RismaGraphDataset, collate_graphs
from .intraday_precip_features import INTRADAY_FEATURE_NAMES, build_intraday_precip_panel
from .step10_lightgbm_baselines import build_station_features
from .step10b_node_scaler import inverse_target
from .train_shared_gru import (
    SharedGRU,
    masked_huber,
    set_seed,
    skill_vs_persistence,
)
from .train_shared_gru_event_gate import masked_huber_weighted
from .train_shared_gru_hybrid import HybridGraphDataset

_PKG = Path(__file__).resolve().parent
_STEP10 = _PKG.parent / "risma_processed" / "step10_stage0"
_OUT = _STEP10 / "e3_followup" / "s1_wetting"
_OUT_INTRADAY = _STEP10 / "e3_followup" / "s1_wetting_intraday"
_ORIGINS = _STEP10 / "origins_long.csv"
_PARQUET = _STEP10 / "parquet_model12_v2"
_SCALER = _STEP10 / "step10b_scaled" / "scaler.json"
_E1_META = _STEP10 / "e3_followup" / "phase1_e1" / "run_meta.json"
_RAW_DIR = _PKG.parent / "risma_data_full"

RAIN_MM = 1.0
DAYS_SINCE_CAP = 30.0
STATE_FEATURE_NAMES_BASE = [
    "precip_1d",
    "precip_3d",
    "precip_7d",
    "precip_intensity_1d",  # daily total unless --intraday (then max 1h)
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
# mutable active list — set in train_variant / main when --intraday
STATE_FEATURE_NAMES = list(STATE_FEATURE_NAMES_BASE)


def build_state_panel(parquet_dir: Path, *, use_intraday: bool = False) -> pd.DataFrame:
    panel = build_station_features(parquet_dir)
    # optional cols from raw parquet merge
    extra_cols = ["soilT_thaw", "frozen_frac_20cm", "vwc_5cm_rep_mean", "vwc_50cm_rep_mean", "vwc_20cm_rep_mean"]
    parts = []
    for path in sorted(Path(parquet_dir).glob("RISMA_*.parquet")):
        raw = pd.read_parquet(path)
        use = ["station", "date"] + [c for c in extra_cols if c in raw.columns]
        parts.append(raw[use])
    if parts:
        extra = pd.concat(parts, ignore_index=True)
        extra["date"] = pd.to_datetime(extra["date"])
        panel["date"] = pd.to_datetime(panel["date"])
        panel = panel.merge(extra, on=["station", "date"], how="left")

    if use_intraday:
        intra = build_intraday_precip_panel(_RAW_DIR)
        panel["date"] = pd.to_datetime(panel["date"])
        panel = panel.merge(intra, on=["station", "date"], how="left")

    out_parts = []
    for st, g in panel.groupby("station"):
        g = g.sort_values("date").copy()
        p = g["precip"].astype(float).to_numpy()
        rainy = (p >= RAIN_MM).astype(float)
        g["precip_1d"] = g.get("precip_1d", g["precip"]).astype(float)
        g["precip_3d"] = g["precip_3d"].astype(float)
        g["precip_7d"] = g["precip_7d"].astype(float)
        if use_intraday:
            for c in INTRADAY_FEATURE_NAMES:
                if c not in g.columns:
                    g[c] = 0.0
                else:
                    g[c] = g[c].astype(float).fillna(0.0)
        # Keep daily total separate from peak intensity (do not overwrite with max_1h)
        g["precip_intensity_1d"] = g["precip_1d"]
        g["n_rain_days_3d"] = pd.Series(rainy).rolling(3, min_periods=1).sum().to_numpy()
        days = np.zeros(len(p), dtype=float)
        last = -10_000
        for i, pi in enumerate(p):
            if pi >= RAIN_MM:
                last = i
                days[i] = 0.0
            else:
                days[i] = float(min(i - last, DAYS_SINCE_CAP)) if last >= 0 else float(DAYS_SINCE_CAP)
        g["days_since_rain"] = days

        v5 = g["vwc_5cm_rep_mean"].astype(float) if "vwc_5cm_rep_mean" in g else g.get("vwc_5cm", pd.Series(np.nan, index=g.index))
        v20 = g["vwc_20cm_rep_mean"].astype(float) if "vwc_20cm_rep_mean" in g else g["vwc_20cm"].astype(float)
        v50 = g["vwc_50cm_rep_mean"].astype(float) if "vwc_50cm_rep_mean" in g else g.get("vwc_50cm", pd.Series(np.nan, index=g.index))
        g["vwc_20cm"] = v20
        g["dvwc_5cm_1d"] = v5.diff(1)
        g["dvwc_20cm_1d"] = v20.diff(1)
        g["dvwc_50cm_1d"] = v50.diff(1)
        g["dvwc_5cm_3d"] = v5.diff(3)
        g["dvwc_20cm_3d"] = v20.diff(3)
        # past-only wetting-propagation cues (no future VWC)
        g["relu_dvwc_5cm_1d"] = g["dvwc_5cm_1d"].clip(lower=0.0)
        g["relu_dvwc_5cm_3d"] = g["dvwc_5cm_3d"].clip(lower=0.0)
        g["relu_dvwc_20cm_1d"] = g["dvwc_20cm_1d"].clip(lower=0.0)
        rise = (g["dvwc_5cm_1d"].fillna(0.0) > 0.005).to_numpy()
        days_sh = np.full(len(g), float(DAYS_SINCE_CAP))
        last_rise = -10_000
        for i, r in enumerate(rise):
            if r:
                last_rise = i
                days_sh[i] = 0.0
            elif last_rise >= 0:
                days_sh[i] = float(min(i - last_rise, DAYS_SINCE_CAP))
        g["days_since_shallow_rise"] = days_sh
        g["precip3d_x_relu5"] = g["precip_3d"].astype(float) * g["relu_dvwc_5cm_1d"].astype(float)
        g["grad_5_20"] = v5 - v20
        g["grad_20_50"] = v20 - v50
        if "soilT_thaw" not in g:
            g["soilT_thaw"] = 1.0
        if "frozen_frac_20cm" not in g:
            g["frozen_frac_20cm"] = 0.0
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True)


# Extra past-only 5→20 propagation features (ablation on top of Rain C)
PROPAGATION_FEATURE_NAMES = [
    "relu_dvwc_5cm_1d",
    "relu_dvwc_5cm_3d",
    "relu_dvwc_20cm_1d",
    "dvwc_5cm_3d",
    "days_since_shallow_rise",
    "precip3d_x_relu5",
]


def build_state_lookup(
    parquet_dir: Path,
    train_keys: pd.DataFrame,
    *,
    use_intraday: bool = False,
) -> tuple[dict[tuple[str, pd.Timestamp], np.ndarray], list[str], dict[str, tuple[float, float]]]:
    panel = build_state_panel(parquet_dir, use_intraday=use_intraday)
    names = list(STATE_FEATURE_NAMES)
    # train-only z-score
    tr = train_keys.copy()
    tr["date"] = pd.to_datetime(tr["date"])
    joined = panel.merge(tr[["station", "date"]].drop_duplicates(), on=["station", "date"], how="inner")
    stats: dict[str, tuple[float, float]] = {}
    for c in names:
        x = joined[c].astype(float).to_numpy()
        x = x[np.isfinite(x)]
        mu = float(np.mean(x)) if len(x) else 0.0
        sd = float(np.std(x)) if len(x) else 1.0
        if sd < 1e-8:
            sd = 1.0
        stats[c] = (mu, sd)

    lookup: dict[tuple[str, pd.Timestamp], np.ndarray] = {}
    for _, row in panel.iterrows():
        key = (str(row["station"]), pd.Timestamp(row["date"]))
        vec = np.zeros(len(names), dtype=np.float32)
        for i, c in enumerate(names):
            v = float(row[c]) if np.isfinite(row[c]) else stats[c][0]
            mu, sd = stats[c]
            vec[i] = (v - mu) / sd
        lookup[key] = vec
    return lookup, names, stats


def past_wet_process_weight(state_raw: dict[str, float], wet_weight: float) -> float:
    """Past-only cue: recent rain or ongoing event — NOT future y."""
    if state_raw.get("precip_3d", 0.0) >= 1.0:
        return wet_weight
    if state_raw.get("precip_1d", 0.0) >= 1.0:
        return wet_weight
    if state_raw.get("days_since_rain", 99.0) <= 1.0:
        return wet_weight
    if state_raw.get("dvwc_5cm_1d", 0.0) >= 0.005:
        return wet_weight
    # propagation / intraday cues when present in state vector
    if state_raw.get("relu_dvwc_5cm_1d", 0.0) >= 0.005:
        return wet_weight
    if state_raw.get("days_since_shallow_rise", 99.0) <= 1.0:
        return wet_weight
    if state_raw.get("precip_max_1h", 0.0) >= 2.0:
        return wet_weight
    if state_raw.get("precip_max_3h", 0.0) >= 5.0:
        return wet_weight
    return 1.0


class SharedGRUS1(nn.Module):
    """Shared GRU encoder + past-only gate + quiet/jump heads."""

    def __init__(
        self,
        f_dyn: int,
        n_static: int,
        n_station_attrs: int,
        n_state: int,
        hidden: int = 64,
        station_emb_dim: int = 4,
        n_horizons: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = f_dyn + 1 + n_static
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.station_proj = nn.Linear(n_station_attrs, station_emb_dim)
        self.state_proj = nn.Sequential(
            nn.Linear(n_state, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        gate_in = hidden + station_emb_dim + hidden
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        head_in = gate_in
        self.quiet_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, n_horizons))
        self.jump_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, n_horizons))
        self.hidden = hidden

    def encode(
        self,
        node_x: torch.Tensor,
        node_mask: torch.Tensor,
        node_static: torch.Tensor,
        station_attrs: torch.Tensor,
        state_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, t, _ = node_x.shape
        mask = node_mask.unsqueeze(-1)
        static = node_static.unsqueeze(2).expand(-1, -1, t, -1)
        x = torch.cat([node_x, mask, static], dim=-1) * mask
        x = x.reshape(b * n, t, -1)
        _, h = self.gru(x)
        h = h[-1].reshape(b, n, -1)
        w = node_mask.mean(dim=-1).clamp_min(0.0)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)
        emb = self.station_proj(station_attrs)
        st = self.state_proj(state_feats)
        ctx = torch.cat([pooled, emb, st], dim=-1)
        g = torch.sigmoid(self.gate(ctx))  # (B,1)
        return ctx, g

    def forward(
        self,
        node_x: torch.Tensor,
        node_mask: torch.Tensor,
        node_static: torch.Tensor,
        station_attrs: torch.Tensor,
        state_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx, g = self.encode(node_x, node_mask, node_static, station_attrs, state_feats)
        y_q = self.quiet_head(ctx)
        # softplus → non-negative wet-up in *scaled* space; still mixed with quiet
        y_j = torch.nn.functional.softplus(self.jump_head(ctx))
        yhat = (1.0 - g) * y_q + g * y_j
        return yhat, g.squeeze(-1)


@torch.no_grad()
def evaluate_s1(
    model: nn.Module,
    loader: DataLoader,
    scaler: dict,
    horizons: tuple[int, ...],
    device: torch.device,
    *,
    variant: str,
) -> dict[str, Any]:
    model.eval()
    rows = []
    for batch in loader:
        if variant == "s0":
            pred = model(
                batch["node_x"].to(device),
                batch["node_mask"].to(device),
                batch["node_static"].to(device),
                batch["station_attrs"].to(device),
            )
            g = None
        else:
            pred, g = model(
                batch["node_x"].to(device),
                batch["node_mask"].to(device),
                batch["node_static"].to(device),
                batch["station_attrs"].to(device),
                batch["origin_feats"].to(device),
            )
        y = batch["y"].cpu().numpy()
        mask = batch["y_mask"].cpu().numpy()
        pred_np = pred.cpu().numpy()
        stations = batch["station"]
        dates = batch["date"]
        for i in range(len(stations)):
            for hi, h in enumerate(horizons):
                if mask[i, hi] <= 0:
                    continue
                p_phys = float(inverse_target(pred_np[i, hi], h, scaler))
                rows.append(
                    {
                        "station": stations[i],
                        "date": str(pd.Timestamp(dates[i]).date()),
                        "horizon": int(h),
                        "y": float(y[i, hi]),
                        "yhat": p_phys,
                        "gate": float(g[i].cpu()) if g is not None else float("nan"),
                    }
                )
    pred_df = pd.DataFrame(rows)
    metrics = score_pred_df(pred_df, horizons)
    metrics["pred_df"] = pred_df
    return metrics


def score_pred_df(pred_df: pd.DataFrame, horizons: tuple[int, ...]) -> dict[str, Any]:
    pred_df = pred_df.copy()
    pred_df["regime"] = "quiet"
    pred_df.loc[pred_df["y"] > 0.01, "regime"] = "wetting"
    pred_df.loc[pred_df["y"] < -0.01, "regime"] = "drying"
    out: dict[str, Any] = {"per_horizon": {}, "regimes": {}}
    skills = []
    for h in horizons:
        sub = pred_df[pred_df.horizon == h]
        y = sub.y.to_numpy(float)
        p = sub.yhat.to_numpy(float)
        rmse = float(np.sqrt(np.mean((p - y) ** 2))) if len(sub) else float("nan")
        sk = skill_vs_persistence(y, p) if len(sub) else float("nan")
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


def quiet_preservation_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    """Past-quiet cues (weight≈1): Huber(pred, 0). Discourages over-reaction on quiet."""
    quiet = (sample_weight <= 1.0 + 1e-5).float()
    zeros = torch.zeros_like(pred)
    return masked_huber_weighted(pred, zeros, mask, quiet, delta=delta)


def station_bootstrap_rmse_delta(
    base: pd.DataFrame,
    other: pd.DataFrame,
    *,
    horizon: int,
    regime: str | None,
    n_boot: int = 400,
    seed: int = 0,
) -> tuple[float, float, float, int]:
    """Paired ΔRMSE = RMSE_other − RMSE_base (<0 ⇒ other better). Station blocks."""
    a = base[base.horizon == horizon].copy()
    b = other[other.horizon == horizon].copy()
    if regime and regime != "all":
        a = a[a.regime == regime]
        b = b[b.regime == regime]
    j = a.merge(
        b[["station", "date", "horizon", "yhat"]].rename(columns={"yhat": "yhat_other"}),
        on=["station", "date", "horizon"],
        how="inner",
    )
    if len(j) < 30:
        return float("nan"), float("nan"), float("nan"), len(j)
    y = j.y.to_numpy(float)
    p0 = j.yhat.to_numpy(float)
    p1 = j.yhat_other.to_numpy(float)
    se0 = (y - p0) ** 2
    se1 = (y - p1) ** 2
    point = float(np.sqrt(se1.mean()) - np.sqrt(se0.mean()))
    blk = j.station.astype(str).to_numpy()
    stations = np.unique(blk)
    se0_b = [se0[blk == s] for s in stations]
    se1_b = [se1[blk == s] for s in stations]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        draw = rng.integers(0, len(stations), size=len(stations))
        s0 = np.concatenate([se0_b[i] for i in draw])
        s1 = np.concatenate([se1_b[i] for i in draw])
        if s0.size == 0:
            continue
        boots.append(float(np.sqrt(s1.mean()) - np.sqrt(s0.mean())))
    if not boots:
        return point, float("nan"), float("nan"), len(j)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi), len(j)


def train_variant(
    variant: str,
    *,
    seeds: tuple[int, ...],
    max_epochs: int,
    patience: int,
    wet_weight: float,
    quiet_lambda: float,
    device: torch.device,
    out_dir: Path,
    use_intraday: bool = False,
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    meta = json.loads(_E1_META.read_text())
    scaler = json.loads(Path(meta["scaler_path"]).read_text())
    horizons = tuple(int(h) for h in meta["horizons"])
    full = RismaGraphDataset(
        origins_path=_ORIGINS,
        parquet_dir=_PARQUET,
        scaler_path=meta["scaler_path"],
        block=None,
        window_days=int(meta["window_days"]),
    )
    train_idx = [i for i in range(len(full)) if full._blocks[i] == "train"]
    val_idx = [i for i in range(len(full)) if full._blocks[i] == "val"]
    test_idx = [i for i in range(len(full)) if full._blocks[i] == "test"]

    train_keys = pd.DataFrame(
        {"station": [full._stations[i] for i in train_idx], "date": [full._dates[i] for i in train_idx]}
    )
    lookup, names, stats = build_state_lookup(_PARQUET, train_keys, use_intraday=use_intraday)
    (out_dir / "state_feature_names.json").write_text(json.dumps(names, indent=2))
    (out_dir / "state_feature_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  state_feats={names} n={len(names)} intraday={use_intraday}", flush=True)

    # sample weights from raw (unscaled) past cues — rebuild quick raw map
    panel = build_state_panel(_PARQUET, use_intraday=use_intraday)
    panel["date"] = pd.to_datetime(panel["date"])
    raw_map: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
    for _, r in panel.iterrows():
        key = (str(r["station"]), pd.Timestamp(r["date"]))
        raw_map[key] = {
            c: float(r[c]) if c in panel.columns and np.isfinite(r[c]) else 0.0
            for c in STATE_FEATURE_NAMES
        }

    class WeightedHybrid(HybridGraphDataset):
        def __getitem__(self, i: int):
            s, feats = super().__getitem__(i)
            key = (s.station, pd.Timestamp(s.date))
            w = 1.0
            if variant in ("s1b", "s1c"):
                raw = raw_map.get(key)
                if raw is not None:
                    w = past_wet_process_weight(raw, wet_weight)
            return s, feats, np.float32(w)

    def collate_w(batch):
        samples = [b[0] for b in batch]
        out = collate_graphs(samples)
        out["origin_feats"] = torch.stack([b[1] for b in batch], dim=0)
        out["sample_weight"] = torch.tensor([float(b[2]) for b in batch], dtype=torch.float32)
        return out

    sample = full[train_idx[0]]
    sample_meta = {
        "f_dyn": int(sample.node_x.shape[-1]),
        "n_static": int(sample.node_static.shape[-1]),
        "n_attrs": int(sample.station_attrs.shape[-1]),
        "n_state": len(names),
    }

    metric_rows = []
    test_preds: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        set_seed(seed)
        if variant == "s0":
            tr = DataLoader(Subset(full, train_idx), batch_size=128, shuffle=True, collate_fn=collate_graphs)
            va = DataLoader(Subset(full, val_idx), batch_size=256, shuffle=False, collate_fn=collate_graphs)
            te = DataLoader(Subset(full, test_idx), batch_size=256, shuffle=False, collate_fn=collate_graphs)
            model: nn.Module = SharedGRU(
                f_dyn=sample_meta["f_dyn"],
                n_static=sample_meta["n_static"],
                n_station_attrs=sample_meta["n_attrs"],
                hidden=64,
                n_horizons=len(horizons),
            ).to(device)
        else:
            tr_ds = WeightedHybrid(full, train_idx, feat_lookup=lookup, n_feats=len(names))
            va_ds = WeightedHybrid(full, val_idx, feat_lookup=lookup, n_feats=len(names))
            te_ds = WeightedHybrid(full, test_idx, feat_lookup=lookup, n_feats=len(names))
            tr = DataLoader(tr_ds, batch_size=128, shuffle=True, collate_fn=collate_w)
            va = DataLoader(va_ds, batch_size=256, shuffle=False, collate_fn=collate_w)
            te = DataLoader(te_ds, batch_size=256, shuffle=False, collate_fn=collate_w)
            model = SharedGRUS1(
                f_dyn=sample_meta["f_dyn"],
                n_static=sample_meta["n_static"],
                n_station_attrs=sample_meta["n_attrs"],
                n_state=len(names),
                hidden=64,
                n_horizons=len(horizons),
            ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        best_skill = -1e9
        best_state = None
        best_epoch = 0
        bad = 0
        t0 = time.time()
        print(f"=== {variant} seed={seed} ===", flush=True)
        for epoch in range(1, max_epochs + 1):
            model.train()
            losses = []
            for batch in tr:
                opt.zero_grad(set_to_none=True)
                if variant == "s0":
                    pred = model(
                        batch["node_x"].to(device),
                        batch["node_mask"].to(device),
                        batch["node_static"].to(device),
                        batch["station_attrs"].to(device),
                    )
                    loss = masked_huber(pred, batch["y_scaled"].to(device), batch["y_mask"].to(device))
                else:
                    pred, _g = model(
                        batch["node_x"].to(device),
                        batch["node_mask"].to(device),
                        batch["node_static"].to(device),
                        batch["station_attrs"].to(device),
                        batch["origin_feats"].to(device),
                    )
                    if variant in ("s1b", "s1c"):
                        loss = masked_huber_weighted(
                            pred,
                            batch["y_scaled"].to(device),
                            batch["y_mask"].to(device),
                            batch["sample_weight"].to(device),
                        )
                        if variant == "s1c" and quiet_lambda > 0:
                            loss = loss + quiet_lambda * quiet_preservation_loss(
                                pred,
                                batch["y_mask"].to(device),
                                batch["sample_weight"].to(device),
                            )
                    else:
                        loss = masked_huber(pred, batch["y_scaled"].to(device), batch["y_mask"].to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.item()))

            val_m = evaluate_s1(model, va, scaler, horizons, device, variant=variant)
            skill = val_m["mean_horizon_skill"]
            improved = np.isfinite(skill) and skill > best_skill + 1e-4
            if improved:
                best_skill = float(skill)
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            print(
                f"  {variant} seed={seed} ep={epoch:03d} loss={np.mean(losses):.4f} "
                f"val_skill={skill:.4f} h1={val_m['per_horizon']['1']['rmse']:.5f} "
                f"h3={val_m['per_horizon']['3']['rmse']:.5f} {'*' if improved else ''} bad={bad}",
                flush=True,
            )
            if bad >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        ckpt = out_dir / "checkpoints" / f"{variant}_seed{seed}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt)

        for block, loader in (("val", va), ("test", te)):
            m = evaluate_s1(model, loader, scaler, horizons, device, variant=variant)
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
                h = int(hs[1:])
                metric_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "block": block,
                        "horizon": h,
                        "slice": reg,
                        "rmse": regm["rmse"],
                        "skill": float("nan"),
                        "n": regm["n"],
                        "best_epoch": best_epoch,
                        "wall_s": time.time() - t0,
                    }
                )
            print(
                f"  {variant} seed={seed} {block}: h1={m['per_horizon']['1']['rmse']:.5f} "
                f"h3={m['per_horizon']['3']['rmse']:.5f} "
                f"wet_h1={m['regimes'].get('h1_wetting', {}).get('rmse', float('nan')):.5f} "
                f"quiet_h1={m['regimes'].get('h1_quiet', {}).get('rmse', float('nan')):.5f}",
                flush=True,
            )
    return pd.DataFrame(metric_rows), test_preds


def write_report(
    out_dir: Path,
    metrics: pd.DataFrame,
    paired: pd.DataFrame | None = None,
    *,
    use_intraday: bool = False,
    vs_s1b: pd.DataFrame | None = None,
) -> None:
    title = "# S1 wetting-response ablation (intraday precip)" if use_intraday else "# S1 wetting-response ablation"
    lines = [
        title,
        "",
        "S0 = SharedGRU · S1a = gate only · S1b = + wet-process weights · "
        "S1c = S1b + quiet-preservation aux loss (ŷ→0 on past-quiet cues).",
        "Gate inputs are past-only. Eval wetting/quiet/drying use observed y.",
    ]
    if use_intraday:
        lines += [
            "",
            "Intraday extras (calendar day t, past-only 15-min tips): "
            "`precip_max_1h`, `precip_max_3h`, `precip_max_15min`, `n_hours_rain`, `precip_pm_share`. "
            "`precip_1d` / `precip_intensity_1d` stay daily totals (not overwritten by max 1h).",
        ]
    lines += [
        "",
        "## Test RMSE (mean over seeds)",
        "",
        "| variant | h | all | wetting | quiet | drying |",
        "|--|--:|--:|--:|--:|--:|",
    ]
    te = metrics[metrics.block == "test"]
    order = ("s0", "s1a", "s1b", "s1b_id", "s1c")
    variants = [v for v in order if v in set(te.variant)] + [
        v for v in sorted(set(te.variant)) if v not in order
    ]
    for variant in variants:
        for h in (1, 3, 7, 14):
            cells = []
            for sl in ("all", "wetting", "quiet", "drying"):
                sub = te[(te.variant == variant) & (te.horizon == h) & (te.slice == sl)]
                if sub.empty or not np.isfinite(sub.rmse).any():
                    cells.append("—")
                else:
                    cells.append(f"{sub.rmse.mean():.5f}")
            lines.append(f"| {variant} | {h} | " + " | ".join(cells) + " |")
    if paired is not None and not paired.empty:
        lines += [
            "",
            "## Paired ΔRMSE vs S0 (test, seed-avg preds; station block-bootstrap 95% CI)",
            "",
            "ΔRMSE = RMSE_model − RMSE_S0. Negative ⇒ better than S0.",
            "",
            "| model | h | slice | ΔRMSE [95% CI] | n |",
            "|--|--:|--|--|--:|",
        ]
        for _, r in paired.iterrows():
            lines.append(
                f"| {r.model} | {int(r.horizon)} | {r.slice} | "
                f"{r.delta:.6f} [{r.lo:.6f}, {r.hi:.6f}] | {int(r.n)} |"
            )
    if vs_s1b is not None and not vs_s1b.empty:
        lines += [
            "",
            "## Paired ΔRMSE vs baseline S1b (test; station block-bootstrap 95% CI)",
            "",
            "ΔRMSE = RMSE_intraday − RMSE_S1b. Negative ⇒ intraday better.",
            "",
            "| h | slice | ΔRMSE [95% CI] | n |",
            "|--:|--|--|--:|",
        ]
        for _, r in vs_s1b.iterrows():
            lines.append(
                f"| {int(r.horizon)} | {r.slice} | "
                f"{r.delta:.6f} [{r.lo:.6f}, {r.hi:.6f}] | {int(r.n)} |"
            )
    lines += [
        "",
        "## Milestone check (test h1/h3)",
        "",
        "Primary: overall h1/h3 ↓ vs S0/S1b. Critical: wetting ↓. Safeguard: quiet not worse.",
        "",
    ]
    (out_dir / "S1_REPORT.md").write_text("\n".join(lines) + "\n")


def _seed_avg_preds(pred_map: dict[int, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for seed, df in pred_map.items():
        d = df.copy()
        d["seed"] = seed
        frames.append(d)
    cat = pd.concat(frames, ignore_index=True)
    g = cat.groupby(["station", "date", "horizon"], as_index=False).agg(
        y=("y", "first"), yhat=("yhat", "mean")
    )
    g["regime"] = "quiet"
    g.loc[g.y > 0.01, "regime"] = "wetting"
    g.loc[g.y < -0.01, "regime"] = "drying"
    return g


def _load_baseline_s1b_preds(baseline_dir: Path, seeds: tuple[int, ...]) -> pd.DataFrame | None:
    frames = []
    for seed in seeds:
        path = baseline_dir / "preds" / f"s1b_seed{seed}_test.csv"
        if not path.exists():
            return None
        frames.append(pd.read_csv(path))
    return _seed_avg_preds({int(s): f for s, f in zip(seeds, frames)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--variants", nargs="+", default=["s0", "s1a", "s1b"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--wet-weight", type=float, default=3.0)
    ap.add_argument("--quiet-lambda", type=float, default=0.5, help="S1c aux loss weight")
    ap.add_argument(
        "--intraday",
        action="store_true",
        help="Add past-only 15-min burst features to S1 state/gate; rename s1b→s1b_id",
    )
    ap.add_argument(
        "--baseline-s1b-dir",
        type=Path,
        default=_OUT,
        help="Existing S1b preds dir for paired Δ vs intraday",
    )
    args = ap.parse_args()

    global STATE_FEATURE_NAMES
    use_intraday = bool(args.intraday)
    if use_intraday:
        STATE_FEATURE_NAMES = list(STATE_FEATURE_NAMES_BASE) + list(INTRADAY_FEATURE_NAMES)
        out = args.out_dir or _OUT_INTRADAY
        # keep CLI name s1b but write as s1b_id so reports don't overwrite baseline
        variants = [("s1b_id" if v == "s1b" else v) for v in args.variants]
        if "s1b_id" not in variants and "s1b" in args.variants:
            variants = [v for v in variants]
    else:
        STATE_FEATURE_NAMES = list(STATE_FEATURE_NAMES_BASE)
        out = args.out_dir or _OUT
        variants = list(args.variants)

    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_m = []
    all_preds: dict[str, dict[int, pd.DataFrame]] = {}
    for variant in variants:
        train_name = "s1b" if variant == "s1b_id" else variant
        m, preds = train_variant(
            train_name,
            seeds=tuple(args.seeds),
            max_epochs=args.max_epochs,
            patience=args.patience,
            wet_weight=args.wet_weight,
            quiet_lambda=args.quiet_lambda,
            device=device,
            out_dir=out,
            use_intraday=use_intraday,
        )
        if variant == "s1b_id":
            m = m.copy()
            m["variant"] = "s1b_id"
            for seed, df in list(preds.items()):
                df = df.copy()
                df["variant"] = "s1b_id"
                preds[seed] = df
                # also rewrite pred files under s1b_id name
                for block in ("val", "test"):
                    src = out / "preds" / f"s1b_seed{seed}_{block}.csv"
                    dst = out / "preds" / f"s1b_id_seed{seed}_{block}.csv"
                    if src.exists():
                        pdf = pd.read_csv(src)
                        pdf["variant"] = "s1b_id"
                        pdf.to_csv(dst, index=False)
        all_m.append(m)
        all_preds[variant] = preds
        m.to_csv(out / f"metrics_{variant}.csv", index=False)
    metrics = pd.concat(all_m, ignore_index=True)
    # merge with any prior metrics for other variants already on disk
    prior = out / "metrics_all.csv"
    if prior.exists():
        old = pd.read_csv(prior)
        keep = old[~old.variant.isin(metrics.variant.unique())]
        metrics = pd.concat([keep, metrics], ignore_index=True)
    metrics.to_csv(out / "metrics_all.csv", index=False)

    paired_rows = []
    if "s0" in all_preds and len(all_preds) > 1:
        s0 = _seed_avg_preds(all_preds["s0"])
        for variant, pmap in all_preds.items():
            if variant == "s0":
                continue
            other = _seed_avg_preds(pmap)
            for h in (1, 3):
                for sl in ("all", "wetting", "quiet"):
                    d, lo, hi, n = station_bootstrap_rmse_delta(
                        s0, other, horizon=h, regime=None if sl == "all" else sl
                    )
                    paired_rows.append(
                        {"model": variant, "horizon": h, "slice": sl, "delta": d, "lo": lo, "hi": hi, "n": n}
                    )
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired.to_csv(out / "paired_delta_vs_s0.csv", index=False)

    vs_s1b = None
    if use_intraday and "s1b_id" in all_preds:
        base = _load_baseline_s1b_preds(args.baseline_s1b_dir, tuple(args.seeds))
        if base is not None:
            other = _seed_avg_preds(all_preds["s1b_id"])
            rows = []
            for h in (1, 3, 7, 14):
                for sl in ("all", "wetting", "quiet", "drying"):
                    d, lo, hi, n = station_bootstrap_rmse_delta(
                        base, other, horizon=h, regime=None if sl == "all" else sl
                    )
                    rows.append({"horizon": h, "slice": sl, "delta": d, "lo": lo, "hi": hi, "n": n})
            vs_s1b = pd.DataFrame(rows)
            vs_s1b.to_csv(out / "paired_delta_vs_s1b.csv", index=False)
            print("Paired vs S1b:", flush=True)
            print(vs_s1b.to_string(index=False), flush=True)

    write_report(
        out,
        metrics,
        paired if not paired.empty else None,
        use_intraday=use_intraday,
        vs_s1b=vs_s1b,
    )
    print("Wrote", out / "S1_REPORT.md", flush=True)


if __name__ == "__main__":
    main()
