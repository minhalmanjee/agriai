"""Hybrid SharedGRU: temporal encoder + LGBM PAST_FEATURES at origin t.

Parity debug showed TinyMLP on engineered features ≈ LightGBM, while the
sequence GRU lagged at h=1. This model keeps the shared GRU pool and
concatenates the same ~38 origin features (train-scaled) into the head.
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
from torch.utils.data import DataLoader, Dataset, Subset

from .graph_tensor_loader import GraphSample, RismaGraphDataset, collate_graphs
from .step10_lightgbm_baselines import (
    PAST_FEATURES,
    build_origin_feat_lookup,
    build_station_features,
)
from .step10b_node_scaler import inverse_target
from .train_shared_gru import (
    SEEDS,
    evaluate as _evaluate_base,
    masked_huber,
    set_seed,
    skill_vs_persistence,
)

_PKG = Path(__file__).resolve().parent
_STEP10 = _PKG.parent / "risma_processed" / "step10_stage0"
_DEFAULT_SCALER = _STEP10 / "step10b_scaled" / "scaler.json"
_DEFAULT_PARQUET = _STEP10 / "parquet_model12_v2"
_DEFAULT_OUT = _STEP10 / "shared_gru_hybrid"


class SharedGRUHybrid(nn.Module):
    def __init__(
        self,
        f_dyn: int,
        n_static: int,
        n_station_attrs: int,
        n_origin_feats: int,
        hidden: int = 64,
        station_emb_dim: int = 4,
        n_horizons: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = f_dyn + 1 + n_static
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.station_proj = nn.Linear(n_station_attrs, station_emb_dim)
        self.origin_proj = nn.Sequential(
            nn.Linear(n_origin_feats, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + station_emb_dim + hidden, n_horizons),
        )

    def forward(
        self,
        node_x: torch.Tensor,
        node_mask: torch.Tensor,
        node_static: torch.Tensor,
        station_attrs: torch.Tensor,
        origin_feats: torch.Tensor,
    ) -> torch.Tensor:
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
        of = self.origin_proj(origin_feats)
        return self.head(torch.cat([pooled, emb, of], dim=-1))


class HybridGraphDataset(Dataset):
    """Subset of RismaGraphDataset; origin_feats come from the loader snapshot."""

    def __init__(
        self,
        base: RismaGraphDataset,
        indices: list[int],
        feat_lookup: dict[tuple[str, pd.Timestamp], np.ndarray] | None = None,
        n_feats: int | None = None,
    ) -> None:
        self.base = base
        self.indices = indices
        # Optional override lookup (legacy audits); prefer sample.origin_feats.
        self.feat_lookup = feat_lookup
        self.n_feats = n_feats if n_feats is not None else int(base.n_origin_feats)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[GraphSample, torch.Tensor]:
        s = self.base[self.indices[i]]
        if self.feat_lookup is not None:
            key = (s.station, pd.Timestamp(s.date))
            feats = self.feat_lookup.get(key)
            if feats is None:
                feats = np.zeros(self.n_feats, dtype=np.float32)
            return s, torch.from_numpy(feats.astype(np.float32))
        if s.origin_feats.numel() == self.n_feats:
            return s, s.origin_feats
        # pad / truncate if mismatched
        of = np.zeros(self.n_feats, dtype=np.float32)
        n = min(self.n_feats, int(s.origin_feats.numel()))
        if n:
            of[:n] = s.origin_feats.numpy()[:n]
        return s, torch.from_numpy(of)


def collate_hybrid(batch: list[tuple[GraphSample, torch.Tensor]]) -> dict[str, Any]:
    samples = [b[0] for b in batch]
    out = collate_graphs(samples)
    out["origin_feats"] = torch.stack([b[1] for b in batch], dim=0)
    return out


def build_feat_lookup(
    parquet_dir: Path,
    train_stations_dates: pd.DataFrame,
    *,
    include_station_code: bool = True,
    drop_ice_frozen_20: bool = False,
) -> tuple[dict[tuple[str, pd.Timestamp], np.ndarray], list[str], dict[str, tuple[float, float]]]:
    """Backward-compatible wrapper around build_origin_feat_lookup."""
    drop = {"ice_frac_20cm", "frozen_frac_20cm"} if drop_ice_frozen_20 else set()
    return build_origin_feat_lookup(
        parquet_dir,
        train_stations_dates,
        drop_feats=drop,
        include_station_code=include_station_code,
    )


@torch.no_grad()
def evaluate_hybrid(model, loader, scaler, horizons, device) -> dict[str, Any]:
    model.eval()
    ys: dict[int, list] = {h: [] for h in horizons}
    yh: dict[int, list] = {h: [] for h in horizons}
    n_near_zero = 0
    n_pred = 0
    all_pred = []
    all_y = []
    for batch in loader:
        pred = model(
            batch["node_x"].to(device),
            batch["node_mask"].to(device),
            batch["node_static"].to(device),
            batch["station_attrs"].to(device),
            batch["origin_feats"].to(device),
        ).cpu().numpy()
        y = batch["y"].numpy()
        mask = batch["y_mask"].numpy()
        for hi, h in enumerate(horizons):
            m = mask[:, hi] > 0
            if not m.any():
                continue
            y_h = y[m, hi]
            p_phys = inverse_target(pred[m, hi], h, scaler)
            ys[h].append(y_h.astype(float))
            yh[h].append(np.asarray(p_phys, dtype=float))
            n_near_zero += int(np.sum(np.abs(p_phys) < 1e-4))
            n_pred += int(m.sum())
            all_y.append(y_h)
            all_pred.append(p_phys)
    per_h = {}
    skills = []
    for h in horizons:
        if not ys[h]:
            per_h[str(h)] = {"n": 0, "rmse": float("nan"), "mae": float("nan"), "skill": float("nan")}
            continue
        y = np.concatenate(ys[h])
        p = np.concatenate(yh[h])
        mse = float(np.mean((p - y) ** 2))
        per_h[str(h)] = {
            "n": int(len(y)),
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(p - y))),
            "skill": skill_vs_persistence(y, p),
            "pred_std": float(np.std(p)),
            "target_std": float(np.std(y)),
        }
        skills.append(per_h[str(h)]["skill"])
    y_all = np.concatenate(all_y) if all_y else np.array([])
    p_all = np.concatenate(all_pred) if all_pred else np.array([])
    return {
        "per_horizon": per_h,
        "mean_horizon_skill": float(np.mean(skills)) if skills else float("nan"),
        "overall_skill": skill_vs_persistence(y_all, p_all) if len(y_all) else float("nan"),
        "frac_near_zero": float(n_near_zero / n_pred) if n_pred else float("nan"),
        "pred_target_std_ratio": float(np.std(p_all) / max(np.std(y_all), 1e-12)) if len(y_all) else float("nan"),
    }


def train_one_seed(
    train_loader,
    val_loader,
    test_loader,
    meta: dict,
    scaler: dict,
    horizons: tuple[int, ...],
    seed: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
    lr: float,
    hidden: int,
) -> dict[str, Any]:
    set_seed(seed)
    model = SharedGRUHybrid(
        f_dyn=meta["f_dyn"],
        n_static=meta["n_static"],
        n_station_attrs=meta["n_attrs"],
        n_origin_feats=meta["n_origin_feats"],
        hidden=hidden,
        n_horizons=len(horizons),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_skill = -1e9
    best_state = None
    best_epoch = 0
    bad = 0
    history = []
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(
                batch["node_x"].to(device),
                batch["node_mask"].to(device),
                batch["node_static"].to(device),
                batch["station_attrs"].to(device),
                batch["origin_feats"].to(device),
            )
            loss = masked_huber(pred, batch["y_scaled"].to(device), batch["y_mask"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        val = evaluate_hybrid(model, val_loader, scaler, horizons, device)
        skill = val["mean_horizon_skill"]
        improved = np.isfinite(skill) and skill > best_skill + 1e-4
        if improved:
            best_skill = float(skill)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        history.append(
            {
                "epoch": epoch,
                "train_huber": float(np.mean(losses)),
                "val_mean_h_skill": skill,
                "val_frac_near_zero": val["frac_near_zero"],
                "val_pred_tgt_std": val["pred_target_std_ratio"],
                **{f"val_skill_h{h}": val["per_horizon"][str(h)]["skill"] for h in horizons},
            }
        )
        print(
            f"  seed={seed} ep={epoch:03d} loss={history[-1]['train_huber']:.4f} "
            f"val_skill={skill:.4f} σ̂/σ={val['pred_target_std_ratio']:.2f} "
            f"{'*' if improved else ''} bad={bad}",
            flush=True,
        )
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_best = evaluate_hybrid(model, val_loader, scaler, horizons, device)
    test_best = evaluate_hybrid(model, test_loader, scaler, horizons, device)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "wall_s": time.time() - t0,
        "history": history,
        "val": val_best,
        "test": test_best,
        "state_dict": best_state,
    }


def run(
    out_dir: Path = _DEFAULT_OUT,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    batch_size: int = 256,
    max_epochs: int = 100,
    patience: int = 20,
    lr: float = 3e-4,
    hidden: int = 64,
    device: str | None = None,
    window_days: int | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    scaler = json.loads(_DEFAULT_SCALER.read_text())

    print(f"device={device_t}", flush=True)
    print("loading graph dataset (legacy headline: 38 feats, no anchored/DOY)...", flush=True)
    ds_kwargs: dict[str, Any] = {
        "block": None,
        "include_origin_feats": True,
        "include_station_code": True,
        "include_anchored_vwc": False,
        "include_doy": False,
    }
    if window_days is not None:
        ds_kwargs["window_days"] = int(window_days)
    full = RismaGraphDataset(**ds_kwargs)
    blocks = full._blocks
    train_idx = [i for i, b in enumerate(blocks) if b == "train"]
    val_idx = [i for i, b in enumerate(blocks) if b == "val"]
    test_idx = [i for i, b in enumerate(blocks) if b == "test"]

    feat_names = full.origin_feat_names
    feat_stats = full.origin_feat_stats
    print(
        f"  n_origin_feats={len(feat_names)} lookup={len(full._origin_lookup)} "
        f"f_dyn={full.f_dyn} n_attrs={full.n_attrs}",
        flush=True,
    )
    (out_dir / "origin_feat_names.json").write_text(json.dumps(feat_names, indent=2) + "\n")
    (out_dir / "origin_feat_stats.json").write_text(
        json.dumps({k: {"mean": v[0], "std": v[1]} for k, v in feat_stats.items()}, indent=2) + "\n"
    )

    train_ds = HybridGraphDataset(full, train_idx)
    val_ds = HybridGraphDataset(full, val_idx)
    test_ds = HybridGraphDataset(full, test_idx)
    horizons = full.horizons
    meta = {
        "f_dyn": full.f_dyn,
        "n_static": int(full.node_static.shape[-1]),
        "n_attrs": full.n_attrs,
        "n_origin_feats": len(feat_names),
        "window": full.window,
        "n_nodes": len(full.nodes),
        "attr_names": full.attr_names,
    }

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_hybrid
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_hybrid
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_hybrid
    )

    metric_rows = []
    seed_rows = []
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    for seed in seeds:
        print(f"\n=== hybrid seed {seed} ===", flush=True)
        res = train_one_seed(
            train_loader,
            val_loader,
            test_loader,
            meta,
            scaler,
            horizons,
            seed,
            device_t,
            max_epochs,
            patience,
            lr,
            hidden,
        )
        if res["state_dict"] is not None:
            torch.save(res["state_dict"], ckpt_dir / f"hybrid_seed{seed}.pt")
        pd.DataFrame(res["history"]).to_csv(out_dir / f"history_seed{seed}.csv", index=False)
        for block in ("val", "test"):
            m = res[block]
            for h in horizons:
                ph = m["per_horizon"][str(h)]
                metric_rows.append(
                    {
                        "seed": seed,
                        "block": block,
                        "horizon": h,
                        "skill": ph["skill"],
                        "rmse": ph["rmse"],
                        "mae": ph["mae"],
                        "n": ph["n"],
                        "pred_std": ph.get("pred_std"),
                        "target_std": ph.get("target_std"),
                        "frac_near_zero": m["frac_near_zero"],
                        "pred_target_std_ratio": m["pred_target_std_ratio"],
                        "mean_horizon_skill": m["mean_horizon_skill"],
                    }
                )
        seed_rows.append(
            {
                "seed": seed,
                "best_epoch": res["best_epoch"],
                "n_params": res["n_params"],
                "wall_s": res["wall_s"],
                "val_mean_horizon_skill": res["val"]["mean_horizon_skill"],
                "test_mean_horizon_skill": res["test"]["mean_horizon_skill"],
                "val_pred_tgt_std": res["val"]["pred_target_std_ratio"],
                **{f"val_skill_h{h}": res["val"]["per_horizon"][str(h)]["skill"] for h in horizons},
                **{f"test_skill_h{h}": res["test"]["per_horizon"][str(h)]["skill"] for h in horizons},
            }
        )
        print(
            f"  BEST ep={res['best_epoch']} val={res['val']['mean_horizon_skill']:.4f} "
            f"test={res['test']['mean_horizon_skill']:.4f} "
            f"σ̂/σ_val={res['val']['pred_target_std_ratio']:.2f}",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "hybrid_metrics.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out_dir / "hybrid_seeds.csv", index=False)

    agg = []
    for block in ("val", "test"):
        for h in horizons:
            sub = metrics[(metrics.block == block) & (metrics.horizon == h)]
            sk = sub["skill"].to_numpy(float)
            agg.append(
                {
                    "block": block,
                    "horizon": h,
                    "skill_median": float(np.median(sk)),
                    "skill_mean": float(np.mean(sk)),
                    "skill_std": float(np.std(sk)),
                    "skill_min": float(np.min(sk)),
                    "skill_max": float(np.max(sk)),
                    "rmse_median": float(sub["rmse"].median()),
                    "pred_tgt_std_median": float(sub["pred_std"].median() / max(sub["target_std"].median(), 1e-12)),
                }
            )
    agg_df = pd.DataFrame(agg)
    agg_df.to_csv(out_dir / "hybrid_summary.csv", index=False)

    lines = [
        "# Shared GRU + engineered origin features (hybrid)",
        "",
        f"- Sequence: N={meta['window']}, nodes={meta['n_nodes']}, f_dyn={meta['f_dyn']} (incl. ΔVWC + precip memory)",
        f"- Headline legacy: f_dyn={meta['f_dyn']}, attrs={meta['n_attrs']}, "
        f"origin feats={len(feat_names)} PAST_FEATURES (no anchored/DOY)",
        f"- Origin feats projected then concat to head",
        f"- AdamW lr={lr}, stop on val mean-horizon skill, seeds={list(seeds)}",
        f"- n_params≈{seed_rows[0]['n_params'] if seed_rows else '?'}",
        "",
        "| block | h | skill median | mean±std | min–max | σ̂/σ_y |",
        "|-------|---|--------------|----------|---------|-------|",
    ]
    for _, r in agg_df.iterrows():
        lines.append(
            f"| {r['block']} | {int(r['horizon'])} | {r['skill_median']:.3f} | "
            f"{r['skill_mean']:.3f}±{r['skill_std']:.3f} | "
            f"{r['skill_min']:.3f}–{r['skill_max']:.3f} | {r['pred_tgt_std_median']:.2f} |"
        )
    lines += [
        "",
        "LGBM past-only val ≈ 0.28 / 0.17 / 0.12. Target: within ~0.02.",
        "",
    ]
    report = "\n".join(lines) + "\n"
    (out_dir / "HYBRID_GRU.md").write_text(report)
    print("\n" + report)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--window", type=int, default=None, help="input window days (default: schema)")
    args = ap.parse_args()
    run(
        out_dir=args.out_dir,
        seeds=tuple(args.seeds),
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        hidden=args.hidden,
        device=args.device,
        window_days=args.window,
    )


if __name__ == "__main__":
    main()
