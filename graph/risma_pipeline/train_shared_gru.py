"""Step 5: small shared GRU on the schema/scaler graph loader.

Model: one GRU shared across sensor nodes → mask-weighted pool → ≤4-d
station projection → 3-horizon head.

Training: masked Huber on scaled Δ; AdamW 3e-4 / wd 1e-4; grad clip 1.0;
early-stop on validation skill in physical units (vs persistence ŷ=0).
Seeds: configurable; report median + spread. Flags near-zero collapse.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .graph_tensor_loader import RismaGraphDataset, collate_graphs
from .step10b_node_scaler import inverse_target

_PKG = Path(__file__).resolve().parent
_STEP10 = _PKG.parent / "risma_processed" / "step10_stage0"
_DEFAULT_SCALER = _STEP10 / "step10b_scaled" / "scaler.json"
_DEFAULT_OUT = _STEP10 / "shared_gru_delta"

SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
NEAR_ZERO_PHYS = 1e-4


class SharedGRU(nn.Module):
    """Shared temporal encoder over nodes; no message passing yet (Step 8)."""

    def __init__(
        self,
        f_dyn: int = 6,
        n_static: int = 5,
        n_station_attrs: int = 5,
        hidden: int = 64,
        station_emb_dim: int = 4,
        n_horizons: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if station_emb_dim > 4:
            raise ValueError("Step 5 caps station embedding at 4 dims")
        in_dim = f_dyn + 1 + n_static  # dyn + mask channel + static
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.station_proj = nn.Linear(n_station_attrs, station_emb_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + station_emb_dim, n_horizons),
        )
        self.hidden = hidden
        self.station_emb_dim = station_emb_dim

    def forward(
        self,
        node_x: torch.Tensor,  # (B, N, T, F)
        node_mask: torch.Tensor,  # (B, N, T)
        node_static: torch.Tensor,  # (B, N, S)
        station_attrs: torch.Tensor,  # (B, A)
    ) -> torch.Tensor:
        b, n, t, f = node_x.shape
        mask = node_mask.unsqueeze(-1)  # (B,N,T,1)
        static = node_static.unsqueeze(2).expand(-1, -1, t, -1)
        x = torch.cat([node_x, mask, static], dim=-1)
        x = x * mask
        x = x.reshape(b * n, t, -1)
        _, h = self.gru(x)
        h = h[-1].reshape(b, n, -1)

        w = node_mask.mean(dim=-1).clamp_min(0.0)  # (B,N)
        w_sum = w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = (h * (w / w_sum).unsqueeze(-1)).sum(dim=1)

        emb = self.station_proj(station_attrs)
        return self.head(torch.cat([pooled, emb], dim=-1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_huber(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    err = pred - y
    abs_err = err.abs()
    quad = torch.clamp(abs_err, max=delta)
    lin = abs_err - quad
    loss = 0.5 * quad**2 + delta * lin
    m = mask.clamp(0, 1)
    denom = m.sum().clamp_min(1.0)
    return (loss * m).sum() / denom


def skill_vs_persistence(y: np.ndarray, yhat: np.ndarray) -> float:
    mse = float(np.mean((yhat - y) ** 2))
    mse_p = float(np.mean(y**2))
    return float(1.0 - mse / mse_p) if mse_p > 0 else float("nan")


@torch.no_grad()
def evaluate(
    model: SharedGRU,
    loader: DataLoader,
    scaler: dict,
    horizons: tuple[int, ...],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    ys: dict[int, list[np.ndarray]] = {h: [] for h in horizons}
    yh: dict[int, list[np.ndarray]] = {h: [] for h in horizons}
    n_near_zero = 0
    n_pred = 0

    for batch in loader:
        pred = model(
            batch["node_x"].to(device),
            batch["node_mask"].to(device),
            batch["node_static"].to(device),
            batch["station_attrs"].to(device),
        ).cpu().numpy()
        y = batch["y"].numpy()
        mask = batch["y_mask"].numpy()
        for hi, h in enumerate(horizons):
            m = mask[:, hi] > 0
            if not m.any():
                continue
            y_h = y[m, hi]
            p_scaled = pred[m, hi]
            p_phys = inverse_target(p_scaled, h, scaler)
            ys[h].append(y_h.astype(float))
            yh[h].append(np.asarray(p_phys, dtype=float))
            n_near_zero += int(np.sum(np.abs(p_phys) < NEAR_ZERO_PHYS))
            n_pred += int(m.sum())

    per_h: dict[str, Any] = {}
    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    for h in horizons:
        if not ys[h]:
            per_h[str(h)] = {"n": 0, "rmse": float("nan"), "mae": float("nan"), "skill": float("nan")}
            continue
        y = np.concatenate(ys[h])
        p = np.concatenate(yh[h])
        all_y.append(y)
        all_p.append(p)
        per_h[str(h)] = {
            "n": int(len(y)),
            "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "mae": float(np.mean(np.abs(p - y))),
            "skill": skill_vs_persistence(y, p),
        }

    if all_y:
        y_all = np.concatenate(all_y)
        p_all = np.concatenate(all_p)
        overall_skill = skill_vs_persistence(y_all, p_all)
        overall_rmse = float(np.sqrt(np.mean((p_all - y_all) ** 2)))
    else:
        overall_skill = float("nan")
        overall_rmse = float("nan")

    # mean of per-horizon skills (used for early stopping)
    skills = [per_h[str(h)]["skill"] for h in horizons if np.isfinite(per_h[str(h)]["skill"])]
    mean_h_skill = float(np.mean(skills)) if skills else float("nan")

    return {
        "per_horizon": per_h,
        "overall_skill": overall_skill,
        "overall_rmse": overall_rmse,
        "mean_horizon_skill": mean_h_skill,
        "frac_near_zero": float(n_near_zero / n_pred) if n_pred else float("nan"),
        "n_pred": n_pred,
    }


def train_one_seed(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    sample_meta: dict[str, int],
    scaler: dict,
    horizons: tuple[int, ...],
    seed: int,
    device: torch.device,
    max_epochs: int = 80,
    patience: int = 15,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    hidden: int = 64,
    dropout: float = 0.1,
    grad_clip: float = 1.0,
) -> dict[str, Any]:
    set_seed(seed)
    model = SharedGRU(
        f_dyn=sample_meta["f_dyn"],
        n_static=sample_meta["n_static"],
        n_station_attrs=sample_meta["n_attrs"],
        hidden=hidden,
        station_emb_dim=4,
        n_horizons=len(horizons),
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_skill = -1e9
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    bad = 0
    history: list[dict[str, Any]] = []
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
            )
            loss = masked_huber(
                pred,
                batch["y_scaled"].to(device),
                batch["y_mask"].to(device),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(float(loss.item()))

        val = evaluate(model, val_loader, scaler, horizons, device)
        row = {
            "epoch": epoch,
            "train_huber": float(np.mean(losses)) if losses else float("nan"),
            "val_mean_h_skill": val["mean_horizon_skill"],
            "val_overall_skill": val["overall_skill"],
            "val_frac_near_zero": val["frac_near_zero"],
        }
        for h in horizons:
            row[f"val_skill_h{h}"] = val["per_horizon"][str(h)]["skill"]
            row[f"val_rmse_h{h}"] = val["per_horizon"][str(h)]["rmse"]
        history.append(row)

        skill = val["mean_horizon_skill"]
        improved = np.isfinite(skill) and skill > best_skill + 1e-4
        if improved:
            best_skill = float(skill)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        print(
            f"  seed={seed} ep={epoch:03d} loss={row['train_huber']:.4f} "
            f"val_skill={skill:.4f} near0={val['frac_near_zero']:.3f} "
            f"{'*' if improved else ''} bad={bad}",
            flush=True,
        )
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_best = evaluate(model, val_loader, scaler, horizons, device)
    test_best = evaluate(model, test_loader, scaler, horizons, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_mean_horizon_skill": best_skill,
        "n_params": n_params,
        "wall_s": time.time() - t0,
        "history": history,
        "val": val_best,
        "test": test_best,
        "state_dict": best_state,
    }


def _flatten_block(seed: int, block: str, metrics: dict) -> list[dict]:
    rows = []
    for h, m in metrics["per_horizon"].items():
        rows.append(
            {
                "seed": seed,
                "block": block,
                "horizon": int(h),
                "n": m["n"],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "skill": m["skill"],
                "overall_skill": metrics["overall_skill"],
                "mean_horizon_skill": metrics["mean_horizon_skill"],
                "frac_near_zero": metrics["frac_near_zero"],
            }
        )
    return rows


def run(
    out_dir: Path = _DEFAULT_OUT,
    seeds: tuple[int, ...] = SEEDS,
    batch_size: int = 128,
    max_epochs: int = 80,
    patience: int = 15,
    device: str | None = None,
    hidden: int = 64,
    lr: float = 3e-4,
    window_days: int | None = None,
    origins_path: Path | None = None,
    origins_scaled_path: Path | None = None,
    scaler_path: Path | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    scaler_p = Path(scaler_path) if scaler_path else _DEFAULT_SCALER
    scaler = json.loads(scaler_p.read_text())

    print(f"device={device_t}", flush=True)
    print("loading dataset (all blocks, shared panel cache)...", flush=True)
    ds_kwargs: dict[str, Any] = {"block": None}
    if window_days is not None:
        ds_kwargs["window_days"] = int(window_days)
    if origins_path is not None:
        ds_kwargs["origins_path"] = Path(origins_path)
        # force rebuild from raw origins + scaler (ignore stale scaled file)
        ds_kwargs["origins_scaled_path"] = Path("/nonexistent_scaled_origins.csv")
    if origins_scaled_path is not None:
        ds_kwargs["origins_scaled_path"] = Path(origins_scaled_path)
    if scaler_path is not None:
        ds_kwargs["scaler_path"] = Path(scaler_path)
    full_ds = RismaGraphDataset(**ds_kwargs)
    blocks = full_ds._blocks
    train_idx = [i for i, b in enumerate(blocks) if b == "train"]
    val_idx = [i for i, b in enumerate(blocks) if b == "val"]
    test_idx = [i for i, b in enumerate(blocks) if b == "test"]
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx)
    horizons = full_ds.horizons
    sample_meta = {
        "f_dyn": full_ds.f_dyn,
        "n_static": int(full_ds.node_static.shape[-1]),
        "n_attrs": full_ds.n_attrs,
        "n_nodes": len(full_ds.nodes),
        "window": full_ds.window,
    }
    print(
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"nodes={sample_meta['n_nodes']} T={sample_meta['window']} "
        f"horizons={horizons}",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_graphs
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_graphs
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_graphs
    )

    metric_rows: list[dict] = []
    seed_summaries: list[dict] = []
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    for seed in seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        res = train_one_seed(
            train_loader,
            val_loader,
            test_loader,
            sample_meta,
            scaler,
            horizons,
            seed=seed,
            device=device_t,
            max_epochs=max_epochs,
            patience=patience,
            hidden=hidden,
            lr=lr,
        )
        if res["state_dict"] is not None:
            torch.save(res["state_dict"], ckpt_dir / f"shared_gru_seed{seed}.pt")
        pd.DataFrame(res["history"]).to_csv(out_dir / f"history_seed{seed}.csv", index=False)
        metric_rows.extend(_flatten_block(seed, "val", res["val"]))
        metric_rows.extend(_flatten_block(seed, "test", res["test"]))
        seed_summaries.append(
            {
                "seed": seed,
                "best_epoch": res["best_epoch"],
                "n_params": res["n_params"],
                "wall_s": res["wall_s"],
                "val_mean_horizon_skill": res["val"]["mean_horizon_skill"],
                "val_overall_skill": res["val"]["overall_skill"],
                "val_frac_near_zero": res["val"]["frac_near_zero"],
                "test_mean_horizon_skill": res["test"]["mean_horizon_skill"],
                "test_overall_skill": res["test"]["overall_skill"],
                "test_frac_near_zero": res["test"]["frac_near_zero"],
                **{
                    f"val_skill_h{h}": res["val"]["per_horizon"][str(h)]["skill"]
                    for h in horizons
                },
                **{
                    f"test_skill_h{h}": res["test"]["per_horizon"][str(h)]["skill"]
                    for h in horizons
                },
                **{
                    f"val_rmse_h{h}": res["val"]["per_horizon"][str(h)]["rmse"]
                    for h in horizons
                },
                **{
                    f"test_rmse_h{h}": res["test"]["per_horizon"][str(h)]["rmse"]
                    for h in horizons
                },
            }
        )
        print(
            f"  BEST ep={res['best_epoch']} val_skill={res['val']['mean_horizon_skill']:.4f} "
            f"test_skill={res['test']['mean_horizon_skill']:.4f} "
            f"near0_val={res['val']['frac_near_zero']:.3f}",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "shared_gru_metrics.csv", index=False)
    summary = pd.DataFrame(seed_summaries)
    summary.to_csv(out_dir / "shared_gru_seeds.csv", index=False)

    # median / spread over seeds
    agg_rows = []
    for block in ("val", "test"):
        for h in horizons:
            sub = metrics[(metrics.block == block) & (metrics.horizon == h)]
            skills = sub["skill"].to_numpy(float)
            rmses = sub["rmse"].to_numpy(float)
            agg_rows.append(
                {
                    "block": block,
                    "horizon": h,
                    "n_seeds": int(len(skills)),
                    "skill_median": float(np.median(skills)),
                    "skill_mean": float(np.mean(skills)),
                    "skill_std": float(np.std(skills, ddof=0)),
                    "skill_min": float(np.min(skills)),
                    "skill_max": float(np.max(skills)),
                    "rmse_median": float(np.median(rmses)),
                    "rmse_mean": float(np.mean(rmses)),
                    "n": int(sub["n"].iloc[0]) if len(sub) else 0,
                }
            )
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out_dir / "shared_gru_summary.csv", index=False)

    near0 = summary["val_frac_near_zero"].median()
    lines = [
        "# Shared GRU (Step 5)",
        "",
        f"- Loader: `graph_tensor_loader` N={sample_meta['window']}, nodes={sample_meta['n_nodes']}",
        f"- Model: shared GRU hidden={hidden}, station_emb≤4, 3-horizon head, f_dyn={sample_meta['f_dyn']}",
        f"- Channels: VWC(+Δ1d/Δ3d) + precip memory 1/3/7/30 + met",
        f"- Train: masked Huber, AdamW lr={lr} / wd=1e-4, clip=1.0, stop on val mean-horizon skill",
        f"- Seeds: {list(seeds)}  |  n_params≈{seed_summaries[0]['n_params'] if seed_summaries else '?'}",
        f"- Val frac |ŷ|<1e-4 (median seed): **{near0:.3f}**",
        f"- schema_id: {scaler.get('schema_id')}",
        "",
        "| block | h | skill median | mean±std | min–max | RMSE median |",
        "|-------|---|--------------|----------|---------|-------------|",
    ]
    for block in ("val", "test"):
        for h in horizons:
            r = agg[(agg.block == block) & (agg.horizon == h)].iloc[0]
            lines.append(
                f"| {block} | {h} | {r['skill_median']:.3f} | "
                f"{r['skill_mean']:.3f}±{r['skill_std']:.3f} | "
                f"{r['skill_min']:.3f}–{r['skill_max']:.3f} | {r['rmse_median']:.5f} |"
            )
    lines += [
        "",
        "## Reading",
        "",
        "- Persistence baseline skill = 0 by construction (ŷ=0 on Δ).",
        "- LGBM past-only reference ≈ val 0.28/0.17/0.12, test 0.21/0.13/0.12 (h=1/3/7).",
        "- Step 7 decision: within ~±0.02 of LGBM → proceed; far below → debug before Step 8.",
        "- High `frac_near_zero` ⇒ collapse to persistence; check lr / scaling / early-stop.",
        "",
    ]
    report = "\n".join(lines) + "\n"
    (out_dir / "SHARED_GRU.md").write_text(report)
    print("\n" + report)

    meta = {
        "sample_meta": sample_meta,
        "horizons": list(horizons),
        "seeds": list(seeds),
        "device": str(device_t),
        "window_days": int(sample_meta["window"]),
        "origins_path": str(Path(origins_path)) if origins_path is not None else str(_STEP10 / "origins_long.csv"),
        "origins_scaled_path": (
            str(Path(origins_scaled_path)) if origins_scaled_path is not None else None
        ),
        "scaler_path": str(scaler_p),
        "schema_id": scaler.get("schema_id"),
        "out_dir": str(out_dir),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "lr": lr,
        "hidden": hidden,
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": batch_size,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--window", type=int, default=None, help="input window days (default: schema)")
    ap.add_argument("--origins", type=Path, default=None)
    ap.add_argument("--origins-scaled", type=Path, default=None)
    ap.add_argument("--scaler", type=Path, default=None)
    args = ap.parse_args()
    run(
        out_dir=args.out_dir,
        seeds=tuple(args.seeds),
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        hidden=args.hidden,
        lr=args.lr,
        window_days=args.window,
        origins_path=args.origins,
        origins_scaled_path=args.origins_scaled,
        scaler_path=args.scaler,
    )


if __name__ == "__main__":
    main()
