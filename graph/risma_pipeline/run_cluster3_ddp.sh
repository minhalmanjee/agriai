#!/usr/bin/env bash
# 3-cluster region DDP: one GPU per thesis cluster.
# Rank0=Carman–Elm Creek, Rank1=Sturgeon Creek, Rank2=Kenaston.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=./venv/bin/python
OUT=graph/risma_processed/step10/e3_followup/cluster3_ddp
DEPTHS=(5 20 50 100)
LAYOUT=Carman,Sturgeon,Kenaston
COMMON=(--variants td_mlp --seeds 0 --resolution daily --window-days 2
        --split-preset hasan_sk4 --target-kind absolute --num-workers 0)

mkdir -p "$OUT"
echo "[$(date +%H:%M:%S)] START cluster3 DDP host=$(hostname) nvidia=$(ls /dev/nvidia* 2>/dev/null | wc -l)" | tee "$OUT/launch.log"
echo "layout=$LAYOUT (Carman MB1/2/3/5/9 | Sturgeon MB10/11/12 | Kenaston SK1–4)" | tee -a "$OUT/launch.log"

for depth in "${DEPTHS[@]}"; do
  ddir="$OUT/td_mlp_daily_w2d_d${depth}"
  mkdir -p "$ddir"
  echo "[$(date +%H:%M:%S)] CLUSTER3 d=$depth" | tee -a "$OUT/launch.log"
  "$PY" -m torch.distributed.run --standalone --nproc_per_node=3 \
    -m graph.risma_pipeline.train_time_depth_gru \
    "${COMMON[@]}" --ddp --ddp-region-layout "$LAYOUT" \
    --target-depth "$depth" --out-dir "$ddir" \
    >"$ddir/train_seed0.log" 2>&1
  echo "[$(date +%H:%M:%S)] CLUSTER3 d=$depth DONE exit=$?" | tee -a "$OUT/launch.log"
done

"$PY" - <<'PY' | tee "$OUT/compare_stdout.txt"
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path("graph/risma_processed/step10/e3_followup")
c3 = root / "cluster3_ddp"
single = root / "region_fault_ddp" / "single_all12"
attr = pd.read_csv("graph/risma_pipeline/station_attributes_model12.csv")
st2c = {str(r.station): str(r.cluster) for _, r in attr.iterrows()}
HS = (1, 3, 7)
DEPTHS = ("5", "20", "50", "100")

def rmse(y, yh):
    return float(np.sqrt(np.mean((yh - y) ** 2)))

def score(pred):
    pred = pred.copy()
    pred["cluster"] = pred.station.map(st2c)
    rows = []
    for h in HS:
        sub = pred[pred.horizon == h]
        for label, g in [
            ("ALL", sub),
            ("Carman_Elm_Creek", sub[sub.cluster == "Carman_Elm_Creek"]),
            ("Sturgeon_Creek", sub[sub.cluster == "Sturgeon_Creek"]),
            ("Kenaston", sub[sub.cluster == "Kenaston"]),
        ]:
            if g.empty:
                continue
            rows.append(dict(h=h, group=label, n=len(g), rmse=rmse(g.y.to_numpy(float), g.yhat.to_numpy(float))))
    return rows

rows = []
for d in DEPTHS:
    meta = json.loads((c3 / f"td_mlp_daily_w2d_d{d}" / "train_meta_td_mlp_seed0.json").read_text())
    pred_c = pd.read_csv(c3 / f"td_mlp_daily_w2d_d{d}" / "preds" / "td_mlp_seed0_test.csv")
    pred_s = pd.read_csv(single / f"td_mlp_daily_w2d_d{d}" / "preds" / "td_mlp_seed0_test.csv")
    sc_c = { (r["group"], r["h"]): r for r in score(pred_c) }
    sc_s = { (r["group"], r["h"]): r for r in score(pred_s) }
    for g in ("ALL", "Carman_Elm_Creek", "Sturgeon_Creek", "Kenaston"):
        for h in HS:
            rc, rs = sc_c[(g, h)], sc_s[(g, h)]
            pct = 100.0 * (rc["rmse"] - rs["rmse"]) / rs["rmse"]
            rows.append(dict(
                depth=d, h=h, group=g, n=rc["n"],
                single=rs["rmse"], cluster3=rc["rmse"], pct_vs_single=pct,
                ddp=bool(meta.get("ddp")),
                region=bool(meta.get("ddp_region_partition")),
                layout=meta.get("ddp_train_stations"),
            ))

df = pd.DataFrame(rows)
df.to_csv(c3 / "compare_cluster3_vs_single_all12.csv", index=False)

all_df = df[df.group == "ALL"].copy()
print("Protocol: Hasan years · abs residual · daily w2d · seed 0 · test=all 12")
print("cluster3 DDP: GPU0=Carman (MB1/2/3/5/9) | GPU1=Sturgeon (MB10/11/12) | GPU2=Kenaston (SK1–4)\n")
print(f"{'depth':>5} {'h':>3} {'single':>10} {'cluster3':>10} {'Δ%':>8} {'n':>6}")
for _, r in all_df.iterrows():
    print(f"{r.depth:>5} {int(r.h):>3} {r.single:10.5f} {r.cluster3:10.5f} {r.pct_vs_single:+7.2f}% {int(r.n):>6}")
a = all_df.pct_vs_single.abs()
print(f"\nALL |Δ|: median={a.median():.2f}% mean={a.mean():.2f}% max={a.max():.2f}%")
print(f"cluster3 better: {(all_df.pct_vs_single<0).sum()}/12  worse: {(all_df.pct_vs_single>0).sum()}/12")

md = [
    "# cluster3 DDP vs single GPU (all 12 stations)",
    "",
    "Protocol: Hasan years · absolute residual · daily w2d · seed 0.",
    "Ranks: Carman–Elm Creek | Sturgeon Creek | Kenaston.",
    "",
    "## ALL-station test RMSE",
    "",
    "| depth | h | single | cluster3 | Δ% |",
    "|--|--:|--:|--:|--:|",
]
for _, r in all_df.iterrows():
    md.append(f"| {r.depth} | {int(r.h)} | {r.single:.5f} | {r.cluster3:.5f} | {r.pct_vs_single:+.2f}% |")
md.append(f"\nMedian |Δ| = {a.median():.2f}% · mean = {a.mean():.2f}% · max = {a.max():.2f}%\n")
md.append("## Per-cluster test RMSE (cluster3)\n")
md.append("| depth | h | Carman | Sturgeon | Kenaston |")
md.append("|--|--:|--:|--:|--:|")
for d in DEPTHS:
    for h in HS:
        c = df[(df.depth == d) & (df.h == h) & (df.group == "Carman_Elm_Creek")].iloc[0]
        s = df[(df.depth == d) & (df.h == h) & (df.group == "Sturgeon_Creek")].iloc[0]
        k = df[(df.depth == d) & (df.h == h) & (df.group == "Kenaston")].iloc[0]
        md.append(f"| {d} | {h} | {c.cluster3:.5f} | {s.cluster3:.5f} | {k.cluster3:.5f} |")
(c3 / "COMPARE_CLUSTER3_VS_SINGLE.md").write_text("\n".join(md) + "\n")
print("Wrote", c3 / "compare_cluster3_vs_single_all12.csv")
PY

echo "[$(date +%H:%M:%S)] ALL_DONE" | tee -a "$OUT/launch.log"
