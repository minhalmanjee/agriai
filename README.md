# agriai 3-cluster / 3-GPU DDP (RISMA)

This repo is the **Carman | Sturgeon | Kenaston** distributed training path only.
One GPU per cluster, three ranks total. Nothing else from the broader pipeline is here on purpose.

## What to run

```bash
# from repo root, with 3 visible GPUs and a Python env that has torch/numpy/pandas
bash graph/risma_pipeline/run_cluster3_ddp.sh
```

That launches `torch.distributed.run --nproc_per_node=3` and trains `td_mlp` at depths 5, 20, 50, 100 (Hasan years, absolute residual, daily, window=2d).

Outputs land under:
`graph/risma_processed/step10/e3_followup/cluster3_ddp/`

## File map

| File | Role |
|------|------|
| `graph/risma_pipeline/run_cluster3_ddp.sh` | Launcher. Hard-codes layout `Carman,Sturgeon,Kenaston`, 3 processes, depth loop, then a small compare vs a local single-GPU baseline if those preds exist. |
| `graph/risma_pipeline/train_time_depth_gru.py` | Training entrypoint. `--ddp` + `--ddp-region-layout` shards stations by cluster, wraps the model in DDP, syncs early-stopping / val metrics across ranks. |
| `graph/risma_pipeline/station_attributes_model12.csv` | Station → cluster table (Carman–Elm Creek, Sturgeon Creek, Kenaston). Used for layout aliases and post-run scoring. |
| `graph/risma_pipeline/train_shared_gru.py` | Shared GRU backbone, losses, skill helpers used by the trainer. |
| `graph/risma_pipeline/train_shared_gru_s1_wetting.py` | Wetting / S1 helpers imported by the trainer. |
| `graph/risma_pipeline/train_shared_gru_hybrid.py` | Hybrid graph dataset used when that variant is selected. |
| `graph/risma_pipeline/graph_tensor_loader.py` | Builds RISMA graph tensors / collate for the DataLoader. |
| `graph/risma_pipeline/depth_specs.py` | Depth schema (5 / 20 / 50 / 100 cm, etc.). |
| `graph/risma_pipeline/step10b_node_scaler.py` | Target scaling / inverse transforms for absolute VWC residuals. |
| `graph/risma_pipeline/intraday_precip_features.py` | Intraday precip feature names (imported even on the daily path). |
| `requirements.txt` | Minimal Python deps for this slice. |

## Rank layout
- **GPU 0 / rank 0** — Carman–Elm Creek (MB1, MB2, MB3, MB5, MB9)
- **GPU 1 / rank 1** — Sturgeon Creek (MB10, MB11, MB12)
- **GPU 2 / rank 2** — Kenaston (SK1–SK4)

Test evaluation uses all 12 stations after training.

## Notes

- Needs preprocessed RISMA graph data on disk under the usual `graph/risma_processed/...` paths expected by the loader (not shipped in this repo).
- Point `PY` in the shell script at your interpreter if you are not using `./venv/bin/python`.
