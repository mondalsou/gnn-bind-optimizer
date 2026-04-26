# GNNBindOptimizer

Heterogeneous GNN for protein-ligand binding affinity prediction + REINFORCE-based molecular generator, with SQL Server persistence, MLflow experiment tracking, and Streamlit UI.

![Overview](Figs/Overview.png)

*Fig 1. Overview of the pipeline.*

---

## Quick Start (Docker Compose)

```bash
# 1. Clone and enter repo
git clone <repo-url> GNNBindOptimizer
cd GNNBindOptimizer

# 2. Copy env file
cp .env.example .env

# 3. Cold start — builds images, trains GNN, runs RL, starts UI
docker compose up --build

# UI:     http://localhost:8501
# MLflow: http://localhost:5000
```

> **Apple Silicon / CPU-only:** Training runs on CPU (HGTConv scatter_reduce not supported on MPS). On a fresh Apple Silicon cold start, PyG native extensions may compile from source; the observed first end-to-end Docker run took ~55 minutes. Cached rebuilds are much faster.

### Dry-run (pre-trained checkpoints, skip training)

```bash
# Start only SQL Server + MLflow + Streamlit
docker compose up -d sqlserver mlflow streamlit

# Populate MLflow with existing checkpoint metrics + convergence curves
python scripts/log_existing_runs.py

# UI:     http://localhost:8501
# MLflow: http://localhost:5000  (3 runs: MTL, STL, RL)
```

---

## Manual / Notebook Setup

```bash
# Python 3.11 recommended
pip install -r requirements.txt

# Install PyG CPU wheels
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
pip install torch-scatter torch-sparse torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-${TORCH}+cpu.html

# Phase 1 — data pipeline + graph construction
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/01_data_graph_pipeline.ipynb \
    --ExecutePreprocessor.timeout=3600

# Phase 2 — GNN training + MTL ablation
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/02_gnn_model.ipynb \
    --ExecutePreprocessor.timeout=3600

# Phase 3 — RL molecular generator
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/03_rl_generator.ipynb \
    --ExecutePreprocessor.timeout=600

# Streamlit UI (local, no SQL Server — falls back to demo data)
streamlit run app/streamlit_app.py
```

For real AutoDock Vina scores on the **GNN vs Vina** page, the local app needs
the `vina` executable on `PATH` plus Meeko's preparation scripts. The Docker
Streamlit image installs these for cold starts, so the most reproducible route is:

```bash
docker compose up --build streamlit
```

### Local data and artifacts

Docker Compose keeps project outputs local:

- `./data:/workspace/data` stores processed graphs and RL outputs.
- `./checkpoints:/workspace/checkpoints` stores GNN and RL checkpoints.
- `./src` and `./notebooks` are mounted for Streamlit/training development.
- MLflow metrics are stored in the SQL Server `mlflowdb` Docker volume.
- New MLflow artifacts are served through the MLflow artifact proxy and stored in the `mlflow_artifacts` Docker volume.

---

## Project Structure

```
GNNBindOptimizer/
├── notebooks/
│   ├── 01_data_graph_pipeline.ipynb   # PDB → HeteroData graphs
│   ├── 02_gnn_model.ipynb             # HeteroGNN training + MTL ablation
│   └── 03_rl_generator.ipynb          # REINFORCE molecular generator
├── src/
│   ├── graph/          # Graph construction utilities
│   ├── rl/             # SMILESTokenizer, SMILESPolicy, load_policy()
│   ├── docking/        # AutoDock Vina + Meeko preparation helpers
│   └── db/             # SQLAlchemy connection helper
├── app/
│   └── streamlit_app.py               # 5-page Streamlit UI
├── db/
│   ├── init.sql                       # SQL Server schema + seed data
│   └── queries.sql                    # Analytical SQL examples
├── scripts/
│   └── log_existing_runs.py           # Backfill MLflow from checkpoints + TensorBoard logs
├── docker/
│   ├── Dockerfile.trainer
│   ├── Dockerfile.rl
│   ├── Dockerfile.streamlit
│   ├── Dockerfile.mlflow
│   └── entrypoint-sqlserver.sh
├── checkpoints/                       # GNN Lightning checkpoints (MTL + STL)
├── data/
│   ├── processed/dataset.pt           # 150 HeteroData graphs
│   └── rl_results/rl_results.json     # Generated molecules + rewards
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── ARCHITECTURE.md
└── README.md
```

---

## Results Summary

### GNN Training (150 PDBbind complexes)

| Model | val RMSE | Pearson r | Pose AUC |
|-------|----------|-----------|----------|
| MTL (affinity + pose + selectivity) | **1.924** | 0.541 | 0.778 |
| STL (affinity only) | 2.034 | 0.489 | — |
| **Test set (MTL)** | **1.702** | **0.579** | **0.796** |

MTL improves validation RMSE by 5.4% (Δ = 0.11) via Kendall uncertainty-weighted multi-task loss. On the small held-out test split, STL is marginally lower-error (`test_rmse=1.683`) than MTL (`test_rmse=1.702`), so this should be read as a demo-scale result rather than a definitive architecture ranking.

### RL Generator (150 steps × 64 samples, pocket 6E9A)

| Metric | Value |
|--------|-------|
| Total sampled SMILES | 9,600 |
| Valid molecules collected | 280 |
| Full-run validity | 2.9% |
| Top molecules saved | 18 |
| Best reward | 0.719 |
| Best predicted pKd | 7.59 |
| Best-reward mol | `O=C(Nc1ccccc1)c1cccc(C(F)(F)F)c1` |

Top saved molecules are reward-ranked valid structures. The low 2.9% validity is an important modeling caveat: the character-level SMILES policy can find useful hits, but it is not yet a chemically robust generator.

---

## Streamlit UI Pages

| Page | Description |
|------|-------------|
| Summary | KPI cards, phase timeline, MTL vs STL bar chart, RL reward curve |
| Binding Predictor | Input SMILES → GNN pKd + pose prob + selectivity |
| RL Generator | Top 18 generated molecules — structure, pKd, QED, reward |
| GNN vs Vina | Live GNN pKd vs real AutoDock Vina docking on top RL molecules |
| SQL Console | Raw SELECT query → rendered table + CSV download |

Header badge shows live pocket and best pKd from `rl_results.json`.

UI smoke test after the cold start:

- Streamlit renders all five pages at `http://localhost:8501`.
- SQL Console connects to SQL Server and the default read-only query returns rows.
- MLflow renders at `http://localhost:5000` and shows `gnn_mtl`, `gnn_stl`, and `reinforce_rl`.
- The GNN vs Vina page defaults to docking 10 molecules; users can lower the slider for faster Vina iteration.

---

## MLflow Experiment Tracking

MLflow runs against SQL Server (`mlflowdb` database). The Docker stack uses `MLFLOW_BACKEND_STORE_URI` for the server and `MLFLOW_TRACKING_URI=http://mlflow:5000` for clients so metrics and artifacts route through the tracking server correctly. The cold-start experiment is **gnn_bind_optimizer**:

| Run | Key metrics |
|-----|-------------|
| `gnn_mtl` | best_val_rmse=1.924, test_rmse=1.702, test_pearson_r=0.579, test_pose_auc=0.796 |
| `gnn_stl` | best_val_rmse=2.034, test_rmse=1.683, test_pearson_r=0.603 |
| `reinforce_rl` | best_pred_pkd=7.59, best_reward_total=0.719, total_valid=280, validity_rate=2.9% |

To backfill MLflow from existing checkpoints + TensorBoard logs:
```bash
python scripts/log_existing_runs.py [--tracking-uri http://localhost:5000]
```

---

## SQL Server Schema

See `db/init.sql` for full DDL. Five tables:

| Table | Purpose |
|-------|---------|
| `experiments` | Config registry (JSON blob) |
| `model_runs` | Per-epoch training metrics |
| `binding_predictions` | On-demand Streamlit predictions |
| `rl_molecules` | Generated molecules + reward breakdown |
| `vina_benchmarks` | GNN vs Vina correlation |

MLflow uses a separate `mlflowdb` database to avoid schema conflicts with `dbo.experiments`.

---

## Key Design Decisions

See `ARCHITECTURE.md` for full rationale on:
- Heterogeneous graph construction (node/edge feature choices, distance cutoffs)
- HGT vs GAT vs SchNet trade-offs
- Multi-task learning with Kendall uncertainty weighting
- REINFORCE vs PPO for molecular generation
- Character-level LSTM vs fragment-based generators
- Oracle approximation (ETKDG + centroid alignment vs full docking)
- AutoDock Vina comparison path (`autodock-vina` executable + Meeko PDBQT prep)
- MLflow + SQL Server backend topology
