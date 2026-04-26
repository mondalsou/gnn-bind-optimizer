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

> **Apple Silicon / CPU-only:** Training runs on CPU (HGTConv scatter_reduce not supported on MPS). Full pipeline ~10 min on a modern laptop.

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

MTL improves val RMSE by 5.4% (Δ = 0.11) via Kendall uncertainty-weighted multi-task loss.

### RL Generator (150 steps × 64 mols, pocket 6E9A)

| Metric | Value |
|--------|-------|
| Total molecules generated | 147 |
| Top molecules saved | 18 |
| Best reward | 0.720 |
| Best predicted pKd | 7.57 |
| Best mol | `Cc1ccc(C(=O)NCCCCCCC(=O)N=O)cc1` |

Top molecules are amide scaffolds with drug-like properties (SA > 0.79, MW < 500). Prior trained for 60 epochs.

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

---

## MLflow Experiment Tracking

MLflow runs against SQL Server (`mlflowdb` database). Three pre-logged runs in experiment **GNNBindOptimizer**:

| Run | Key metrics |
|-----|-------------|
| `gnn_mtl_baseline` | val_rmse=1.924, Pearson r=0.541, Pose AUC=0.778 — 30-epoch convergence curve |
| `gnn_stl_ablation` | val_rmse=2.034, Pearson r=0.489 — 17-epoch convergence curve |
| `rl_reinforce_egfr` | best_pkd=7.57, best_reward=0.720, validity=100% — reward curve (mean + max per step) |

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
