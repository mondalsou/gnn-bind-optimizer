# GNNBindOptimizer

Heterogeneous GNN for protein-ligand binding affinity prediction + REINFORCE-based molecular generator, with SQL Server persistence and Streamlit UI.

Built as a take-home SBDD exercise. Target: EGFR kinase (PDB 1IEP / PDBbind refined set).

---

## Quick Start (Docker Compose)

```bash
# 1. Clone and enter repo
git clone <repo-url> GNNBindOptimizer
cd GNNBindOptimizer

# 2. Copy env file and set password (or use defaults)
cp .env.example .env

# 3. Cold start — builds images, trains GNN, runs RL, starts UI
docker compose up --build

# UI: http://localhost:8501
# MLflow: http://localhost:5000
```

> **Apple Silicon / CPU-only:** Training runs on CPU (HGTConv scatter_reduce is not supported on MPS). Full pipeline takes ~10 min on a modern laptop.

---

## Manual / Notebook Setup

```bash
# Python 3.11 recommended
pip install -r requirements.txt

# Install PyG CPU wheels (auto-detects torch version)
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
pip install torch-scatter torch-sparse torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-${TORCH}+cpu.html

# Phase 1 — data pipeline + graph construction
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/phase1_data_graph_pipeline.ipynb \
    --ExecutePreprocessor.timeout=3600

# Phase 2 — GNN training + MTL ablation
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/phase2_gnn_model.ipynb \
    --ExecutePreprocessor.timeout=3600

# Phase 3 — RL molecular generator
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/phase3_rl_generator.ipynb \
    --ExecutePreprocessor.timeout=600

# Streamlit UI (local, no SQL Server required — falls back to demo data)
streamlit run app/streamlit_app.py
```

---

## Project Structure

```
GNNBindOptimizer/
├── notebooks/
│   ├── phase1_data_graph_pipeline.ipynb   # PDB → HeteroData graphs
│   ├── phase2_gnn_model.ipynb             # HeteroGNN training + MTL ablation
│   └── phase3_rl_generator.ipynb         # REINFORCE molecular generator
├── src/
│   ├── graph/          # Graph construction utilities
│   ├── models/         # HeteroGNN + prediction heads
│   │   └── gnn_state.pt                  # Exported weights + test metrics
│   ├── rl/             # SMILESTokenizer, SMILESPolicy, load_policy()
│   └── db/             # SQLAlchemy connection helper
├── app/
│   └── streamlit_app.py                  # 5-page UI
├── db/
│   ├── init.sql                          # SQL Server schema + seed data
│   └── queries.sql                       # 7 example analytical queries
├── docker/
│   ├── Dockerfile.trainer
│   ├── Dockerfile.rl
│   ├── Dockerfile.streamlit
│   ├── Dockerfile.mlflow
│   └── entrypoint-sqlserver.sh
├── checkpoints/                          # GNN + RL policy checkpoints
├── data/
│   ├── processed/dataset.pt              # 150 HeteroData graphs
│   └── rl_results/rl_results.json        # Generated molecules + rewards
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── ARCHITECTURE.md                       # Design decisions + rationale
└── README.md
```

---

## Results Summary

### Phase 2 — GNN Training (150 PDBbind complexes)

| Model | val RMSE | Pearson r | Pose AUC |
|-------|----------|-----------|----------|
| MTL (affinity + pose + selectivity) | **1.924** | 0.541 | 0.778 |
| STL (affinity only) | 2.034 | 0.489 | — |
| **Test set (MTL)** | **1.702** | **0.579** | **0.796** |

MTL improves val RMSE by 5.4% (Δ = 0.11) via Kendall uncertainty-weighted multi-task loss.

### Phase 3 — RL Generator (300 steps × 32 mols)

| Metric | Value |
|--------|-------|
| Valid molecules collected | 66 |
| Best reward | 0.709 |
| Best predicted pKd | 7.58 |
| Best mol | `NS(=O)(=O)c1ccc(C(=O)N2CCC(O)(c3ccccc3)CC2)cc1` |

Top molecules are sulfonamide scaffolds with drug-like properties (QED > 0.8, SA > 0.79, MW < 500).

---

## Streamlit UI Pages

| Page | Description |
|------|-------------|
| Dashboard | Experiment table + MTL vs STL bar chart |
| Binding Predictor | Input SMILES → GNN pKd + pose + selectivity |
| RL Browser | Table + scatter + 2D structures of generated mols |
| GNN vs Vina | Parity plot on benchmark set |
| SQL Console | Raw SELECT query → rendered table + CSV download |

---

## SQL Server Schema

See `db/init.sql` for full DDL. `db/queries.sql` contains 7 analytical queries covering:
- Best RL molecule per experiment
- MTL vs STL ablation comparison
- RL reward trajectory (10-step moving average)
- Drug-likeness distribution (QED/SA buckets)
- GNN vs Vina parity
- Top binding predictions across all runs
- Pareto-front molecules (high affinity + high drug-likeness)

---

## Key Design Decisions

See `ARCHITECTURE.md` for full rationale on:
- Heterogeneous graph construction (node/edge feature choices, distance cutoffs)
- HGT vs GAT vs SchNet trade-offs
- Multi-task learning with Kendall uncertainty weighting
- REINFORCE vs PPO for molecular generation
- Character-level LSTM vs fragment-based generators
- Oracle approximation (ETKDG + centroid alignment vs full docking)
