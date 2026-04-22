# GNNBindOptimizer — Implementation Plan

## Stack
- **GNN**: PyTorch Geometric, heterogeneous graph (protein + ligand nodes + interaction edges)
- **RL**: RDKit + REINVENT-style SMILES generator, GNN as frozen scorer
- **DB**: SQL Server 2022 + MLflow (SQL backend)
- **Frontend**: Streamlit
- **Infra**: Docker Compose (5 services)

---

## Phase 1 — Data & Graph Pipeline (Day 1, ~8h)

1. **Dataset**: PDBbind v2020 refined set (protein-ligand complexes + pKd/pIC50 labels)
2. **Graph construction** (`src/graph/`):
   - Protein pocket nodes: residue-level features (AA type, SASA, phi/psi) — cutoff 6Å from ligand
   - Ligand nodes: atom features (atomic num, hybridization, charge, ring membership)
   - Edges: ligand-ligand (bonds), protein-protein (Cα-Cα < 8Å), protein-ligand (< 5Å = interaction edges)
   - Edge features: distance, bond type, H-bond donor/acceptor flags
3. **`DataModule`**: PyTorch Lightning, handles PDB parsing with BioPython + RDKit

---

## Phase 2 — GNN Architecture (Day 1-2, ~6h)

`src/models/gnn_bind.py`:
- `HeteroGNN`: 4 × `HGTConv` layers (Heterogeneous Graph Transformer) with residual
- Global readout: attention pooling over ligand + pocket nodes → concat → MLP head
- **3 output heads** (multi-task):
  - `affinity_head`: scalar pKd/pIC50
  - `pose_head`: binary (RMSD < 2Å)
  - `selectivity_head`: binary (EGFR vs ABL1)
- Loss: weighted sum — MSE + BCE + BCE, ablation with/without MTL

---

## Phase 3 — RL Generator (Day 2, ~6h)

`src/rl/`:
- **Policy**: RNN-based SMILES generator (pre-trained on ChEMBL)
- **Frozen oracle**: trained GNN predicts pKd for generated mol + EGFR pocket graph
- **Reward components**:
  1. `r_affinity` = GNN pKd prediction (normalized)
  2. `r_qed` = QED drug-likeness score (RDKit)
  3. `r_sa` = 1 - SA_score/10 (synthetic accessibility)
  4. `r_mw` = penalty if MW > 500
  - `R = 0.5·r_affinity + 0.2·r_qed + 0.2·r_sa + 0.1·r_mw`
- **Algorithm**: REINFORCE with KL-div penalty vs prior (prevent mode collapse)
- Each generated mol → logged to SQL Server

---

## Phase 4 — SQL Server Schema (Day 2, ~3h)

`db/init.sql` tables:
```sql
experiments(id, name, created_at, config_json)
model_runs(id, experiment_id, mlflow_run_id, epoch, val_rmse, val_auc_pose, val_auc_select)
binding_predictions(id, run_id, smiles, pocket_pdb, pred_pkd, pred_pose_prob, pred_select_prob, created_at)
rl_molecules(id, experiment_id, step, smiles, reward, r_affinity, r_qed, r_sa, r_mw, created_at)
vina_benchmarks(id, smiles, pocket_pdb, vina_score, gnn_pred_pkd)
```
- MLflow configured with `MLFLOW_TRACKING_URI=mssql+pyodbc://...`

---

## Phase 5 — Docker Compose Stack (Day 3, ~4h)

`docker-compose.yml` services:
1. `sqlserver` — mcr.microsoft.com/mssql/server:2022-latest
2. `mlflow` — custom image, SQL Server backend
3. `gnn-trainer` — training job (exits after done, checkpoints to volume)
4. `rl-agent` — RL loop (depends on trained GNN checkpoint)
5. `streamlit` — UI, reads SQL Server

---

## Phase 6 — Streamlit UI (Day 3, ~4h)

`app/streamlit_app.py` pages:
- **Dashboard**: experiment table, loss curves from MLflow
- **Binding Predictor**: input SMILES → live GNN prediction → store to DB
- **RL Browser**: table of RL-generated mols with reward breakdown, 2D structure
- **GNN vs Vina Scatter**: parity plot on benchmark set
- **SQL Console**: raw query box → renders table

---

## Phase 7 — Docs & Recording (Day 4, ~4h)

- `ARCHITECTURE.md`: graph construction decisions, GNN justification, RL reward design, schema rationale
- `README.md`: quick-start, env setup
- `db/queries.sql`: 5+ example queries
- Screen recording (3-5 min): cold `docker compose up` → full UI walkthrough

---

## Repo Structure
```
GNNBindOptimizer/
├── src/
│   ├── graph/          # PDB→hetero graph pipeline
│   ├── models/         # GNN + multi-task heads
│   ├── rl/             # SMILES generator + REINFORCE
│   └── db/             # SQLAlchemy models
├── app/                # Streamlit
├── db/                 # init.sql, queries.sql
├── docker/             # Dockerfiles per service
├── docker-compose.yml
├── ARCHITECTURE.md
├── README.md
└── .env.example
```

---

## Timeline
| Day | Work | Hours |
|-----|------|-------|
| 1 | Data pipeline + graph construction + GNN build | ~14h |
| 2 | GNN training + MTL ablation + RL loop | ~9h |
| 3 | Docker Compose + SQL schema + Streamlit UI | ~8h |
| 4 | ARCHITECTURE.md + README + screen recording | ~4h |
| **Total** | | **~31h** |
