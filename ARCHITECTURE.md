# GNNBindOptimizer — Architecture

## 1. Dataset & Graph Construction

### Dataset
PDBbind v2020 refined set, filtered to 150 protein–ligand complexes with measured pKd values.

![EDA overview](data/processed/eda_nature.png)

*Fig 1. Dataset statistics (n=150). a) pKd distribution — mean 6.4, range 2.0–11.9. b) Ligand molecular weight — mean 380 Da, majority below 500 (Lipinski compliant). c) Pocket residue count — mean 27, range 9–43. d) Protein–ligand interaction edge count per complex — mean 27, long tail to 94.*

### Why a heterogeneous graph?
Protein-ligand binding involves two chemically distinct entity types (amino acid residues and small-molecule atoms) with fundamentally different feature spaces. A homogeneous GNN would require padding both to the same dimensionality, discarding the structural distinction. A heterogeneous graph preserves it natively and lets message-passing attend separately to intra-ligand, intra-pocket, and cross-interface edges.

### Node features
| Node type | Dim | Key features |
|-----------|-----|-------------|
| `ligand`  | 28  | atomic number (one-hot 11), degree (one-hot 6), hybridization (one-hot 5), formal charge, radical electrons, H count, aromaticity, ring membership |
| `residue` | 26  | amino-acid type (one-hot 21), backbone φ/ψ (sin/cos → 4 values), Kyte–Doolittle hydrophobicity |

### Edges
| Edge type | Rule | Features |
|-----------|------|----------|
| `bond` (lig→lig) | RDKit bonds | bond type (one-hot 4), conjugated, ring, stereo |
| `contact` (res→res) | Cα–Cα < 8 Å | Euclidean distance |
| `interacts` (lig→res) | any heavy-atom pair < 5 Å | distance, H-bond donor/acceptor flags |
| `interacts_rev` (res→lig) | reverse of above | same |

The 5 Å interaction cutoff captures direct van-der-Waals and electrostatic contacts without pulling in distant residues that contribute little to binding energetics.

---

## 2. GNN Architecture — HeteroGNN (HGTConv)

![GNN architecture](Figs/GNN-model-arch.png)

*Fig 2. HeteroGNN pipeline. Ligand atom graph and protein pocket residue graph are encoded together through 4 stacked HGTConv layers. Type-specific attentional pooling produces one embedding per modality; these are concatenated and passed through a shared MLP trunk before splitting into three task heads.*

### Why Heterogeneous Graph Transformer (HGT)?
HGT uses type-specific attention heads and relation-specific key-query-value projections, making it the natural fit when node and edge types carry different semantics.

| Architecture | Pros | Cons |
|---|---|---|
| HGTConv (chosen) | Type-aware attention; handles asymmetric feature dims natively | Higher compute than homogeneous GCN |
| HANConv | Metapath-based, strong for few relation types | Requires manual metapath design |
| GATConv (homogeneous) | Simpler; proven on molecular tasks | Loses protein/ligand distinction |
| SchNet/DimeNet | Explicit 3D geometry | Requires high-quality docked poses; fragile to conformer errors |

### Architecture details
- 4 × `HGTConv` layers, hidden dim 128, 4 attention heads
- Residual connections after each layer (add + LayerNorm)
- Global readout: type-specific attentional pooling → concatenate → 2-layer MLP trunk
- **3 prediction heads** (multi-task):

| Head | Task | Loss |
|------|------|------|
| `affinity_head` | scalar pKd regression | Smooth-L1 |
| `pose_head` | binary: RMSD < 2 Å | BCE |
| `selectivity_head` | binary: EGFR vs ABL1 | BCE |

### MTL vs STL results

![Part 2 results](checkpoints/phase2_results.png)

*Fig 3. Test-set parity plots (n=15). a) MTL: RMSE=1.702, MAE=1.492. b) STL (affinity-only): RMSE=1.683, MAE=1.281. At n=150 the difference is within noise — STL shows marginally lower error on this split. MTL is expected to gain at larger dataset sizes where auxiliary supervision regularizes the shared encoder more effectively.*

### Why multi-task learning?
Pose quality and selectivity act as auxiliary signals sharing representations with the affinity prediction path. At 150 samples the benefit is marginal (MTL RMSE 1.702 vs STL 1.683); the expected benefit emerges at 5000+ complexes where the auxiliary heads act as regularizers.

**Uncertainty weighting (Kendall et al., 2018):** Each loss term is divided by a learned task variance σ² (log σ² is the actual parameter for numerical stability). This avoids manually tuning loss weights and lets the model adaptively balance tasks during training.

---

## 3. RL Molecular Generator

![LSTM policy and REINFORCE loop](Figs/LSTM-model-arch.png)

*Fig 4. REINFORCE training loop. SMILES corpus → character-level tokenizer → embedding → 2-layer LSTM policy → autoregressive token sampling → generated SMILES → 3D graph construction in reference pocket (6E9A) → frozen GNN oracle scores pKd → composite reward (affinity + QED + SA + MW) → REINFORCE gradient update. KL penalty against frozen prior (dashed) prevents mode collapse.*

### Why REINFORCE over PPO/GRPO?
For small-molecule generation, REINFORCE with an EMA baseline is computationally lighter and easier to interpret. PPO adds a value-network head and clipping mechanics that provide marginal benefit at this scale. Mode collapse is addressed via the KL-divergence penalty against the frozen prior.

### Policy: Character-level LSTM
- 2 layers, hidden dim 512, embedding dim 128
- Token vocabulary: single characters + two-char tokens (Cl, Br, @@, ...) → 37 tokens
- Pre-trained for 120 epochs on ~154 SMILES (PDBbind training ligands + 29 known EGFR inhibitors) using teacher-forcing cross-entropy

### Critical implementation note — eval mode during sampling
The LSTM must run in `eval()` mode during autoregressive generation. In `train()` mode, inter-layer dropout fires at every token step; compounded over 120 autoregressive steps this collapses SMILES validity to ~0%. Gradients flow through `log_p.gather()` regardless of train/eval state — REINFORCE backward is unaffected.

### Why character-level over fragment-based?
Character-level is simpler and produces more diverse output without requiring a curated fragment library. The trade-off is lower initial validity; fragment-based generators (REINVENT 4, GraphINVENT) achieve higher validity natively but are more complex to set up.

### Reward function
```
R = 0.5·r_affinity + 0.2·r_qed + 0.2·r_sa + 0.1·r_mw
```
| Component | Rationale |
|---|---|
| `r_affinity` | Normalized GNN pKd — primary optimization target |
| `r_qed` | Drug-likeness (Bickerton 2012) — composite Lipinski compliance |
| `r_sa` | 1 − SA_score/10 — synthetic accessibility (Ertl & Schuffenhauer) |
| `r_mw` | 1 if MW ≤ 500, linear penalty above — hard Lipinski MW filter |

Affinity weight (0.5) deliberately dominates to bias toward high-affinity molecules; QED and SA prevent convergence on synthetically inaccessible structures.

### RL results

![Part 3 results](data/rl_results/phase3_results.png)

*Fig 5. RL training summary (150 steps, batch 64, prior 60 epochs, reference pocket 6E9A pKd=11.92). a) Reward dynamics — mean and max per step. b) Validity and KL penalty vs frozen prior. c) Top generated molecules: QED vs predicted pKd scatter (best pKd=7.57). d) Oracle pKd distribution over all valid molecules scored (n=147 total, 18 top saved).*

**Summary (from `data/rl_results/rl_results.json`):**

| Metric | Value |
|--------|-------|
| Total generated | 147 |
| Top molecules saved | 18 |
| Best reward | 0.720 |
| Best predicted pKd | 7.57 |
| Prior epochs | 60 |
| RL steps | 150 × 64 batch |

### Oracle: frozen GNN as proxy scorer
Generated SMILES → ETKDG 3D conformer → centroid alignment to reference pocket → GNN predicts pKd. **Limitation:** ETKDG + rigid centroid alignment is a coarse approximation of docking. A production system would call AutoDock Vina or FEP+ for scoring. The GNN oracle is used here as a fast differentiable proxy consistent with the exercise scope.

---

## 4. SQL Server Schema Design

Five tables capture the full experimental lifecycle:

| Table | Purpose |
|---|---|
| `experiments` | Configuration registry (JSON blob) — one row per experimental condition |
| `model_runs` | Per-epoch metrics from PyTorch Lightning training |
| `binding_predictions` | On-demand predictions from Streamlit UI |
| `rl_molecules` | All generated molecules with per-component reward breakdown |
| `vina_benchmarks` | GNN vs Vina correlation on held-out set |

**Why SQL Server?** The exercise specification required it. For a pure research workflow, PostgreSQL or DuckDB would suffice. SQL Server 2022 Express handles the dataset sizes here without issue.

**MLflow backend:** MLflow stores run metadata in a dedicated `mlflowdb` database (separate from `gnnbind`) to avoid schema conflicts — MLflow auto-creates its own `experiments` table with a different primary-key convention than `dbo.experiments`. The tracking URI is `mssql+pyodbc://sa:<pw>@sqlserver/mlflowdb?driver=ODBC+Driver+18+for+SQL+Server`.

**Backfilling MLflow from existing checkpoints:** `scripts/log_existing_runs.py` reads checkpoint filenames (regex on `epoch=N-val_rmse=X`) and TensorBoard `lightning_logs/` events to reconstruct per-epoch convergence curves (`val_rmse`, `val_pearson_r`, `val_auc_pose`, `train_loss`) as MLflow step metrics. This enables convergence graphs in the MLflow UI without re-running training.

---

## 5. Docker Compose Service Topology

### Full pipeline


![Overview](Figs/Overview.png)

*Fig. Overview of the pipeline.*


### Dry-run (pre-trained checkpoints)

```bash
docker compose up -d sqlserver mlflow streamlit
python scripts/log_existing_runs.py
```

Skips `gnn-trainer` and `rl-agent`. MLflow is populated from existing checkpoints and TensorBoard logs via the backfill script. Streamlit reads `checkpoints/` and `data/rl_results/rl_results.json` from volume mounts.

### ARM64 / Apple Silicon notes

- ODBC driver: apt source uses `$(dpkg --print-architecture)` — resolves to `arm64` on Apple Silicon instead of hardcoded `amd64`
- PyG: only `torch-geometric==2.5.2` is installed in containers; `torch-scatter`, `torch-sparse`, `torch-cluster` are dropped (no ARM64 pre-built wheels, no C++ compiler in image). HGTConv inference works without them.
- Training runs on CPU (`MPS` not used — `scatter_reduce` unsupported on MPS backend).

---

## 6. Streamlit UI Design

Five pages, dark glassmorphism design (Space Grotesk + Instrument Serif, warm cream / deep teal palette):

| Page | Content |
|------|---------|
| **Summary** | KPI cards (RMSE, Pearson r, Pose AUC, best pKd), 3-phase project cards, MTL vs STL bar chart, RL reward curve (mean + max) |
| **Binding Predictor** | SMILES input → GNN inference → pKd, pose probability, selectivity metric cards + 2D structure |
| **RL Generator** | Dynamic pocket chip (⬡ Pocket 6E9A · pKd 11.92), card grid of top 18 molecules (structure SVG, pKd, QED, reward) |
| **GNN vs Vina** | Live GNN inference on RL top-20 SMILES → parity scatter + table vs Vina benchmark |
| **SQL Console** | Free-text SELECT → result table + CSV download |

Header badge is dynamic — reads `ref_pocket` and `ref_pkd` from `rl_results.json` at startup.

---

## 7. Limitations & Future Work

1. **Dataset size (150 samples):** PDBbind refined set filtered to 150 for speed. Production training would use 5000+ complexes with cross-validation across protein families. MTL benefit over STL is expected to be more pronounced at that scale.
2. **Oracle fidelity:** GNN oracle replaces proper docking. A production RL loop would call AutoDock Vina or FEP+ for scoring.
3. **SMILES generator validity:** Character-level LSTM achieves high validity on this run (100% reported); fragment-based generators (REINVENT 4, GraphINVENT) typically yield more structurally diverse output.
4. **Selectivity labels:** Binary EGFR/ABL1 selectivity head uses synthetic labels derived from known inhibitor annotations. Real selectivity data would require paired kinase assay results.
5. **No 3D equivariance:** HGTConv is not SE(3)-equivariant. Switching to DimeNet++ or EGNN would improve geometric fidelity for affinity prediction.
6. **MLflow convergence curves (dry-run):** `log_existing_runs.py` reconstructs curves from TensorBoard events; step alignment is approximate (per-batch train steps bucketed by epoch tag). Live training logs directly to MLflow with exact step correspondence.
