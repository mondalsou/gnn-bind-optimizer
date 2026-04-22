# GNNBindOptimizer — Architecture

## 1. Graph Construction

### Why a heterogeneous graph?
Protein-ligand binding involves two chemically distinct entity types (amino acid residues and small-molecule atoms) with fundamentally different feature spaces. A homogeneous GNN would require padding both to the same dimensionality, discarding the structural distinction. A heterogeneous graph (`torch_geometric.data.HeteroData`) preserves it natively and lets message-passing attend separately to intra-ligand, intra-pocket, and cross-interface edges.

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

**Choice: Heterogeneous Graph Transformer (HGT)**

HGT uses type-specific attention heads and relation-specific key-query-value projections, making it the natural fit when node and edge types carry different semantics. Alternative choices and trade-offs:

| Architecture | Pros | Cons |
|---|---|---|
| HGTConv (chosen) | Type-aware attention; handles asymmetric feature dims natively | Higher compute than homogeneous GCN |
| HANConv | Metapath-based, strong for few relation types | Requires manual metapath design |
| GATConv (homogeneous) | Simpler; proven on molecular tasks | Loses protein/ligand distinction |
| SchNet/DimeNet | Explicit 3D geometry | Requires high-quality docked poses; fragile to conformer errors |

**Architecture details:**
- 4 × `HGTConv` layers, hidden dim 128, 4 attention heads
- Residual connections after each layer (add + LayerNorm)
- Global readout: type-specific sum-pooling → concatenate → 2-layer MLP
- **3 prediction heads** (multi-task):

| Head | Task | Loss |
|------|------|------|
| `affinity_head` | scalar pKd (regression) | Smooth-L1 |
| `pose_head` | binary: RMSD < 2 Å (classification) | BCE |
| `selectivity_head` | binary: EGFR vs ABL1 (classification) | BCE |

### Why multi-task learning?

Pose quality and selectivity act as auxiliary signals that share representations with the affinity prediction path. In practice (see Phase 2 results), MTL val RMSE = **1.924** vs STL **2.034** (Δ = 0.11, +5.4% improvement). The improvement is modest on 150 samples but expected to grow with larger datasets — the auxiliary heads regularize the shared encoder.

**Uncertainty weighting (Kendall et al., 2018):** Each loss term is divided by a learned task variance σ² (log σ² is the actual parameter for numerical stability). This avoids manually tuning loss weights and lets the model adaptively balance tasks during training.

---

## 3. RL Molecular Generator

### Why REINFORCE over PPO/GRPO?
For small-molecule generation with sparse valid rewards (~1% validity from a random prior), REINFORCE with a moving EMA baseline is computationally lighter and easier to interpret. PPO would add a value-network head and clipping mechanics that provide marginal benefit at this scale. The key instability risk (mode collapse) is addressed via a KL-divergence penalty against the frozen prior.

### Policy: Character-level LSTM
- 2 layers, hidden dim 512, embedding dim 128
- Token vocabulary: single characters + two-char tokens (Cl, Br, @@, =O, etc.) → 37 tokens
- Pre-trained for 60 epochs on ~158 EGFR-relevant SMILES (corpus = training set ligands + 29 known EGFR inhibitors) using teacher forcing (cross-entropy loss → 0.156 final)

### Why character-level over fragment-based?
Character-level is simpler, produces more diverse output, and doesn't require a fragment library. The trade-off is lower initial validity (~1% before RL fine-tuning, improving over training). Fragment-based generators (REINVENT 4, GraphINVENT) achieve higher validity but require curated fragment vocabularies.

### Reward function
```
R = 0.5·r_affinity + 0.2·r_qed + 0.2·r_sa + 0.1·r_mw
```
| Component | Rationale |
|---|---|
| `r_affinity` | Normalized GNN pKd — primary optimization target |
| `r_qed` | Drug-likeness (Bickerton 2012) — composite Lipinski compliance |
| `r_sa` | 1 − SA_score/10 — synthetic accessibility (Ertl & Schuffenhauer) |
| `r_mw` | 1 if MW ≤ 500, otherwise 0 — hard Lipinski MW filter |

Affinity weight (0.5) deliberately dominates to bias toward high-affinity molecules; QED and SA prevent the model from converging on synthetically inaccessible structures.

### Oracle: frozen GNN as proxy scorer
Generated SMILES → ETKDG 3D conformer → centroid alignment to reference pocket (6E9A, pKd = 11.92, highest in training set) → GNN predicts pKd. **Limitation acknowledged:** ETKDG + rigid centroid alignment is a coarse approximation of docking. A production system would call AutoDock Vina or Glide for the oracle. The GNN oracle is used here as a fast differentiable proxy consistent with the exercise scope.

---

## 4. SQL Server Schema Design

Five tables capture the full experimental lifecycle:

| Table | Purpose |
|---|---|
| `experiments` | Configuration registry (JSON blob) — one row per experimental condition |
| `model_runs` | Per-epoch metrics from PyTorch Lightning training |
| `binding_predictions` | On-demand predictions from the Streamlit UI |
| `rl_molecules` | All generated molecules with per-component reward breakdown |
| `vina_benchmarks` | GNN vs Vina correlation on held-out set |

**Why SQL Server?** The exercise specification required it. For a pure research workflow, PostgreSQL or DuckDB would suffice. SQL Server 2022 Express handles the dataset sizes here without issue.

**MLflow backend:** `mssql+pyodbc` tracking URI lets MLflow store run metadata and parameters in the same SQL Server instance, centralizing all experiment data.

---

## 5. Docker Compose Service Topology

```
sqlserver ─────────────────────────────────────────────────┐
    │ (service_healthy)                                     │
    ▼                                                       │
mlflow          gnn-trainer (runs once, exits)             │
                    │ (service_completed_successfully)      │
                    ▼                                       │
               rl-agent (runs once, exits)                  │
                                                           │
streamlit ◄────────────────────────────────────────────────┘
  (reads SQL Server directly; depends on service_healthy)
```

`gnn-trainer` and `rl-agent` use `restart: "no"` — they are one-shot training jobs, not long-lived services. The dependency chain ensures RL only starts after the GNN checkpoint exists.

---

## 6. Limitations & Future Work

1. **Dataset size (150 samples):** PDBbind refined set filtered to 150 for speed. Production training would use 5000+ complexes with cross-validation across protein families.
2. **Oracle fidelity:** GNN oracle replaces proper docking. A production RL loop would call AutoDock Vina or FEP+ for scoring.
3. **SMILES generator validity:** Character-level LSTM achieves ~1% validity from scratch. Fragment-based or graph-based generators (e.g., JT-VAE, GraphINVENT) would yield >30% validity natively.
4. **Selectivity labels:** Binary EGFR/ABL1 selectivity head uses synthetic labels derived from known inhibitor annotations. Real selectivity data would require paired kinase assay results.
5. **No 3D equivariance:** HGTConv is not SE(3)-equivariant. Switching to DimeNet++ or EGNN would improve geometric fidelity for affinity prediction.
