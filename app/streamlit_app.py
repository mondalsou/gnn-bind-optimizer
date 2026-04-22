"""GNNBindOptimizer — Streamlit UI (5 pages)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

st.set_page_config(
    page_title="GNNBindOptimizer",
    page_icon="🧬",
    layout="wide",
)

# ── DB helper (graceful fallback when SQL Server is unreachable) ─────────────
def _db_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    try:
        from src.db.connection import query_df
        return query_df(sql, params)
    except Exception as e:
        st.warning(f"DB unavailable — showing demo data. ({e})")
        return pd.DataFrame()

# ── GNN inference helper ─────────────────────────────────────────────────────
@st.cache_resource
def _load_gnn():
    import torch
    from src.models.gnn_bind import HeteroGNN
    ckpt_path = Path("checkpoints") / "gnn_state.pt"
    if not ckpt_path.exists():
        ckpt_path = Path("src/models/gnn_state.pt")
    if not ckpt_path.exists():
        return None, None
    bundle = torch.load(str(ckpt_path), map_location="cpu")
    hp = bundle["hparams"]
    model = HeteroGNN(
        ligand_node_dim=hp["ligand_node_dim"],
        protein_node_dim=hp["protein_node_dim"],
        hidden_dim=hp["hidden_dim"],
        num_layers=hp["num_layers"],
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle

@st.cache_data(show_spinner=False)
def _load_rl_results() -> pd.DataFrame:
    path = Path("data/rl_results/rl_results.json")
    if not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text())
    mols = data.get("top_molecules", data.get("molecules", []))
    if not mols:
        return pd.DataFrame()
    df = pd.DataFrame(mols)
    # normalise column names (reward_total → reward)
    if "reward_total" in df.columns and "reward" not in df.columns:
        df = df.rename(columns={"reward_total": "reward"})
    return df

# ── Sidebar navigation ────────────────────────────────────────────────────────
PAGES = [
    "Dashboard",
    "Binding Predictor",
    "RL Browser",
    "GNN vs Vina",
    "SQL Console",
]
page = st.sidebar.radio("Navigate", PAGES)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Dashboard
# ═════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("GNNBindOptimizer — Dashboard")
    st.caption("Experiment tracking + training metrics")

    # Experiment table
    st.subheader("Experiments")
    exps = _db_query("SELECT id, name, created_at, config_json FROM dbo.experiments ORDER BY id")
    if exps.empty:
        # fallback demo
        exps = pd.DataFrame([
            {"id": 1, "name": "gnn_mtl_baseline",  "config_json": '{"model":"HeteroGNN","heads":3}'},
            {"id": 2, "name": "gnn_stl_ablation",  "config_json": '{"model":"HeteroGNN","heads":1}'},
            {"id": 3, "name": "rl_reinforce_egfr", "config_json": '{"policy":"LSTM","steps":300}'},
        ])
    st.dataframe(exps, use_container_width=True)

    # Training metrics
    st.subheader("Model Runs")
    runs = _db_query("""
        SELECT e.name AS experiment, mr.model_type, mr.epoch,
               mr.val_rmse, mr.val_pearson_r, mr.val_auc_pose
        FROM dbo.model_runs mr
        JOIN dbo.experiments e ON e.id = mr.experiment_id
        ORDER BY mr.val_rmse ASC
    """)
    if runs.empty:
        runs = pd.DataFrame([
            {"experiment": "gnn_mtl_baseline", "model_type": "MTL", "epoch": 19,
             "val_rmse": 1.924, "val_pearson_r": 0.541, "val_auc_pose": 0.778},
            {"experiment": "gnn_stl_ablation", "model_type": "STL", "epoch": 9,
             "val_rmse": 2.034, "val_pearson_r": 0.489, "val_auc_pose": None},
        ])

    col1, col2, col3 = st.columns(3)
    if not runs.empty:
        best = runs.iloc[0]
        col1.metric("Best val RMSE", f"{best['val_rmse']:.3f}")
        col2.metric("Best Pearson r", f"{best['val_pearson_r']:.3f}" if pd.notna(best['val_pearson_r']) else "—")
        col3.metric("Best Pose AUC", f"{best['val_auc_pose']:.3f}" if pd.notna(best['val_auc_pose']) else "—")

    st.dataframe(runs, use_container_width=True)

    # MTL vs STL bar
    st.subheader("MTL vs STL — val RMSE")
    import altair as alt
    ablation = runs[runs["val_rmse"].notna()][["experiment", "model_type", "val_rmse"]].drop_duplicates()
    if not ablation.empty:
        chart = alt.Chart(ablation).mark_bar().encode(
            x=alt.X("experiment:N", title="Experiment"),
            y=alt.Y("val_rmse:Q", title="val RMSE", scale=alt.Scale(zero=False)),
            color=alt.Color("model_type:N"),
            tooltip=["experiment", "model_type", "val_rmse"],
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Binding Predictor
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Binding Predictor":
    st.title("Binding Affinity Predictor")
    st.caption("Enter a SMILES string — GNN predicts pKd, pose quality, and EGFR selectivity")

    smiles_input = st.text_input(
        "SMILES",
        value="CCc1nn(C)c2cnc(Nc3cc(NC(=O)c4ccccc4)ccc3OCC)nc12",
        help="Erlotinib scaffold by default",
    )

    predict_btn = st.button("Predict")
    if predict_btn and smiles_input:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            st.error("Invalid SMILES — RDKit could not parse.")
        else:
            model, bundle = _load_gnn()
            if model is None:
                st.warning("GNN checkpoint not found — showing placeholder prediction.")
                pred_pkd, pred_pose, pred_select = 7.4, 0.72, 0.61
            else:
                import torch
                from src.graph.build_graph import smiles_to_hetero_graph_simple
                ref_path = Path("data/processed/dataset.pt")
                if ref_path.exists():
                    dataset = torch.load(str(ref_path), map_location="cpu")
                    ref_graph = dataset[0]
                    ref_centroid = ref_graph["ligand"].pos.mean(0).numpy()
                else:
                    ref_graph, ref_centroid = None, np.zeros(3)

                try:
                    g = smiles_to_hetero_graph_simple(smiles_input, ref_graph, ref_centroid)
                    with torch.no_grad():
                        out = model(g.x_dict, g.edge_index_dict, g.edge_attr_dict)
                    pred_pkd    = float(out["affinity"].item())
                    pred_pose   = float(torch.sigmoid(out["pose"]).item())
                    pred_select = float(torch.sigmoid(out["selectivity"]).item())
                except Exception as e:
                    st.warning(f"Inference failed: {e}. Showing placeholder.")
                    pred_pkd, pred_pose, pred_select = 7.4, 0.72, 0.61

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted pKd", f"{pred_pkd:.2f}")
            col2.metric("Pose Quality (prob RMSD<2Å)", f"{pred_pose:.2f}")
            col3.metric("EGFR Selectivity Prob", f"{pred_select:.2f}")

            # 2D structure
            from rdkit.Chem import Draw
            import io
            img = Draw.MolToImage(mol, size=(300, 200))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.image(buf.getvalue(), caption=smiles_input)

            # Log to DB
            _db_query("""
                INSERT INTO dbo.binding_predictions
                    (run_id, smiles, pocket_pdb, pred_pkd, pred_pose_prob, pred_select_prob)
                VALUES (1, :smiles, '1IEP', :pkd, :pose, :sel)
            """, {"smiles": smiles_input, "pkd": pred_pkd, "pose": pred_pose, "sel": pred_select})

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RL Browser
# ═════════════════════════════════════════════════════════════════════════════
elif page == "RL Browser":
    st.title("RL-Generated Molecules")
    st.caption("REINFORCE + frozen GNN oracle | reward = 0.5·affinity + 0.2·QED + 0.2·SA + 0.1·MW")

    # Load from DB or local JSON fallback
    df = _db_query("""
        SELECT smiles, reward, r_affinity, r_qed, r_sa, r_mw, pred_pkd, step
        FROM dbo.rl_molecules
        WHERE experiment_id = (
            SELECT id FROM dbo.experiments WHERE name = 'rl_reinforce_egfr'
        )
        ORDER BY reward DESC
    """)
    if df.empty:
        df = _load_rl_results()

    if df.empty:
        st.info("No RL results found. Run Phase 3 notebook first.")
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total mols", len(df))
        col2.metric("Best reward", f"{df['reward'].max():.3f}")
        col3.metric("Best pKd", f"{df['pred_pkd'].max():.2f}" if 'pred_pkd' in df else "—")
        col4.metric("Mean QED", f"{df['r_qed'].mean():.3f}" if 'r_qed' in df else "—")

        # Reward scatter
        import altair as alt
        st.subheader("Reward vs Predicted pKd")
        if 'pred_pkd' in df and 'reward' in df:
            scatter = alt.Chart(df.reset_index()).mark_circle(size=60, opacity=0.7).encode(
                x=alt.X("pred_pkd:Q", title="Predicted pKd"),
                y=alt.Y("reward:Q", title="Total Reward"),
                color=alt.Color("r_qed:Q", title="QED", scale=alt.Scale(scheme="viridis")),
                tooltip=["smiles", "reward", "pred_pkd", "r_qed", "r_sa"],
            ).properties(height=350)
            st.altair_chart(scatter, use_container_width=True)

        # Top molecules table with structures
        st.subheader("Top 10 Molecules")
        top10 = df.nlargest(10, "reward")[["smiles","reward","pred_pkd","r_qed","r_sa","r_mw"]]
        st.dataframe(top10, use_container_width=True)

        # Draw structures for top 5
        st.subheader("Structures — Top 5")
        from rdkit import Chem
        from rdkit.Chem import Draw
        import io
        cols = st.columns(5)
        for i, (_, row) in enumerate(top10.head(5).iterrows()):
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol:
                img = Draw.MolToImage(mol, size=(180, 140))
                buf = io.BytesIO(); img.save(buf, format="PNG")
                cols[i].image(buf.getvalue(), caption=f"R={row['reward']:.3f}")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — GNN vs Vina Scatter
# ═════════════════════════════════════════════════════════════════════════════
elif page == "GNN vs Vina":
    st.title("GNN vs AutoDock Vina — Benchmark Parity")
    st.caption("Correlation between GNN predicted pKd and Vina docking scores on held-out set")

    df = _db_query("""
        SELECT smiles, pocket_pdb, vina_score, gnn_pred_pkd,
               ABS(vina_score - gnn_pred_pkd) AS abs_error
        FROM dbo.vina_benchmarks
        ORDER BY abs_error DESC
    """)

    if df.empty:
        # Demo data for visualization when DB unavailable or empty
        rng = np.random.RandomState(42)
        true_pkd = rng.uniform(4, 10, 30)
        df = pd.DataFrame({
            "smiles": [f"C{'c'*i}O" for i in range(30)],
            "pocket_pdb": ["1IEP"] * 30,
            "vina_score": true_pkd + rng.normal(0, 0.8, 30),
            "gnn_pred_pkd": true_pkd + rng.normal(0, 1.2, 30),
        })
        df["abs_error"] = (df["vina_score"] - df["gnn_pred_pkd"]).abs()
        st.info("Showing synthetic demo data — populate dbo.vina_benchmarks to see real results.")

    import altair as alt
    from scipy.stats import pearsonr

    r, _ = pearsonr(df["vina_score"], df["gnn_pred_pkd"])
    rmse = np.sqrt(((df["vina_score"] - df["gnn_pred_pkd"])**2).mean())

    col1, col2 = st.columns(2)
    col1.metric("Pearson r (GNN vs Vina)", f"{r:.3f}")
    col2.metric("RMSE", f"{rmse:.3f}")

    # Parity plot
    lim_min = min(df["vina_score"].min(), df["gnn_pred_pkd"].min()) - 0.5
    lim_max = max(df["vina_score"].max(), df["gnn_pred_pkd"].max()) + 0.5

    scatter = alt.Chart(df).mark_circle(size=80, opacity=0.7).encode(
        x=alt.X("vina_score:Q", title="Vina Score (pKd proxy)", scale=alt.Scale(domain=[lim_min, lim_max])),
        y=alt.Y("gnn_pred_pkd:Q", title="GNN Predicted pKd", scale=alt.Scale(domain=[lim_min, lim_max])),
        color=alt.Color("abs_error:Q", title="|Error|", scale=alt.Scale(scheme="reds")),
        tooltip=["pocket_pdb", "vina_score", "gnn_pred_pkd", "abs_error"],
    ).properties(height=450)

    diagonal = alt.Chart(
        pd.DataFrame({"x": [lim_min, lim_max], "y": [lim_min, lim_max]})
    ).mark_line(color="gray", strokeDash=[4,4]).encode(x="x:Q", y="y:Q")

    st.altair_chart(scatter + diagonal, use_container_width=True)

    st.subheader("Data table")
    st.dataframe(df.sort_values("abs_error", ascending=False), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SQL Console
# ═════════════════════════════════════════════════════════════════════════════
elif page == "SQL Console":
    st.title("SQL Console")
    st.caption("Raw query against gnnbind database — SELECT only, results rendered as table")

    default_query = """SELECT TOP 20
    e.name AS experiment,
    rl.smiles,
    rl.reward,
    rl.pred_pkd,
    rl.r_qed,
    rl.r_sa
FROM dbo.rl_molecules rl
JOIN dbo.experiments e ON e.id = rl.experiment_id
ORDER BY rl.reward DESC"""

    sql = st.text_area("SQL Query", value=default_query, height=200)
    run_btn = st.button("Run Query")

    if run_btn:
        if sql.strip().upper().startswith("SELECT"):
            with st.spinner("Querying..."):
                result = _db_query(sql)
            if result.empty:
                st.info("Query returned 0 rows.")
            else:
                st.success(f"{len(result)} rows returned")
                st.dataframe(result, use_container_width=True)
                csv = result.to_csv(index=False)
                st.download_button("Download CSV", csv, "query_result.csv", "text/csv")
        else:
            st.error("Only SELECT queries are allowed in this console.")
