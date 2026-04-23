"""GNNBindOptimizer — Streamlit UI"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re, io, json
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="GNNBindOptimizer",
    page_icon="🧬",
    layout="wide",
)

# ── Design tokens (from AIDesigner run cbe8ebe5) ──────────────────────────────
# Palette: warm cream (#f7f0e8) bg, deep teal sidebar (#193a3b), glassmorphism cards
# Typography: Space Grotesk + Instrument Serif italic accents
# Molecules: RDKit SVG inline, 420×300, custom element palette, MCS diff highlight
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@1&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', -apple-system, sans-serif !important;
    background: #f7f0e8 !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #193a3b 0%, #224c4d 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #edf7f5 !important; }
[data-testid="stSidebar"] .stRadio label { color: #edf7f5 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: rgba(237,247,245,0.5) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Main canvas ─────────────────────────────────────────────────────────── */
.stApp, .main .block-container {
    background: #f7f0e8 !important;
    max-width: 1400px !important;
    padding: 1.5rem 2rem !important;
}

/* ── Header ──────────────────────────────────────────────────────────────── */
.gnn-header {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.6);
    border-radius: 28px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 40px rgba(25,58,59,0.06);
}
.gnn-logo {
    font-size: 1.5rem;
    font-weight: 700;
    color: #193a3b;
    letter-spacing: -0.03em;
}
.gnn-logo em {
    font-family: 'Instrument Serif', Georgia, serif;
    font-style: italic;
    font-weight: 400;
    color: #347a7b;
}
.gnn-tagline { font-size: 0.76rem; color: #5f6b76; margin-top: 3px; }
.gnn-badge {
    background: #e0f0f0;
    color: #0f5f5e;
    border-radius: 999px;
    padding: 0.4rem 1rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.gnn-badge::before {
    content: '';
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #0f766e;
    display: inline-block;
}

/* ── Glassmorphism card ───────────────────────────────────────────────────── */
.glass {
    background: rgba(255,255,255,0.84);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.6);
    border-radius: 24px;
    box-shadow: 0 4px 40px rgba(25,58,59,0.06), 0 1px 8px rgba(25,58,59,0.03);
}

/* ── Metric card ─────────────────────────────────────────────────────────── */
.metric-card {
    background: rgba(255,255,255,0.84);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.6);
    border-radius: 24px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 4px 40px rgba(25,58,59,0.06);
    margin-bottom: 1rem;
}
.metric-card.teal  { border-left: 4px solid #0f766e; }
.metric-card.amber { border-left: 4px solid #d97706; }
.metric-card.blue  { border-left: 4px solid #2563eb; }
.metric-card.coral { border-left: 4px solid #d97757; }
.metric-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #5f6b76;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 2.1rem;
    font-weight: 700;
    color: #19212a;
    letter-spacing: -0.04em;
    line-height: 1.1;
}
.metric-sub { font-size: 0.72rem; color: #9aa4ae; margin-top: 3px; }
.metric-delta-good { display:inline-block; background:#dcfce7; color:#15803d; border-radius:6px; padding:2px 7px; font-size:0.7rem; font-weight:700; margin-left:6px; }
.metric-delta-bad  { display:inline-block; background:#fee2e2; color:#b91c1c; border-radius:6px; padding:2px 7px; font-size:0.7rem; font-weight:700; margin-left:6px; }

/* ── Molecule panel ──────────────────────────────────────────────────────── */
.mol-panel {
    background: linear-gradient(135deg, #fbfcfe 0%, #eef3f8 100%);
    border: 1px solid rgba(71,85,105,0.14);
    border-radius: 20px;
    padding: 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 280px;
    overflow: hidden;
}
.mol-panel svg { width: 100%; height: auto; max-height: 280px; }
.mol-fallback {
    font-family: 'SFMono-Regular', ui-monospace, monospace;
    font-size: 0.78rem;
    color: #5f6b76;
    background: #f0f4f8;
    border: 1px dashed #c7d2dc;
    border-radius: 12px;
    padding: 1rem;
    word-break: break-all;
}
.change-note {
    font-size: 0.72rem;
    color: #b7791f;
    background: rgba(183,121,31,0.1);
    border-radius: 999px;
    padding: 3px 10px;
    display: inline-block;
    margin-top: 6px;
}

/* ── Score bar ───────────────────────────────────────────────────────────── */
.score-row { display:flex; align-items:center; gap:10px; margin:0.6rem 0; }
.score-name { font-size:0.82rem; color:#5f6b76; width:160px; flex-shrink:0; font-weight:500; }
.score-bar-bg { flex:1; background:rgba(25,58,59,0.07); border-radius:999px; height:8px; overflow:hidden; }
.score-bar    { height:100%; border-radius:999px; }
.score-val    { font-size:0.8rem; font-weight:700; font-family:ui-monospace,monospace; width:46px; text-align:right; }
.score-delta  { font-size:0.7rem; font-weight:700; width:42px; }
.delta-pos    { color:#15803d; }
.delta-neg    { color:#b42318; }

/* ── Alert pill ──────────────────────────────────────────────────────────── */
.pill {
    display:inline-block; border-radius:999px;
    padding:0.3rem 0.75rem; font-size:0.72rem; font-weight:700;
}
.pill-green  { background:rgba(21,128,61,0.10); color:#15803d; }
.pill-yellow { background:rgba(183,121,31,0.12); color:#b7791f; }
.pill-red    { background:rgba(180,35,24,0.10); color:#b42318; }
.pill-teal   { background:#e0f0f0; color:#0f5f5e; }

/* ── Section label ───────────────────────────────────────────────────────── */
.section-label {
    font-size:0.68rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:#9aa4ae; margin-bottom:0.6rem;
}
.section-title {
    font-size:1.1rem; font-weight:700; color:#19212a; margin-bottom:0.2rem;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background: #193a3b !important;
    color: white !important;
    border: none !important;
    border-radius: 999px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    padding: 0.55rem 1.5rem !important;
    box-shadow: 0 4px 14px rgba(25,58,59,0.18) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #224c4d !important;
    box-shadow: 0 6px 20px rgba(25,58,59,0.25) !important;
}

/* ── Inputs ──────────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: rgba(255,255,255,0.9) !important;
    border: 1px solid rgba(25,33,42,0.12) !important;
    border-radius: 14px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: #19212a !important;
    box-shadow: 0 2px 8px rgba(25,58,59,0.04) !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.7) !important;
    border-radius: 999px !important;
    padding: 4px !important;
    border: 1px solid rgba(25,33,42,0.08) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    color: #5f6b76 !important;
    padding: 0.6rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #19212a !important;
    box-shadow: 0 2px 8px rgba(25,33,42,0.08) !important;
}

/* ── DataFrame ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid rgba(25,33,42,0.07) !important;
}

/* ── Altair charts ───────────────────────────────────────────────────────── */
.vega-embed { border-radius: 16px !important; }

/* ── Download button ─────────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: transparent !important;
    color: #0f766e !important;
    border: 1px solid rgba(15,118,110,0.3) !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gnn-header">
  <div>
    <div class="gnn-logo">GNN<em>Bind</em>Optimizer</div>
    <div class="gnn-tagline">Structure-aware binding affinity prediction + RL molecular generation</div>
  </div>
  <div class="gnn-badge">Model: HeteroGNN &nbsp;·&nbsp; Oracle: Frozen &nbsp;·&nbsp; Pocket: 6E9A &nbsp;·&nbsp; pKd 11.92</div>
</div>
""", unsafe_allow_html=True)

# ── DB helpers ────────────────────────────────────────────────────────────────
def _db_configured() -> bool:
    """True only when DB_SERVER is explicitly set — i.e. running inside Docker stack."""
    return os.environ.get("DB_SERVER", "").strip() not in ("", "localhost")

def _db_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    if not _db_configured():
        return pd.DataFrame()   # silently use demo data when running locally
    try:
        from src.db.connection import query_df
        return query_df(sql, params)
    except Exception as e:
        st.warning(f"DB error — demo data shown. ({e})")
        return pd.DataFrame()

def _db_execute(sql: str, params: dict | None = None) -> None:
    if not _db_configured():
        return
    try:
        from src.db.connection import get_engine
        from sqlalchemy import text
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text(sql), params or {})
    except Exception:
        pass

# ── Molecule SVG renderer ─────────────────────────────────────────────────────
def mol_svg(smiles: str, prev_smiles: str | None = None,
            w: int = 420, h: int = 300) -> tuple[str | None, bool]:
    """
    Inline SVG via rdMolDraw2D with:
    - Custom element palette matching lead_optimization_agent reference
    - Optional MCS-based yellow highlight of changed region vs prev_smiles
    Returns (svg_string, has_highlight).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFMCS, rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, False

        highlight_atoms, highlight_bonds = [], []
        highlight_atom_colors, highlight_bond_colors = {}, {}

        if prev_smiles:
            prev = Chem.MolFromSmiles(prev_smiles)
            if prev is not None:
                try:
                    mcs = rdFMCS.FindMCS(
                        [prev, mol],
                        atomCompare=rdFMCS.AtomCompare.CompareElements,
                        bondCompare=rdFMCS.BondCompare.CompareOrder,
                        timeout=3,
                    )
                    if mcs and mcs.smartsString:
                        patt = Chem.MolFromSmarts(mcs.smartsString)
                        match = mol.GetSubstructMatch(patt)
                        if match:
                            core = set(match)
                            highlight_atoms = [a.GetIdx() for a in mol.GetAtoms()
                                               if a.GetIdx() not in core]
                            highlight_atom_colors = {i: (0.98, 0.83, 0.18) for i in highlight_atoms}
                except Exception:
                    pass

        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
        opts   = drawer.drawOptions()
        opts.bondLineWidth          = 2.2
        opts.padding                = 0.06
        opts.fixedBondLength        = 40
        opts.clearBackground        = False
        opts.highlightRadius        = 0.32
        opts.highlightBondWidthMultiplier = 16
        opts.atomHighlightsAreCircles     = True
        # Custom element palette — matches reference app
        opts.updateAtomPalette({
            6:  (0.10, 0.13, 0.16),   # C  dark
            7:  (0.09, 0.35, 0.82),   # N  blue
            8:  (0.84, 0.15, 0.16),   # O  red
            9:  (0.00, 0.60, 0.69),   # F  cyan
            15: (0.58, 0.20, 0.78),   # P  purple
            16: (0.84, 0.45, 0.08),   # S  orange
            17: (0.09, 0.53, 0.35),   # Cl green
            35: (0.60, 0.20, 0.10),   # Br brown-red
        })

        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer, mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_atom_colors,
            highlightBonds=highlight_bonds,
            highlightBondColors=highlight_bond_colors,
        )
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg = re.sub(r"<\?xml[^>]*\?>", "", svg).strip()
        # transparent background
        svg = svg.replace("opacity:1.0;fill:#ffffff", "opacity:0;fill:#ffffff", 1)
        return svg, bool(highlight_atoms)
    except Exception:
        return None, False


def mol_svg_html(smiles: str, prev_smiles: str | None = None) -> str:
    svg, highlighted = mol_svg(smiles, prev_smiles)
    if svg:
        note = ('<div class="change-note">🟡 Yellow = modified region vs previous</div>'
                if highlighted else "")
        return f'<div class="mol-panel">{svg}</div>{note}'
    return f'<div class="mol-panel"><div class="mol-fallback">{smiles}</div></div>'


def score_bar_html(label, value, display, pct, color, delta=""):
    delta_html = ""
    if delta:
        cls = "delta-pos" if delta.startswith("+") else "delta-neg"
        delta_html = f'<span class="score-delta {cls}">{delta}</span>'
    return f"""
<div class="score-row">
  <span class="score-name">{label}</span>
  <div class="score-bar-bg"><div class="score-bar" style="width:{pct:.0f}%;background:{color}"></div></div>
  <span class="score-val" style="color:{color}">{display}</span>
  {delta_html}
</div>"""


# ── GNN oracle ────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_gnn():
    import torch, torch.nn as nn
    from torch_geometric.nn import HGTConv
    from torch_geometric.nn.aggr import AttentionalAggregation

    ckpt = Path(__file__).parent.parent / "src" / "models" / "gnn_state.pt"
    if not ckpt.exists():
        return None, None

    saved = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    hp    = saved["hparams"]

    NODE_TYPES = ["ligand", "residue"]
    EDGE_TYPES = [("ligand","bond","ligand"),("residue","contact","residue"),
                  ("residue","interacts","ligand"),("ligand","interacts","residue")]
    META = (NODE_TYPES, EDGE_TYPES)

    class HeteroGNN(nn.Module):
        def __init__(self, lig_in, prot_in, hidden, n_layers, n_heads, dropout, multitask):
            super().__init__()
            self.multitask = multitask; self.n_layers = n_layers
            self.lig_proj  = nn.Sequential(nn.Linear(lig_in,hidden),nn.LayerNorm(hidden),nn.ReLU())
            self.prot_proj = nn.Sequential(nn.Linear(prot_in,hidden),nn.LayerNorm(hidden),nn.ReLU())
            self.convs = nn.ModuleList([
                HGTConv({nt:hidden for nt in NODE_TYPES},hidden,META,heads=n_heads)
                for _ in range(n_layers)])
            self.norms = nn.ModuleList([
                nn.ModuleDict({nt:nn.LayerNorm(hidden) for nt in NODE_TYPES})
                for _ in range(n_layers)])
            self.dropout   = nn.Dropout(dropout)
            self.lig_pool  = AttentionalAggregation(gate_nn=nn.Linear(hidden,1))
            self.prot_pool = AttentionalAggregation(gate_nn=nn.Linear(hidden,1))
            self.trunk = nn.Sequential(
                nn.Linear(hidden*2,hidden),nn.LayerNorm(hidden),nn.ReLU(),
                nn.Dropout(dropout),nn.Linear(hidden,hidden//2),nn.ReLU())
            hi = hidden//2
            self.affinity_head    = nn.Linear(hi,1)
            self.pose_head        = nn.Linear(hi,1)
            self.selectivity_head = nn.Linear(hi,1)

        def forward(self, batch):
            import torch
            x = {"ligand":self.lig_proj(batch["ligand"].x),
                 "residue":self.prot_proj(batch["residue"].x)}
            ei = {}
            for et in EDGE_TYPES:
                try: ei[et] = batch[et].edge_index
                except: pass
            for conv, nd in zip(self.convs, self.norms):
                xn = conv(x, ei)
                for nt in NODE_TYPES:
                    if nt in xn and xn[nt] is not None:
                        x[nt] = nd[nt](self.dropout(xn[nt]) + x[nt])
            le = self.lig_pool(x["ligand"],  batch["ligand"].batch)
            pe = self.prot_pool(x["residue"], batch["residue"].batch)
            g  = self.trunk(torch.cat([le,pe],dim=-1))
            out = {"affinity": self.affinity_head(g).squeeze(-1)}
            if self.multitask:
                out["pose"]        = self.pose_head(g).squeeze(-1)
                out["selectivity"] = self.selectivity_head(g).squeeze(-1)
            return out

    model = HeteroGNN(**hp)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, saved


@st.cache_resource
def _load_all_pockets() -> dict:
    """Returns {pdb_id: graph} for all 150 dataset pockets, sorted by pKd desc."""
    import torch
    p = Path(__file__).parent.parent / "data" / "processed" / "dataset.pt"
    if not p.exists():
        return {}
    graphs = torch.load(str(p), map_location="cpu", weights_only=False)
    pocket_map = {}
    for g in graphs:
        pdb = g.pdb_id.replace("_pocket", "").lower()
        pocket_map[pdb] = g
    return pocket_map


def _pocket_options() -> list[tuple[str, str, float]]:
    """Returns [(label, pdb_id, pkd), ...] sorted by pKd desc."""
    pockets = _load_all_pockets()
    entries = []
    for pdb, g in pockets.items():
        pkd = round(g.y_affinity.item(), 2)
        entries.append((f"{pdb.upper()}  —  pKd {pkd:.2f}", pdb, pkd))
    return sorted(entries, key=lambda x: -x[2])


def _build_graph(smiles, pocket_id: str = "6e9a"):
    import torch, numpy as np
    from rdkit import Chem; from rdkit.Chem import AllChem, rdchem
    from torch_geometric.data import HeteroData; from scipy.spatial.distance import cdist

    pockets = _load_all_pockets()
    ref = pockets.get(pocket_id.lower())
    if ref is None:
        # fallback to highest-pKd pocket
        ref = max(pockets.values(), key=lambda g: g.y_affinity.item()) if pockets else None
    if ref is None:
        return None
    centroid = ref["ligand"].pos.numpy().mean(axis=0)

    AT  = ["H","C","N","O","F","P","S","Cl","Br","I"]
    DEG = [0,1,2,3,4,5]
    HYB = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
           rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
           rdchem.HybridizationType.SP3D2]
    BT  = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
           rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC]
    HD, HA = {"N","O"}, {"N","O","F"}
    oh = lambda v,c,uk=True: [int(v==x) for x in c]+([int(v not in c)] if uk else [])

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms()<2: return None
    try:
        mh = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mh, AllChem.ETKDGv3())!=0: return None
        AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
        mol = Chem.RemoveHs(mh)
    except: return None
    if mol.GetNumConformers()==0: return None

    cf = mol.GetConformer()
    pos= np.array([cf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())],dtype=np.float32)
    pos= pos - pos.mean(0) + centroid
    xl = torch.tensor([oh(a.GetSymbol(),AT)+oh(a.GetDegree(),DEG)+oh(a.GetHybridization(),HYB)+
                       [a.GetFormalCharge(),int(a.IsInRing()),int(a.GetIsAromatic()),a.GetTotalNumHs()]
                       for a in mol.GetAtoms()], dtype=torch.float)
    ls,ld,le=[],[],[]
    for b in mol.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        ef=oh(b.GetBondType(),BT,uk=False)+[int(b.IsInRing()),int(b.GetIsAromatic())]
        d=float(np.linalg.norm(pos[i]-pos[j]))
        ls+=[i,j]; ld+=[j,i]; le+=[ef+[d],ef+[d]]
    ca=ref["residue"].pos.numpy(); D=cdist(ca,pos)
    ps,pd2,pe,lss,ld2,le2=[],[],[],[],[],[]
    for p2,l in zip(*np.where(D<=5.0)):
        d=float(D[p2,l]); la=mol.GetAtomWithIdx(int(l))
        ef=[d,int(la.GetSymbol() in HD),int(la.GetSymbol() in HA)]
        ps.append(int(p2)); pd2.append(int(l)); pe.append(ef)
        lss.append(int(l)); ld2.append(int(p2)); le2.append(ef)
    ei=lambda s,d:(torch.tensor([s,d],dtype=torch.long) if s else torch.zeros((2,0),dtype=torch.long))
    g=HeteroData()
    g["ligand"].x=xl; g["ligand"].pos=torch.tensor(pos,dtype=torch.float)
    g["ligand"].batch=torch.zeros(len(xl),dtype=torch.long)
    g["residue"].x=ref["residue"].x.clone(); g["residue"].pos=ref["residue"].pos.clone()
    g["residue"].batch=torch.zeros(len(ref["residue"].x),dtype=torch.long)
    g["ligand","bond","ligand"].edge_index=ei(ls,ld)
    g["ligand","bond","ligand"].edge_attr=(torch.tensor(le,dtype=torch.float) if le else torch.zeros((0,7)))
    g["residue","contact","residue"].edge_index=ref["residue","contact","residue"].edge_index.clone()
    g["residue","contact","residue"].edge_attr=ref["residue","contact","residue"].edge_attr.clone()
    g["residue","interacts","ligand"].edge_index=ei(ps,pd2)
    g["residue","interacts","ligand"].edge_attr=(torch.tensor(pe,dtype=torch.float) if pe else torch.zeros((0,3)))
    g["ligand","interacts","residue"].edge_index=ei(lss,ld2)
    g["ligand","interacts","residue"].edge_attr=(torch.tensor(le2,dtype=torch.float) if le2 else torch.zeros((0,3)))
    return g


@st.cache_data(show_spinner=False)
def _rl_df() -> pd.DataFrame:
    p = Path(__file__).parent.parent / "data" / "rl_results" / "rl_results.json"
    if not p.exists(): return pd.DataFrame()
    d = json.loads(p.read_text())
    mols = d.get("top_molecules", [])
    if not mols: return pd.DataFrame()
    df = pd.DataFrame(mols)
    if "reward_total" in df and "reward" not in df:
        df = df.rename(columns={"reward_total":"reward"})
    return df


# ── Sidebar nav ───────────────────────────────────────────────────────────────
st.sidebar.markdown('<p>Navigation</p>', unsafe_allow_html=True)
PAGES = ["Dashboard", "Binding Predictor", "RL Browser", "GNN vs Vina", "SQL Console"]
page  = st.sidebar.radio("", PAGES, label_visibility="collapsed")

# DB status pill — no warning, just a quiet indicator
if _db_configured():
    st.sidebar.markdown('<p style="margin-top:2rem">Database</p>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<span style="background:rgba(21,128,61,0.2);color:#86efac;border-radius:999px;'
        'padding:3px 10px;font-size:0.7rem;font-weight:700">● SQL Server connected</span>',
        unsafe_allow_html=True)
else:
    st.sidebar.markdown('<p style="margin-top:2rem">Database</p>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<span style="background:rgba(255,255,255,0.08);color:rgba(237,247,245,0.4);'
        'border-radius:999px;padding:3px 10px;font-size:0.7rem;font-weight:600">'
        '○ Local mode · demo data</span>',
        unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Dashboard
# ═════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    exps = _db_query("SELECT id, name, created_at, config_json FROM dbo.experiments ORDER BY id")
    if exps.empty:
        exps = pd.DataFrame([
            {"id":1,"name":"gnn_mtl_baseline",  "config_json":'{"model":"HeteroGNN","heads":3}'},
            {"id":2,"name":"gnn_stl_ablation",  "config_json":'{"model":"HeteroGNN","heads":1}'},
            {"id":3,"name":"rl_reinforce_egfr", "config_json":'{"policy":"LSTM","steps":300}'},
        ])
    runs = _db_query("""
        SELECT e.name AS experiment, mr.model_type, mr.epoch,
               mr.val_rmse, mr.val_pearson_r, mr.val_auc_pose
        FROM dbo.model_runs mr
        JOIN dbo.experiments e ON e.id = mr.experiment_id
        ORDER BY mr.val_rmse ASC
    """)
    if runs.empty:
        runs = pd.DataFrame([
            {"experiment":"gnn_mtl_baseline","model_type":"MTL","epoch":19,
             "val_rmse":1.924,"val_pearson_r":0.541,"val_auc_pose":0.778},
            {"experiment":"gnn_stl_ablation","model_type":"STL","epoch":9,
             "val_rmse":2.034,"val_pearson_r":0.489,"val_auc_pose":None},
        ])

    best = runs.iloc[0] if not runs.empty else {}
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-card teal">
        <div class="metric-label">Best val RMSE</div>
        <div class="metric-value">{best.get('val_rmse',0):.3f}</div>
        <div class="metric-sub">{best.get('experiment','—')}</div>
    </div>""", unsafe_allow_html=True)
    pr = f"{best['val_pearson_r']:.3f}" if pd.notna(best.get('val_pearson_r')) else "—"
    c2.markdown(f"""<div class="metric-card amber">
        <div class="metric-label">Best Pearson r</div>
        <div class="metric-value">{pr}</div>
        <div class="metric-sub">affinity head</div>
    </div>""", unsafe_allow_html=True)
    auc = f"{best['val_auc_pose']:.3f}" if pd.notna(best.get('val_auc_pose')) else "—"
    c3.markdown(f"""<div class="metric-card blue">
        <div class="metric-label">Pose AUC</div>
        <div class="metric-value">{auc}</div>
        <div class="metric-sub">RMSD &lt; 2Å</div>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.4, 1], gap="large")
    with col_l:
        st.markdown('<div class="section-label">Experiments</div>', unsafe_allow_html=True)
        st.dataframe(exps, use_container_width=True, hide_index=True)
        st.markdown('<div class="section-label" style="margin-top:1.2rem">Model Runs</div>', unsafe_allow_html=True)
        st.dataframe(runs, use_container_width=True, hide_index=True)

    with col_r:
        st.markdown('<div class="section-label">MTL vs STL — val RMSE</div>', unsafe_allow_html=True)
        import altair as alt
        abl = runs[runs["val_rmse"].notna()][["experiment","model_type","val_rmse"]].drop_duplicates()
        if not abl.empty:
            ch = (alt.Chart(abl)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("experiment:N", title=None, axis=alt.Axis(labelAngle=-20, labelColor="#9aa4ae")),
                    y=alt.Y("val_rmse:Q", title="val RMSE", scale=alt.Scale(zero=False),
                            axis=alt.Axis(labelColor="#9aa4ae", titleColor="#9aa4ae", gridColor="#e8e0d5")),
                    color=alt.Color("model_type:N", scale=alt.Scale(
                        domain=["MTL","STL"], range=["#0f766e","#2563eb"]),
                        legend=alt.Legend(labelColor="#5f6b76", titleColor="#5f6b76")),
                    tooltip=["experiment","model_type","val_rmse"],
                )
                .properties(height=220)
                .configure_view(strokeWidth=0, fill="rgba(255,255,255,0)")
            )
            st.altair_chart(ch, use_container_width=True)

        st.markdown('<div class="section-label" style="margin-top:1rem">RL Summary</div>', unsafe_allow_html=True)
        rl_df = _rl_df()
        if not rl_df.empty:
            st.markdown(f"""
            <div class="glass" style="padding:1.2rem 1.4rem">
                {score_bar_html("Best Reward",   rl_df['reward'].max(),    f"{rl_df['reward'].max():.3f}",   rl_df['reward'].max()*100,   "#0f766e")}
                {score_bar_html("Best pKd",      rl_df['pred_pkd'].max(),  f"{rl_df['pred_pkd'].max():.2f}", min(100,(rl_df['pred_pkd'].max()-2)/10*100), "#2563eb") if 'pred_pkd' in rl_df else ""}
                {score_bar_html("Mean QED",      rl_df['r_qed'].mean(),    f"{rl_df['r_qed'].mean():.3f}",   rl_df['r_qed'].mean()*100,   "#d97706") if 'r_qed' in rl_df else ""}
                {score_bar_html("Mol Count",     len(rl_df),               str(len(rl_df)),                  min(100,len(rl_df)/2),        "#7c3aed")}
            </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Binding Predictor
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Binding Predictor":
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED as RDKitQED

    # ── Pocket selector ───────────────────────────────────────────────────────
    pocket_options = _pocket_options()   # [(label, pdb_id, pkd), ...]
    labels  = [o[0] for o in pocket_options]
    pdb_ids = [o[1] for o in pocket_options]

    default_idx = next((i for i, p in enumerate(pdb_ids) if p == "6e9a"), 0)

    st.markdown('<div class="section-label">Protein Pocket</div>', unsafe_allow_html=True)
    pocket_col1, pocket_col2 = st.columns([2, 1])
    with pocket_col1:
        selected_label = st.selectbox(
            "Select pocket (PDB ID — pKd)",
            labels,
            index=default_idx,
            label_visibility="collapsed",
            help="All 150 PDBbind training pockets, sorted by binding affinity (pKd)",
        )
    selected_idx = labels.index(selected_label)
    selected_pdb = pdb_ids[selected_idx]
    selected_pkd = pocket_options[selected_idx][2]
    with pocket_col2:
        st.markdown(f"""
        <div class="metric-card teal" style="padding:0.8rem 1rem;margin-bottom:0">
            <div class="metric-label">Pocket pKd</div>
            <div class="metric-value" style="font-size:1.5rem">{selected_pkd:.2f}</div>
            <div class="metric-sub">{selected_pdb.upper()}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    col_in, col_mol = st.columns([1, 1.2], gap="large")

    with col_in:
        st.markdown('<div class="section-label">Ligand SMILES</div>', unsafe_allow_html=True)
        smiles_input = st.text_input(
            "SMILES string",
            value="C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
            help="Default: erlotinib",
            label_visibility="collapsed",
        )
        predict_btn = st.button("Run GNN Prediction")

    with col_mol:
        st.markdown('<div class="section-label">Ligand Structure</div>', unsafe_allow_html=True)
        if smiles_input:
            st.markdown(mol_svg_html(smiles_input), unsafe_allow_html=True)
        else:
            st.markdown('<div class="mol-panel"><span style="color:#9aa4ae;font-size:0.85rem">Enter SMILES above</span></div>', unsafe_allow_html=True)

    if predict_btn and smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            st.error("Invalid SMILES — RDKit could not parse.")
        else:
            with st.spinner(f"Running GNN oracle against pocket {selected_pdb.upper()}…"):
                import torch
                model, _ = _load_gnn()
                if model is None:
                    pred_pkd, pred_pose, pred_select = 7.49, 0.72, 0.61
                    st.warning("Checkpoint not found — placeholder values shown.")
                else:
                    try:
                        # ── Primary pocket inference ──────────────────────
                        g = _build_graph(smiles_input, pocket_id=selected_pdb)
                        if g is None: raise ValueError("Conformer generation failed")
                        with torch.no_grad():
                            out = model(g)
                        pred_pkd  = float(out["affinity"].item())
                        pred_pose = float(torch.sigmoid(out["pose"]).item()) if "pose" in out else 0.0

                        # ── EGFR selectivity: ΔpKd vs 6E9A reference ─────
                        # selectivity_head uses synthetic binary labels — not
                        # meaningful. Real selectivity = pKd(target) − pKd(EGFR).
                        if selected_pdb.lower() == "6e9a":
                            # already scoring against EGFR — no second pass needed
                            egfr_pkd = pred_pkd
                            delta_pkd = 0.0
                        else:
                            g_egfr = _build_graph(smiles_input, pocket_id="6e9a")
                            if g_egfr is not None:
                                with torch.no_grad():
                                    out_egfr = model(g_egfr)
                                egfr_pkd  = float(out_egfr["affinity"].item())
                                delta_pkd = pred_pkd - egfr_pkd   # + means more selective for target
                            else:
                                egfr_pkd  = float("nan")
                                delta_pkd = float("nan")
                    except Exception as e:
                        st.warning(f"Inference failed: {e} — placeholder shown.")
                        pred_pkd, pred_pose = 7.49, 0.72
                        egfr_pkd, delta_pkd = 7.49, 0.0

            qed_val = float(RDKitQED.qed(mol))
            mw_val  = float(Descriptors.MolWt(mol))

            # ── Selectivity display helpers ───────────────────────────────
            if selected_pdb.lower() == "6e9a":
                sel_label = "pKd vs EGFR (6E9A)"
                sel_value = f"{pred_pkd:.2f}"
                sel_sub   = "scoring against EGFR directly"
                sel_color = "teal"
                delta_html = ""
            elif not np.isnan(delta_pkd):
                sign      = "+" if delta_pkd >= 0 else ""
                d_class   = "metric-delta-good" if delta_pkd >= 0 else "metric-delta-bad"
                sel_label = f"ΔpKd vs EGFR (6E9A)"
                sel_value = f"{sign}{delta_pkd:.2f}"
                sel_sub   = f"target {pred_pkd:.2f}  —  EGFR {egfr_pkd:.2f}"
                sel_color = "teal" if delta_pkd >= 0 else "coral"
                delta_html = f'<span class="{d_class}">{sign}{delta_pkd:.2f}</span>'
            else:
                sel_label = "EGFR ΔpKd"
                sel_value = "—"
                sel_sub   = "EGFR graph build failed"
                sel_color = "amber"
                delta_html = ""

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"""<div class="metric-card teal">
                <div class="metric-label">Predicted pKd ({selected_pdb.upper()})</div>
                <div class="metric-value">{pred_pkd:.2f}</div>
                <div class="metric-sub">vs pocket pKd {selected_pkd:.2f}</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class="metric-card amber">
                <div class="metric-label">Pose Quality</div>
                <div class="metric-value">{pred_pose:.2f}</div>
                <div class="metric-sub">P(RMSD &lt; 2Å)</div>
            </div>""", unsafe_allow_html=True)
            c3.markdown(f"""<div class="metric-card {sel_color}">
                <div class="metric-label">{sel_label}</div>
                <div class="metric-value">{sel_value}</div>
                <div class="metric-sub">{sel_sub}</div>
            </div>""", unsafe_allow_html=True)

            pkd_pct      = min(100, max(0, (pred_pkd  - 2) / 10 * 100))
            egfr_pct     = min(100, max(0, (egfr_pkd  - 2) / 10 * 100)) if not np.isnan(egfr_pkd) else 0
            egfr_display = f"{egfr_pkd:.2f}" if not np.isnan(egfr_pkd) else "—"

            st.markdown(f"""
            <div class="glass" style="padding:1.4rem 1.6rem;margin-top:0.5rem">
                <div class="section-label">Binding &amp; Drug-likeness Profile</div>
                {score_bar_html(f"pKd · {selected_pdb.upper()}", pred_pkd, f"{pred_pkd:.2f}", pkd_pct,
                    "#0f766e" if pred_pkd>8 else "#2563eb" if pred_pkd>6 else "#d97706")}
                {score_bar_html("pKd · EGFR (6E9A ref)", egfr_pkd if not np.isnan(egfr_pkd) else 0,
                    egfr_display, egfr_pct, "#347a7b")}
                {score_bar_html("QED (drug-likeness)", qed_val, f"{qed_val:.3f}", qed_val*100,
                    "#15803d" if qed_val>0.6 else "#d97706" if qed_val>0.4 else "#b42318")}
                {score_bar_html("Pose confidence",    pred_pose, f"{pred_pose:.2f}", pred_pose*100,
                    "#15803d" if pred_pose>0.7 else "#d97706")}
                {score_bar_html("Mol weight (Da)",    mw_val, f"{mw_val:.0f}", min(100,mw_val/700*100),
                    "#15803d" if mw_val<=500 else "#d97706" if mw_val<=700 else "#b42318")}
            </div>""", unsafe_allow_html=True)

            _db_execute("""
                INSERT INTO dbo.binding_predictions
                    (run_id, smiles, pocket_pdb, pred_pkd, pred_pose_prob, pred_select_prob)
                VALUES (1, :smiles, :pocket, :pkd, :pose, :sel)
            """, {"smiles": smiles_input, "pocket": selected_pdb.upper(),
                  "pkd": pred_pkd, "pose": pred_pose, "sel": pred_select})

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RL Browser
# ═════════════════════════════════════════════════════════════════════════════
elif page == "RL Browser":
    st.markdown('<div class="section-label">REINFORCE · frozen GNN oracle · R = 0.5·affinity + 0.2·QED + 0.2·SA + 0.1·MW</div>',
                unsafe_allow_html=True)

    df = _db_query("""
        SELECT smiles, reward, r_affinity, r_qed, r_sa, r_mw, pred_pkd, step
        FROM dbo.rl_molecules
        WHERE experiment_id=(SELECT id FROM dbo.experiments WHERE name='rl_reinforce_egfr')
        ORDER BY reward DESC
    """)
    if df.empty:
        df = _rl_df()
    if "reward_total" in df.columns and "reward" not in df.columns:
        df = df.rename(columns={"reward_total":"reward"})

    if df.empty:
        st.info("No RL results — run Phase 3 notebook first.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"""<div class="metric-card teal">
            <div class="metric-label">Total Molecules</div>
            <div class="metric-value">{len(df)}</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card amber">
            <div class="metric-label">Best Reward</div>
            <div class="metric-value">{df['reward'].max():.3f}</div>
        </div>""", unsafe_allow_html=True)
        pkd_max = f"{df['pred_pkd'].max():.2f}" if 'pred_pkd' in df.columns else "—"
        c3.markdown(f"""<div class="metric-card blue">
            <div class="metric-label">Best pKd</div>
            <div class="metric-value">{pkd_max}</div>
        </div>""", unsafe_allow_html=True)
        qed_m = f"{df['r_qed'].mean():.3f}" if 'r_qed' in df.columns else "—"
        c4.markdown(f"""<div class="metric-card coral">
            <div class="metric-label">Mean QED</div>
            <div class="metric-value">{qed_m}</div>
        </div>""", unsafe_allow_html=True)

        tab_chart, tab_mols, tab_structs = st.tabs(["Reward Landscape", "Top Molecules", "Structures"])

        with tab_chart:
            import altair as alt
            if 'pred_pkd' in df.columns and 'reward' in df.columns:
                pdf = df.dropna(subset=["pred_pkd","reward"]).reset_index(drop=True)
                sc = (alt.Chart(pdf)
                    .mark_circle(size=80, opacity=0.85)
                    .encode(
                        x=alt.X("pred_pkd:Q", title="Predicted pKd",
                                axis=alt.Axis(labelColor="#9aa4ae", titleColor="#9aa4ae", gridColor="#e8e0d5")),
                        y=alt.Y("reward:Q",   title="Total Reward",
                                axis=alt.Axis(labelColor="#9aa4ae", titleColor="#9aa4ae", gridColor="#e8e0d5")),
                        color=alt.Color("r_qed:Q", title="QED",
                                        scale=alt.Scale(scheme="tealblues"),
                                        legend=alt.Legend(labelColor="#5f6b76",titleColor="#5f6b76")),
                        tooltip=["smiles","reward","pred_pkd","r_qed","r_sa"],
                    )
                    .properties(height=380)
                    .configure_view(strokeWidth=0, fill="rgba(251,252,254,0.9)")
                )
                st.altair_chart(sc, use_container_width=True)

        with tab_mols:
            top = df.nlargest(10, "reward").reset_index(drop=True)
            # Rank pill + property bars per row
            for i, row in top.iterrows():
                rank_color = "#0f766e" if i==0 else "#193a3b"
                pkd_str = f"{row['pred_pkd']:.2f}" if pd.notna(row.get('pred_pkd')) else "—"
                r_color = "#15803d" if row['reward']>0.6 else "#2563eb" if row['reward']>0.4 else "#d97706"
                qed_v = row.get('r_qed', 0) or 0
                sa_v  = row.get('r_sa', 0) or 0
                st.markdown(f"""
                <div class="glass" style="padding:1rem 1.4rem;margin-bottom:0.7rem">
                  <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem">
                    <span style="width:26px;height:26px;border-radius:50%;background:{rank_color};
                          color:white;display:flex;align-items:center;justify-content:center;
                          font-size:0.72rem;font-weight:700;flex-shrink:0">{i+1}</span>
                    <span style="font-family:ui-monospace,monospace;font-size:0.75rem;color:#5f6b76;
                          flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{row['smiles']}</span>
                    <span class="pill pill-teal">pKd {pkd_str}</span>
                    <span class="pill" style="background:rgba(15,118,110,0.1);color:{r_color}">R={row['reward']:.3f}</span>
                  </div>
                  {score_bar_html("QED", qed_v, f"{qed_v:.3f}", qed_v*100,
                      "#15803d" if qed_v>0.6 else "#d97706")}
                  {score_bar_html("SA  (1−score/10)", sa_v, f"{sa_v:.3f}", sa_v*100,
                      "#2563eb" if sa_v>0.7 else "#d97706")}
                </div>""", unsafe_allow_html=True)

        with tab_structs:
            top5 = df.nlargest(5, "reward").reset_index(drop=True)
            from rdkit import Chem
            cols = st.columns(5)
            for i, row in top5.iterrows():
                mol = Chem.MolFromSmiles(row["smiles"])
                if not mol:
                    continue
                svg, _ = mol_svg(row["smiles"], w=300, h=220)
                pkd_s = f"pKd {row['pred_pkd']:.2f}" if pd.notna(row.get('pred_pkd')) else ""
                with cols[i]:
                    if svg:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,#fbfcfe,#eef3f8);
                             border:1px solid rgba(71,85,105,0.12);border-radius:16px;
                             padding:0.6rem;text-align:center">
                          {svg}
                          <div style="font-size:0.7rem;color:#5f6b76;margin-top:4px">
                            R={row['reward']:.3f} &nbsp; {pkd_s}
                          </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.code(row["smiles"][:30], language=None)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — GNN vs Vina
# ═════════════════════════════════════════════════════════════════════════════
elif page == "GNN vs Vina":
    st.markdown('<div class="section-label">GNN vs AutoDock Vina — held-out benchmark</div>',
                unsafe_allow_html=True)

    df = _db_query("""
        SELECT smiles, pocket_pdb, vina_score, gnn_pred_pkd,
               ABS(vina_score - gnn_pred_pkd) AS abs_error
        FROM dbo.vina_benchmarks ORDER BY abs_error DESC
    """)
    if df.empty:
        rng = np.random.RandomState(42)
        true_pkd = rng.uniform(4, 10, 30)
        df = pd.DataFrame({
            "smiles": [f"C{'c'*i}O" for i in range(30)],
            "pocket_pdb": ["6E9A"]*30,
            "vina_score":   true_pkd + rng.normal(0,0.8,30),
            "gnn_pred_pkd": true_pkd + rng.normal(0,1.2,30),
        })
        df["abs_error"] = (df["vina_score"]-df["gnn_pred_pkd"]).abs()
        st.info("Synthetic demo — populate dbo.vina_benchmarks for real results.")

    from scipy.stats import pearsonr
    r, _  = pearsonr(df["vina_score"], df["gnn_pred_pkd"])
    rmse  = float(np.sqrt(((df["vina_score"]-df["gnn_pred_pkd"])**2).mean()))
    n_good= int((df["abs_error"] < 1.0).sum())

    c1,c2,c3 = st.columns(3)
    r_color = "#15803d" if r>0.7 else "#d97706" if r>0.4 else "#b42318"
    c1.markdown(f"""<div class="metric-card teal">
        <div class="metric-label">Pearson r</div>
        <div class="metric-value" style="color:{r_color}">{r:.3f}</div>
        <div class="metric-sub">GNN vs Vina</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card amber">
        <div class="metric-label">RMSE</div>
        <div class="metric-value">{rmse:.3f}</div>
        <div class="metric-sub">pKd units</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card blue">
        <div class="metric-label">|Error| &lt; 1.0</div>
        <div class="metric-value">{n_good}/{len(df)}</div>
        <div class="metric-sub">within 1 pKd unit</div>
    </div>""", unsafe_allow_html=True)

    import altair as alt
    lmin = min(df["vina_score"].min(), df["gnn_pred_pkd"].min()) - 0.5
    lmax = max(df["vina_score"].max(), df["gnn_pred_pkd"].max()) + 0.5
    sc = (alt.Chart(df)
        .mark_circle(size=90, opacity=0.85)
        .encode(
            x=alt.X("vina_score:Q",   title="Vina Score (pKd proxy)",
                    scale=alt.Scale(domain=[lmin,lmax]),
                    axis=alt.Axis(labelColor="#9aa4ae",titleColor="#9aa4ae",gridColor="#e8e0d5")),
            y=alt.Y("gnn_pred_pkd:Q", title="GNN Predicted pKd",
                    scale=alt.Scale(domain=[lmin,lmax]),
                    axis=alt.Axis(labelColor="#9aa4ae",titleColor="#9aa4ae",gridColor="#e8e0d5")),
            color=alt.Color("abs_error:Q", title="|Error|",
                            scale=alt.Scale(scheme="orangered"),
                            legend=alt.Legend(labelColor="#5f6b76",titleColor="#5f6b76")),
            tooltip=["pocket_pdb","vina_score","gnn_pred_pkd","abs_error"],
        ).properties(height=420)
    )
    diag = (alt.Chart(pd.DataFrame({"x":[lmin,lmax],"y":[lmin,lmax]}))
        .mark_line(color="#c4b8aa", strokeDash=[5,4])
        .encode(x="x:Q",y="y:Q"))
    combined = ((sc+diag)
        .configure_view(strokeWidth=0, fill="rgba(251,252,254,0.9)")
    )
    st.altair_chart(combined, use_container_width=True)

    st.markdown('<div class="section-label" style="margin-top:0.5rem">Data Table</div>', unsafe_allow_html=True)
    st.dataframe(df.sort_values("abs_error", ascending=False), use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SQL Console
# ═════════════════════════════════════════════════════════════════════════════
elif page == "SQL Console":
    st.markdown('<div class="section-label">SELECT-only queries against gnnbind database</div>',
                unsafe_allow_html=True)

    DEFAULT_SQL = """SELECT TOP 20
    e.name        AS experiment,
    rl.smiles,
    rl.reward,
    rl.pred_pkd,
    rl.r_qed,
    rl.r_sa
FROM dbo.rl_molecules rl
JOIN dbo.experiments e ON e.id = rl.experiment_id
ORDER BY rl.reward DESC"""

    sql = st.text_area("SQL Query", value=DEFAULT_SQL, height=200)

    c1, c2 = st.columns([1, 5])
    run_btn = c1.button("Run Query")

    if run_btn:
        if sql.strip().upper().startswith("SELECT"):
            with st.spinner("Querying…"):
                result = _db_query(sql)
            if result.empty:
                st.info("0 rows returned (or DB unavailable).")
            else:
                st.markdown(f'<span class="pill pill-green">{len(result)} rows</span>', unsafe_allow_html=True)
                st.dataframe(result, use_container_width=True, hide_index=True)
                c2.download_button("Download CSV", result.to_csv(index=False),
                                   "query_result.csv", "text/csv")
        else:
            st.error("Only SELECT statements allowed.")
