"""GNNBindOptimizer — Streamlit UI"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re, io, json
from html import escape
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

:root {
    --bg: #f7f0e8;
    --ink: #19212a;
    --muted: #5f6b76;
    --faint: #9aa4ae;
    --teal: #0f766e;
    --teal-dark: #193a3b;
    --blue: #2563eb;
    --amber: #d97706;
    --coral: #d97757;
    --card: rgba(255,255,255,0.84);
    --line: rgba(25,33,42,0.09);
    --shadow: 0 4px 40px rgba(25,58,59,0.06), 0 1px 8px rgba(25,58,59,0.03);
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', -apple-system, sans-serif !important;
    background: var(--bg) !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 1.3rem !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #193a3b 0%, #224c4d 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #edf7f5 !important; }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] { display: none; }
[data-testid="stSidebar"] .block-container {
    padding: 1.4rem 1.1rem 1.7rem !important;
}
.sidebar-brand {
    padding: 0.4rem 0.25rem 1.25rem;
    border-bottom: 1px solid rgba(255,255,255,0.10);
    margin-bottom: 1.1rem;
}
.sidebar-logo {
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #ffffff;
}
.sidebar-logo em {
    font-family: 'Instrument Serif', Georgia, serif;
    font-style: italic;
    font-weight: 400;
    color: #9ff5e7;
}
.sidebar-caption {
    color: rgba(237,247,245,0.58) !important;
    font-size: 0.76rem;
    line-height: 1.35;
    margin-top: 0.35rem;
}
[data-testid="stSidebar"] .stRadio label {
    color: #edf7f5 !important;
    font-size: 0.98rem !important;
    font-weight: 500 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    padding: 0.28rem 0 !important;
}
[data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"] {
    font-size: 1.1rem !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
    font-size: 0.98rem !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 0.25rem;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    border-radius: 12px;
    padding: 0.42rem 0.55rem !important;
    transition: background 0.15s ease, transform 0.15s ease;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: rgba(237,247,245,0.5) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.sidebar-card {
    margin-top: 1rem;
    padding: 0.9rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.10);
}
.sidebar-card-title {
    color: rgba(237,247,245,0.62);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}
.sidebar-card-line {
    display:flex;
    justify-content:space-between;
    gap: 0.8rem;
    font-size: 0.76rem;
    color: rgba(237,247,245,0.70);
    margin-top: 0.35rem;
}
.sidebar-card-line strong {
    color: #ffffff;
    font-weight: 700;
}

/* ── Main canvas ─────────────────────────────────────────────────────────── */
.stApp, .main .block-container {
    background: var(--bg) !important;
    max-width: 1400px !important;
    padding: 1.5rem 2rem !important;
}

/* ── Header ──────────────────────────────────────────────────────────────── */
.gnn-header {
    background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,255,255,0.72));
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.6);
    border-radius: 22px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.35rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    box-shadow: var(--shadow);
}
.gnn-header-main {
    min-width: 0;
}
.gnn-header-meta {
    display:flex;
    align-items:center;
    gap:0.55rem;
    flex-wrap:wrap;
    justify-content:flex-end;
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
.gnn-badge.secondary {
    background: rgba(25,33,42,0.06);
    color: #5f6b76;
}
.gnn-badge.secondary::before { background: #9aa4ae; }

/* ── Page framing ────────────────────────────────────────────────────────── */
.page-head {
    margin: 0.2rem 0 1.15rem;
    display:flex;
    align-items:flex-end;
    justify-content:space-between;
    gap:1.2rem;
}
.page-kicker {
    font-size:0.68rem;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:0.12em;
    color:#0f766e;
    margin-bottom:0.35rem;
}
.page-title {
    font-size:2rem;
    font-weight:700;
    color:#193a3b;
    letter-spacing:-0.03em;
    line-height:1.04;
    margin:0;
}
.page-copy {
    color:#7a8b8c;
    font-size:0.95rem;
    margin:0.45rem 0 0;
    max-width: 760px;
}
.page-chip-row {
    display:flex;
    gap:0.45rem;
    flex-wrap:wrap;
    justify-content:flex-end;
}
.micro-chip {
    background:#f0ebe3;
    color:#6d6848;
    font-size:0.68rem;
    font-family:ui-monospace,monospace;
    padding:0.26rem 0.62rem;
    border-radius:7px;
    border:1px solid rgba(109,104,72,0.18);
    white-space:nowrap;
}
.micro-chip.teal {
    background:#193a3b;
    color:#7efce1;
    border-color:#2d5e5f;
    font-weight:700;
}
.micro-chip.blue {
    background:#d9e8ee;
    color:#335356;
    border-color:#bed1d7;
}
.micro-chip.amber {
    background:#e9e3c7;
    color:#6d6848;
    border-color:#d5cca6;
}

/* ── Glassmorphism card ───────────────────────────────────────────────────── */
.glass {
    background: var(--card);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.6);
    border-radius: 20px;
    box-shadow: var(--shadow);
}

/* ── Metric card ─────────────────────────────────────────────────────────── */
.metric-card {
    background: var(--card);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.6);
    border-radius: 18px;
    padding: 1.4rem 1.5rem;
    box-shadow: var(--shadow);
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
.stTextArea textarea,
.stSelectbox [data-baseweb="select"] > div,
.stNumberInput input {
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

.empty-state {
    border: 1px dashed rgba(25,58,59,0.22);
    background: rgba(255,255,255,0.45);
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    color: #5f6b76;
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

@media (max-width: 900px) {
    .stApp, .main .block-container {
        padding: 1rem !important;
    }
    .gnn-header,
    .page-head {
        align-items:flex-start;
        flex-direction:column;
    }
    .gnn-header-meta,
    .page-chip-row {
        justify-content:flex-start;
    }
    .page-title { font-size:1.55rem; }
    .metric-value { font-size:1.65rem; }
    .score-row { align-items:flex-start; flex-direction:column; gap:5px; }
    .score-name, .score-val, .score-delta { width:auto; text-align:left; }
    .score-bar-bg { width:100%; flex:none; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
_rl_json_path = Path(__file__).parent.parent / "data" / "rl_results" / "rl_results.json"
try:
    _rl_cfg  = json.loads(_rl_json_path.read_text()).get("config", {})
    _hdr_pocket = _rl_cfg.get("ref_pocket", "6e9a").upper()
    _hdr_pkd    = _rl_cfg.get("ref_pkd", 0)
    _hdr_pkd_str = f"{_hdr_pkd:.2f}" if _hdr_pkd else "—"
except Exception:
    _hdr_pocket, _hdr_pkd_str = "6E9A", "—"

st.markdown(f"""
<div class="gnn-header">
  <div class="gnn-header-main">
    <div class="gnn-logo">GNN<em>Bind</em>Optimizer</div>
    <div class="gnn-tagline">Structure-aware binding affinity prediction + RL molecular generation</div>
  </div>
  <div class="gnn-header-meta">
    <div class="gnn-badge">Pocket {_hdr_pocket} · pKd {_hdr_pkd_str}</div>
    <div class="gnn-badge secondary">HeteroGNN · frozen oracle</div>
  </div>
</div>
""", unsafe_allow_html=True)


def page_header(kicker: str, title: str, copy: str, chips: list[tuple[str, str]] | None = None) -> None:
    """Render a consistent page title area with optional status chips."""
    chip_html = ""
    if chips:
        rendered = []
        for label, tone in chips:
            css_class = f"micro-chip {tone}".strip()
            rendered.append(f'<span class="{css_class}">{escape(label)}</span>')
        chip_html = f'<div class="page-chip-row">{"".join(rendered)}</div>'

    st.markdown(f"""
    <div class="page-head">
      <div>
        <div class="page-kicker">{escape(kicker)}</div>
        <h2 class="page-title">{escape(title)}</h2>
        <p class="page-copy">{escape(copy)}</p>
      </div>
      {chip_html}
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
    return f'<div class="mol-panel"><div class="mol-fallback">{escape(smiles)}</div></div>'


@st.cache_data(show_spinner=False)
def _pocket_svg(pdb_id: str) -> str:
    """
    Generate a unique SVG ribbon from actual Cα coordinates via PCA → 2D.
    Returns an <svg> string to embed directly in the dark viewport.
    """
    import numpy as np
    from sklearn.decomposition import PCA

    pockets = _load_all_pockets()
    g = pockets.get(pdb_id.lower())
    if g is None:
        return ""

    pos = g["residue"].pos.numpy()          # [N, 3]
    lig_centroid = g["ligand"].pos.numpy().mean(axis=0)

    # Project to 2D, centre and scale to fit 500×400 viewBox
    pca   = PCA(n_components=2)
    pos2d = pca.fit_transform(pos)
    c2d   = pca.transform(lig_centroid.reshape(1, -1))[0]

    def _scale(pts, cx, cy, W=500, H=400, pad=60):
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        span = (mx - mn).clip(min=1e-6)
        s = min((W - 2*pad) / span[0], (H - 2*pad) / span[1])
        return (pts - mn) * s + np.array([cx - (mx-mn)[0]*s/2,
                                           cy - (mx-mn)[1]*s/2])

    W, H = 500, 400
    pts  = _scale(pos2d,  W/2, H/2)
    lig  = _scale(c2d.reshape(1,-1), W/2, H/2)[0]

    # Smooth catmull-rom → cubic bezier path through Cα backbone
    def _catmull_to_bezier(pts):
        """Convert ordered Cα positions to SVG cubic bezier path string."""
        if len(pts) < 2:
            return ""
        segs = [f"M {pts[0,0]:.1f},{pts[0,1]:.1f}"]
        for i in range(len(pts) - 1):
            p0 = pts[max(0, i-1)]
            p1 = pts[i]
            p2 = pts[min(len(pts)-1, i+1)]
            p3 = pts[min(len(pts)-1, i+2)]
            # control points (catmull-rom tension=0.5)
            cp1x = p1[0] + (p2[0] - p0[0]) / 6
            cp1y = p1[1] + (p2[1] - p0[1]) / 6
            cp2x = p2[0] - (p3[0] - p1[0]) / 6
            cp2y = p2[1] - (p3[1] - p1[1]) / 6
            segs.append(f"C {cp1x:.1f},{cp1y:.1f} {cp2x:.1f},{cp2y:.1f} {p2[0]:.1f},{p2[1]:.1f}")
        return " ".join(segs)

    backbone = _catmull_to_bezier(pts)

    # Beta-sheet arrows: sample every ~8 residues
    arrows = []
    step = max(3, len(pts) // 10)
    for i in range(0, len(pts) - step, step):
        p1, p2 = pts[i], pts[i + step]
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        length = (dx**2 + dy**2)**0.5 + 1e-6
        nx, ny = -dy/length * 8, dx/length * 8   # normal
        # arrowhead polygon: base left, base right, tip
        bx, by = p1[0] + dx*0.3, p1[1] + dy*0.3
        tx, ty = p1[0] + dx*0.85, p1[1] + dy*0.85
        pts_arrow = (
            f"{bx+nx:.1f},{by+ny:.1f} "
            f"{bx-nx:.1f},{by-ny:.1f} "
            f"{tx:.1f},{ty:.1f}"
        )
        arrows.append(f'<polygon points="{pts_arrow}" fill="#14b8a6" fill-opacity="0.75"/>')

    arrows_svg = "\n".join(arrows)

    # Ligand CPK at binding centroid
    lx, ly = float(lig[0]), float(lig[1])

    svg = f"""<svg viewBox="0 0 {W} {H}" style="width:100%;height:100%;filter:drop-shadow(0 0 14px rgba(15,118,110,0.45))">
  <defs>
    <radialGradient id="gC" cx="30%" cy="30%" r="70%">
      <stop offset="0%" stop-color="#fff"/><stop offset="100%" stop-color="#888"/>
    </radialGradient>
    <radialGradient id="gN2" cx="30%" cy="30%" r="70%">
      <stop offset="0%" stop-color="#60a5fa"/><stop offset="100%" stop-color="#1d4ed8"/>
    </radialGradient>
    <radialGradient id="gO2" cx="30%" cy="30%" r="70%">
      <stop offset="0%" stop-color="#f87171"/><stop offset="100%" stop-color="#991b1b"/>
    </radialGradient>
    <linearGradient id="gR2" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f766e"/>
      <stop offset="50%" stop-color="#4ade80"/>
      <stop offset="100%" stop-color="#115e59"/>
    </linearGradient>
    <filter id="glow"><feGaussianBlur stdDeviation="6" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <!-- Backbone trace (unique per pocket) -->
  <path d="{backbone}" fill="none" stroke="url(#gR2)" stroke-width="20"
        stroke-linecap="round" stroke-linejoin="round" filter="url(#glow)" opacity="0.8"/>
  <!-- Thin inner highlight -->
  <path d="{backbone}" fill="none" stroke="rgba(110,231,183,0.4)" stroke-width="5"
        stroke-linecap="round" stroke-linejoin="round"/>

  <!-- Beta-sheet arrows -->
  {arrows_svg}

  <!-- Ligand glow -->
  <circle cx="{lx:.1f}" cy="{ly:.1f}" r="36" fill="rgba(255,255,255,0.1)" filter="url(#glow)"/>

  <!-- CPK ligand atoms -->
  <line x1="{lx-22:.1f}" y1="{ly:.1f}" x2="{lx:.1f}" y2="{ly:.1f}" stroke="#888" stroke-width="9" stroke-linecap="round"/>
  <line x1="{lx:.1f}" y1="{ly:.1f}" x2="{lx+16:.1f}" y2="{ly-18:.1f}" stroke="#888" stroke-width="9" stroke-linecap="round"/>
  <line x1="{lx:.1f}" y1="{ly:.1f}" x2="{lx+22:.1f}" y2="{ly+14:.1f}" stroke="#888" stroke-width="9" stroke-linecap="round"/>
  <circle cx="{lx-22:.1f}" cy="{ly:.1f}" r="14" fill="url(#gO2)"/>
  <circle cx="{lx:.1f}" cy="{ly:.1f}" r="16" fill="url(#gC)"/>
  <circle cx="{lx+16:.1f}" cy="{ly-18:.1f}" r="12" fill="url(#gN2)"/>
  <circle cx="{lx+22:.1f}" cy="{ly+14:.1f}" r="12" fill="url(#gN2)"/>
</svg>"""

    return svg


@st.cache_data(show_spinner=False)
def _pocket_viz(pdb_id: str):
    """
    2D PCA projection of pocket Cα atoms.
    Returns (png_bytes, stats_dict) — cached per pocket.
    """
    import io, numpy as np, matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pockets = _load_all_pockets()
    g = pockets.get(pdb_id.lower())
    if g is None:
        return None, {}

    pos        = g["residue"].pos.numpy()          # [N, 3]
    n_res      = len(pos)
    edge_index = g["residue", "contact", "residue"].edge_index.numpy()
    n_contacts = edge_index.shape[1] // 2

    # 2D PCA projection
    pca   = PCA(n_components=2)
    pos2d = pca.fit_transform(pos)                 # [N, 2]

    # Depth along the third PC for colour
    pca3     = PCA(n_components=3)
    pos3d    = pca3.fit_transform(pos)
    depth    = pos3d[:, 2]
    d_norm   = (depth - depth.min()) / (depth.ptp() + 1e-8)

    # Ligand centroid projected to same 2D space
    lig_pos    = g["ligand"].pos.numpy()
    centroid   = lig_pos.mean(axis=0, keepdims=True)
    c2d        = pca.transform(centroid)[0]

    # Mean contact edge length (Å)
    edge_dists = []
    for k in range(0, edge_index.shape[1], 2):
        s, d = edge_index[0, k], edge_index[1, k]
        edge_dists.append(np.linalg.norm(pos[s] - pos[d]))
    mean_edge = float(np.mean(edge_dists)) if edge_dists else 0.0

    fig, ax = plt.subplots(figsize=(5.2, 4.2), facecolor="#fbfcfe")
    ax.set_facecolor("#f0f4f8")

    # Contact edges — draw only every other to keep it readable
    for k in range(0, edge_index.shape[1], 2):
        s, d = edge_index[0, k], edge_index[1, k]
        ax.plot([pos2d[s, 0], pos2d[d, 0]],
                [pos2d[s, 1], pos2d[d, 1]],
                color="#a8c0c0", lw=0.55, alpha=0.35, zorder=1)

    # Residue nodes
    sc = ax.scatter(pos2d[:, 0], pos2d[:, 1],
                    c=d_norm, cmap="YlOrRd", vmin=0, vmax=1,
                    s=62, edgecolors="white", linewidths=0.5,
                    zorder=3, alpha=0.88)

    # Ligand centroid marker
    ax.scatter(c2d[0], c2d[1], s=140, marker="*",
               color="#d97706", edgecolors="white", linewidths=0.8,
               zorder=5, label="Ligand centroid")

    cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Depth (PC3)", fontsize=7, color="#7a8b8c")
    cb.ax.tick_params(labelsize=6, colors="#9aa4ae")

    ax.set_xlabel("PC1", fontsize=8, color="#7a8b8c")
    ax.set_ylabel("PC2", fontsize=8, color="#7a8b8c")
    ax.tick_params(labelsize=7, colors="#9aa4ae")
    for sp in ax.spines.values():
        sp.set_color("#d0d7de")
    ax.legend(fontsize=7, loc="upper right",
              framealpha=0.75, edgecolor="#d0d7de")
    ax.set_title(f"{pdb_id.upper()} — pocket Cα map",
                 fontsize=9, color="#193a3b", fontweight="600", pad=8)

    plt.tight_layout(pad=0.6)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                facecolor="#fbfcfe")
    plt.close(fig)
    buf.seek(0)

    stats = {
        "n_residues":  n_res,
        "n_contacts":  n_contacts,
        "mean_edge_A": round(mean_edge, 2),
        "pkd":         round(float(g.y_affinity.item()), 2),
    }
    return buf.getvalue(), stats


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


@st.cache_data(show_spinner=False, ttl=60)
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


@st.cache_data(show_spinner=False, ttl=60)
def _rl_summary() -> dict:
    """Returns summary stats from rl_results.json (full-run counts, not just top_molecules)."""
    p = Path(__file__).parent.parent / "data" / "rl_results" / "rl_results.json"
    if not p.exists(): return {}
    d = json.loads(p.read_text())
    s = d.get("summary", {})
    hist = d.get("history", {})
    # Validity: from history, count steps where at least 1 valid mol was scored
    validity_list = hist.get("validity", [])
    steps = len(hist.get("step", []))
    config = d.get("config", {})
    batch = config.get("rl_batch", 64)
    total_gen = steps * batch if steps else s.get("total_generated", 0)
    # Estimate valid from validity rate × total generated
    mean_val = float(np.mean(validity_list)) if validity_list else s.get("validity_rate", 0)
    total_valid = int(total_gen * mean_val)
    return {
        "total_generated": total_gen,
        "total_valid": total_valid,
        "validity_rate": mean_val,
        "best_reward": s.get("best_reward", 0),
        "best_pkd": s.get("best_pkd", 0),
        "steps": steps,
        "ref_pocket": config.get("ref_pocket", "6e9a").upper(),
        "ref_pkd": config.get("ref_pkd", 0),
    }


# ── Sidebar nav ───────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div class="sidebar-brand">
  <div class="sidebar-logo">GNN<em>Bind</em></div>
  <div class="sidebar-caption">Binding affinity prediction, molecule generation, and SQL-backed experiment review.</div>
</div>
<p>Navigation</p>
""", unsafe_allow_html=True)
PAGES = ["Summary", "Binding Predictor", "RL Generator", "GNN vs Vina", "SQL Console"]
page  = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

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

st.sidebar.markdown(f"""
<div class="sidebar-card">
  <div class="sidebar-card-title">Active Context</div>
  <div class="sidebar-card-line"><span>Pocket</span><strong>{escape(_hdr_pocket)}</strong></div>
  <div class="sidebar-card-line"><span>Reference pKd</span><strong>{escape(_hdr_pkd_str)}</strong></div>
  <div class="sidebar-card-line"><span>Model</span><strong>HGTConv MTL</strong></div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Summary
# ═════════════════════════════════════════════════════════════════════════════
if page == "Summary":
    import altair as alt, re, torch as _torch

    # ── Dynamic data sources ──────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def _summary_data():
        root = Path(__file__).parent.parent
        # Dataset size
        ds_path = root / "data" / "processed" / "dataset.pt"
        try:
            ds = _torch.load(str(ds_path), map_location="cpu", weights_only=False)
            n_complexes = len(ds)
        except Exception:
            n_complexes = 150

        # GNN metrics — parse best checkpoint filename
        ckpt_dir = root / "checkpoints"
        mtl_rmse = stl_rmse = mtl_r = stl_r = None
        mtl_epochs = stl_epochs = 0
        for f in ckpt_dir.glob("*.ckpt"):
            m = re.search(r"epoch=(\d+)-val_rmse=([\d.]+?)(?:-|\.ckpt)", f.name)
            if not m: continue
            ep, rmse = int(m.group(1)), float(m.group(2))
            if "mtl" in f.name and (mtl_rmse is None or rmse < mtl_rmse):
                mtl_rmse, mtl_epochs = rmse, ep + 1
            if "stl" in f.name and (stl_rmse is None or rmse < stl_rmse):
                stl_rmse, stl_epochs = rmse, ep + 1
        # Pearson r from known training (not in filename — use hardcoded from run logs)
        mtl_r = 0.541; stl_r = 0.489

        # RL metrics
        rl_path = root / "data" / "rl_results" / "rl_results.json"
        rl_best_pkd = rl_best_rew = rl_validity = rl_steps = 0
        rl_pocket = "6E9A"
        rl_reward_history = []
        rl_max_history = []
        if rl_path.exists():
            import json as _json
            d = _json.loads(rl_path.read_text())
            s = d.get("summary", {})
            c = d.get("config", {})
            h = d.get("history", {})
            rl_best_pkd  = s.get("best_pkd", 0)
            rl_best_rew  = s.get("best_reward", 0)
            rl_steps     = len(h.get("step", []))
            rl_pocket    = c.get("ref_pocket", "6e9a").upper()
            val_list     = h.get("validity", [])
            rl_validity  = float(np.mean(val_list)) if val_list else 0
            rl_reward_history = h.get("reward_mean", [])
            rl_max_history    = h.get("reward_max",  [])

        return {
            "n_complexes": n_complexes,
            "mtl_rmse": mtl_rmse or 1.924, "mtl_r": mtl_r, "mtl_epochs": mtl_epochs or 20,
            "stl_rmse": stl_rmse or 2.034, "stl_r": stl_r, "stl_epochs": stl_epochs or 10,
            "rl_best_pkd": rl_best_pkd, "rl_best_rew": rl_best_rew,
            "rl_validity": rl_validity, "rl_steps": rl_steps,
            "rl_pocket": rl_pocket,
            "rl_reward_history": rl_reward_history,
            "rl_max_history":    rl_max_history,
        }

    S = _summary_data()

    page_header(
        "Project Overview",
        "Structure-Aware Drug Design Workspace",
        "A compact readout of dataset construction, HeteroGNN performance, and RL molecular optimization progress.",
        [
            ("PDBbind v2020", "amber"),
            (f"{S['rl_steps']} RL steps", "blue"),
            (f"Pocket {S['rl_pocket']}", "teal"),
        ],
    )

    # ── 5 KPI chips ───────────────────────────────────────────────────────────
    k1,k2,k3,k4,k5 = st.columns(5)
    def _kpi(col, label, value, sub, color="teal"):
        col.markdown(f'<div class="metric-card {color}">'
                     f'<div class="metric-label">{label}</div>'
                     f'<div class="metric-value">{value}</div>'
                     f'<div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

    _kpi(k1, "Complexes",     str(S["n_complexes"]),         "PDBbind refined", "teal")
    _kpi(k2, "GNN val RMSE",  f"{S['mtl_rmse']:.3f}",        "MTL · pKd units", "amber")
    _kpi(k3, "Pearson r",     f"{S['mtl_r']:.3f}",            "affinity head",   "blue")
    _kpi(k4, "Best RL pKd",   f"{S['rl_best_pkd']:.2f}",     f"pocket {S['rl_pocket']}", "teal")
    _kpi(k5, "RL Validity",   f"{S['rl_validity']:.0%}",      f"{S['rl_steps']} steps",   "coral")

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)

    # ── 3 Pipeline phase cards ────────────────────────────────────────────────
    ph1, ph2, ph3 = st.columns(3, gap="large")

    def _phase_card(col, phase, icon, title, metric_label, metric_val, lines):
        body = "".join(f'<div style="display:flex;align-items:flex-start;gap:6px;margin-bottom:4px">'
                       f'<span style="color:#0f766e;font-size:0.8rem;margin-top:1px">▸</span>'
                       f'<span style="font-size:0.82rem;color:#5f6b76;line-height:1.4">{l}</span></div>'
                       for l in lines)
        col.markdown(f"""
        <div class="glass" style="padding:1.4rem 1.5rem;height:100%">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:1rem">
            <span style="background:#193a3b;color:#7efce1;font-size:0.68rem;
                  font-family:ui-monospace,monospace;padding:2px 8px;border-radius:4px;
                  font-weight:700">{phase}</span>
            <span style="font-size:1.1rem">{icon}</span>
          </div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1.05rem;font-weight:700;
                      color:#193a3b;margin-bottom:0.6rem">{title}</div>
          <div style="background:rgba(25,58,59,0.06);border-radius:10px;padding:0.7rem 1rem;
                      margin-bottom:1rem;display:flex;align-items:baseline;gap:8px">
            <span style="font-family:ui-monospace,monospace;font-size:1.5rem;font-weight:700;
                         color:#0f766e">{metric_val}</span>
            <span style="font-size:0.72rem;color:#7a8b8c;text-transform:uppercase;
                         letter-spacing:0.07em">{metric_label}</span>
          </div>
          {body}
        </div>""", unsafe_allow_html=True)

    _phase_card(ph1, "PHASE 1", "🗂️", "Dataset & Graph Construction",
        "complexes", str(S["n_complexes"]),
        [f"PDBbind v2020 refined set",
         "HeteroData graph: ligand atoms + pocket residues",
         "pKd range 2.0 – 11.9 · mean 6.4",
         "Edges: bond · contact · interacts (5 Å cutoff)"])

    _phase_card(ph2, "PHASE 2", "🧠", "HeteroGNN Training",
        "val RMSE (MTL)", f"{S['mtl_rmse']:.3f}",
        [f"4 × HGTConv layers · hidden 128 · 4 heads",
         f"MTL: RMSE {S['mtl_rmse']:.3f} · r {S['mtl_r']:.3f} ({S['mtl_epochs']} epochs)",
         f"STL: RMSE {S['stl_rmse']:.3f} · r {S['stl_r']:.3f} ({S['stl_epochs']} epochs)",
         "Heads: affinity · pose quality · selectivity"])

    _phase_card(ph3, "PHASE 3", "⚗️", "RL Molecular Generation",
        "best pKd", f"{S['rl_best_pkd']:.2f}",
        [f"REINFORCE · character-level LSTM policy",
         f"{S['rl_steps']} steps · batch 64 · pocket {S['rl_pocket']}",
         f"Best reward {S['rl_best_rew']:.3f} · validity {S['rl_validity']:.0%}",
         "R = 0.5·affinity + 0.2·QED + 0.2·SA + 0.1·MW"])

    st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)

    # ── Two charts ────────────────────────────────────────────────────────────
    ch_l, ch_r = st.columns([1, 1.6], gap="large")

    with ch_l:
        st.markdown('<div class="section-label">MTL vs STL — val RMSE</div>', unsafe_allow_html=True)
        abl = pd.DataFrame([
            {"Model": "MTL (3 heads)", "val RMSE": S["mtl_rmse"], "type": "MTL"},
            {"Model": "STL (affinity)", "val RMSE": S["stl_rmse"], "type": "STL"},
        ])
        ch = (alt.Chart(abl)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("Model:N", title=None,
                        axis=alt.Axis(labelColor="#9aa4ae", labelFontSize=11)),
                y=alt.Y("val RMSE:Q", scale=alt.Scale(zero=True),
                        axis=alt.Axis(labelColor="#9aa4ae", titleColor="#9aa4ae",
                                      gridColor="#e8e0d5", title="val RMSE (pKd)")),
                color=alt.Color("type:N",
                    scale=alt.Scale(domain=["MTL","STL"], range=["#0f766e","#2563eb"]),
                    legend=alt.Legend(labelColor="#5f6b76", titleColor="#5f6b76")),
                tooltip=["Model","val RMSE"],
            ).properties(height=260)
             .configure_view(strokeWidth=0, fill="rgba(255,255,255,0)")
        )
        st.altair_chart(ch, use_container_width=True)

    with ch_r:
        st.markdown('<div class="section-label">RL Reward Curve — 300 training steps</div>',
                    unsafe_allow_html=True)
        if S["rl_reward_history"]:
            steps = list(range(len(S["rl_reward_history"])))
            rl_hist = pd.DataFrame({
                "step":       steps * 2,
                "reward":     S["rl_reward_history"] + S["rl_max_history"],
                "series":     ["Mean reward"] * len(steps) + ["Max reward"] * len(steps),
            })
            rc = (alt.Chart(rl_hist)
                .mark_line(strokeWidth=1.8, opacity=0.9)
                .encode(
                    x=alt.X("step:Q", title="Training step",
                            axis=alt.Axis(labelColor="#9aa4ae", titleColor="#9aa4ae",
                                          gridColor="#e8e0d5")),
                    y=alt.Y("reward:Q", title="Reward",
                            axis=alt.Axis(labelColor="#9aa4ae", titleColor="#9aa4ae",
                                          gridColor="#e8e0d5")),
                    color=alt.Color("series:N",
                        scale=alt.Scale(domain=["Mean reward","Max reward"],
                                        range=["#7a8b8c","#0f766e"]),
                        legend=alt.Legend(labelColor="#5f6b76", titleColor="#5f6b76")),
                    tooltip=["step","series","reward"],
                ).properties(height=260)
                 .configure_view(strokeWidth=0, fill="rgba(255,255,255,0)")
            )
            st.altair_chart(rc, use_container_width=True)
        else:
            st.info("Run Phase 3 notebook to populate RL reward history.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Binding Predictor
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Binding Predictor":
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED as RDKitQED

    page_header(
        "Interactive Oracle",
        "Binding Predictor",
        "Choose a protein pocket, inspect its graph-derived microenvironment, and score a ligand SMILES with the frozen HeteroGNN.",
        [
            ("Live GNN inference", "teal"),
            ("RDKit descriptors", "blue"),
            ("EGFR delta", "amber"),
        ],
    )

    # ── Pocket selector ───────────────────────────────────────────────────────
    pocket_options = _pocket_options()   # [(label, pdb_id, pkd), ...]
    labels  = [o[0] for o in pocket_options]
    pdb_ids = [o[1] for o in pocket_options]

    default_idx = next((i for i, p in enumerate(pdb_ids) if p == "6e9a"), 0)

    st.markdown('<div class="section-label">Protein Pocket</div>', unsafe_allow_html=True)
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

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ── Cinematic Pocket Hero ─────────────────────────────────────────────────
    _, pocket_stats = _pocket_viz(selected_pdb)
    n_res  = pocket_stats.get("n_residues", "—")
    n_con  = pocket_stats.get("n_contacts",  "—")
    mean_e = pocket_stats.get("mean_edge_A", "—")
    pv_pkd = pocket_stats.get("pkd", "—")

    pocket_svg_str = _pocket_svg(selected_pdb)

    import streamlit.components.v1 as components
    components.html(f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#f7f0e8; font-family:'Space Grotesk',sans-serif; padding:0; }}

  .hero {{
    display:flex; gap:0; border-radius:24px; overflow:hidden;
    border:1px solid rgba(255,255,255,0.7);
    box-shadow:0 8px 40px rgba(25,58,59,0.08);
    height:460px; width:100%;
  }}

  /* ── Dark 3D Viewport ── */
  .viewport {{
    flex:1; background:#0d1f20; position:relative; overflow:hidden;
    cursor:crosshair;
  }}
  .vp-grid {{
    position:absolute; inset:0;
    background-image:linear-gradient(rgba(255,255,255,0.03) 1px,transparent 1px),
                     linear-gradient(90deg,rgba(255,255,255,0.03) 1px,transparent 1px);
    background-size:40px 40px;
  }}
  .vp-glow {{
    position:absolute; top:50%; left:50%;
    transform:translate(-50%,-50%);
    width:340px; height:340px;
    background:radial-gradient(circle,rgba(15,118,110,0.35) 0%,transparent 70%);
    border-radius:50%;
  }}
  .vp-hud-tl {{
    position:absolute; top:14px; left:14px; z-index:10;
    display:flex; align-items:center; gap:8px;
    background:rgba(13,31,32,0.85); backdrop-filter:blur(8px);
    border:1px solid rgba(255,255,255,0.1);
    padding:6px 12px; border-radius:8px;
    font-family:'JetBrains Mono',monospace; font-size:11px; color:rgba(255,255,255,0.75);
  }}
  .hud-dot {{ width:7px; height:7px; border-radius:50%; background:#d97706; animation:hud-pulse 2s ease-in-out infinite; }}
  @keyframes hud-pulse {{ 0%,100%{{opacity:0.6}} 50%{{opacity:1}} }}
  .vp-hud-br {{
    position:absolute; bottom:14px; left:14px; z-index:10;
    display:flex; gap:16px;
    font-family:'JetBrains Mono',monospace; font-size:9px; color:rgba(255,255,255,0.3);
  }}
  .vp-inner {{
    position:absolute; inset:0;
    display:flex; align-items:center; justify-content:center;
    transition:transform 0.15s ease-out;
  }}

  /* protein ribbon */
  .ribbon-wrap {{
    position:absolute; width:72%; height:72%;
    animation:spin-slow 22s linear infinite;
    opacity:0.65;
  }}
  @keyframes spin-slow {{ from{{transform:rotate(0deg)}} to{{transform:rotate(360deg)}} }}

  /* ligand cpk */
  .ligand-wrap {{
    position:relative; z-index:20;
    animation:float 8s ease-in-out infinite;
  }}
  @keyframes float {{ 0%,100%{{transform:translateY(0)}} 50%{{transform:translateY(-12px)}} }}
  .ligand-glow {{
    position:absolute; inset:-20px;
    background:radial-gradient(circle,rgba(255,255,255,0.18) 0%,transparent 70%);
    border-radius:50%;
    animation:glow-pulse 3s ease-in-out infinite;
  }}
  @keyframes glow-pulse {{ 0%,100%{{opacity:0.6;transform:scale(1)}} 50%{{opacity:1;transform:scale(1.1)}} }}

  /* annotation */
  .annotation {{
    position:absolute; top:20%; right:12%;
    display:flex; flex-direction:column; align-items:flex-end; gap:4px;
    pointer-events:none;
  }}
  .ann-chip {{
    background:rgba(13,31,32,0.85); backdrop-filter:blur(6px);
    border:1px solid rgba(255,255,255,0.15);
    padding:4px 10px; border-radius:6px;
    font-family:'JetBrains Mono',monospace; font-size:10px;
    color:rgba(255,255,255,0.8);
    display:flex; align-items:center; gap:6px;
  }}
  .ann-dot {{ width:5px; height:5px; border-radius:50%; background:white; opacity:0.7; }}

</style>
</head>
<body>
<div class="hero">

  <!-- 3D Viewport -->
  <div class="viewport" id="vp">
    <div class="vp-grid"></div>
    <div class="vp-glow"></div>
    <div class="vp-hud-tl">
      <div class="hud-dot"></div>
      {selected_pdb.upper()} · ATP Binding Cleft
    </div>
    <div class="vp-hud-br">
      <span>RESIDUES {n_res}</span>
      <span>CONTACTS {n_con}</span>
      <span>GNN ORACLE ACTIVE</span>
    </div>

    <div class="vp-inner" id="vp-inner">
      <!-- Pocket-specific ribbon + CPK generated from Cα coords -->
      {pocket_svg_str}

      <!-- Annotation -->
      <div class="annotation">
        <div class="ann-chip"><div class="ann-dot"></div>LIGAND POSE</div>
      </div>
    </div><!-- vp-inner -->
  </div><!-- viewport -->

</div><!-- hero -->

<script>
  const inner = document.getElementById('vp-inner');
  const vp    = document.getElementById('vp');
  vp.addEventListener('mousemove', e => {{
    const r  = vp.getBoundingClientRect();
    const x  = (e.clientX - r.left  - r.width  / 2) / (r.width  / 2);
    const y  = (e.clientY - r.top   - r.height / 2) / (r.height / 2);
    inner.style.transform = `perspective(900px) rotateX(${{-y * 8}}deg) rotateY(${{x * 8}}deg)`;
  }});
  vp.addEventListener('mouseleave', () => {{
    inner.style.transform = 'perspective(900px) rotateX(0deg) rotateY(0deg)';
  }});
</script>
</body>
</html>
""", height=480)

    # ── Micro-Environment stat chips (horizontal row below viewport) ──────────
    _chip_css = """
    <style>
    .me-row { display:flex; gap:14px; margin:10px 0 4px; }
    .me-chip {
        flex:1; border-radius:14px; padding:14px 18px;
        border:1px solid rgba(255,255,255,0.6);
        background:rgba(255,255,255,0.72); backdrop-filter:blur(10px);
        display:flex; flex-direction:column; gap:6px;
        border-left-width:4px; border-left-style:solid;
        box-shadow:0 2px 12px rgba(25,58,59,0.06);
    }
    .me-chip-label {
        font-size:10px; font-weight:700; text-transform:uppercase;
        letter-spacing:0.09em; color:#7a8b8c;
        display:flex; align-items:center; gap:7px;
    }
    .me-dot { width:8px; height:8px; border-radius:50%; }
    .me-val {
        font-family:'JetBrains Mono',monospace;
        font-size:30px; font-weight:700;
        letter-spacing:-0.04em; line-height:1; color:#193a3b;
    }
    .me-sub { font-size:10px; color:#9aa4ae; }
    </style>
    """
    st.markdown(_chip_css, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.markdown(f"""
        <div class="me-chip" style="border-left-color:#0f766e">
          <div class="me-chip-label">
            <div class="me-dot" style="background:#0f766e"></div>Residues
          </div>
          <div class="me-val">{n_res}</div>
          <div class="me-sub">Cα nodes in graph</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="me-chip" style="border-left-color:#d97706">
          <div class="me-chip-label">
            <div class="me-dot" style="background:#d97706"></div>Contacts
          </div>
          <div class="me-val">{n_con}</div>
          <div class="me-sub">mean {mean_e} Å</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="me-chip" style="border-left-color:#2563eb">
          <div class="me-chip-label">
            <div class="me-dot" style="background:#2563eb"></div>Target pKd
          </div>
          <div class="me-val">{pv_pkd}</div>
          <div class="me-sub">{selected_pdb.upper()} · PDBbind</div>
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
            c1.markdown(f"""
            <div class="metric-card teal" style="border-top:3px solid #0f766e;border-left:none;padding:1.4rem 1.6rem">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.8rem">
                  <div class="metric-label">Predicted pKd · {selected_pdb.upper()}</div>
                  <span style="font-size:0.68rem;background:rgba(255,255,255,0.8);border:1px solid rgba(0,0,0,0.06);padding:2px 8px;border-radius:6px;color:#7a8b8c">GNN Output</span>
                </div>
                <div style="font-family:ui-monospace,monospace;font-size:3rem;font-weight:700;letter-spacing:-0.04em;color:#0f766e;line-height:1">{pred_pkd:.2f}</div>
                <div class="metric-sub" style="margin-top:6px">vs pocket pKd {selected_pkd:.2f}</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""
            <div class="metric-card amber" style="border-top:3px solid #d97706;border-left:none;padding:1.4rem 1.6rem">
                <div class="metric-label" style="margin-bottom:0.8rem">Pose Quality</div>
                <div style="font-family:ui-monospace,monospace;font-size:3rem;font-weight:700;letter-spacing:-0.04em;color:#d97706;line-height:1">{pred_pose:.2f}</div>
                <div class="metric-sub" style="margin-top:6px">P(RMSD &lt; 2Å)</div>
            </div>""", unsafe_allow_html=True)
            delta_color = "#0f766e" if sel_color=="teal" else "#2563eb" if sel_color=="blue" else "#d97757"
            c3.markdown(f"""
            <div class="metric-card {sel_color}" style="border-top:3px solid {delta_color};border-left:none;padding:1.4rem 1.6rem">
                <div class="metric-label" style="margin-bottom:0.8rem">{sel_label}</div>
                <div style="font-family:ui-monospace,monospace;font-size:3rem;font-weight:700;letter-spacing:-0.04em;color:{delta_color};line-height:1">{sel_value}</div>
                <div class="metric-sub" style="margin-top:6px">{sel_sub}</div>
            </div>""", unsafe_allow_html=True)

            pkd_pct      = min(100, max(0, (pred_pkd  - 2) / 10 * 100))
            egfr_pct     = min(100, max(0, (egfr_pkd  - 2) / 10 * 100)) if not np.isnan(egfr_pkd) else 0
            egfr_display = f"{egfr_pkd:.2f}" if not np.isnan(egfr_pkd) else "—"

            st.markdown(f"""
            <div class="glass" style="padding:1.6rem 2rem;margin-top:0.5rem">
                <div class="section-label" style="margin-bottom:1.2rem">Physicochemical Profile</div>
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
                  "pkd": pred_pkd, "pose": pred_pose, "sel": float(delta_pkd)})

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RL Browser
# ═════════════════════════════════════════════════════════════════════════════
elif page == "RL Generator":
    from rdkit import Chem

    # ── Summary loaded first so pocket name is available for header ───────────
    _rl_summ_pre = _rl_summary()
    _ref_pocket  = _rl_summ_pre.get("ref_pocket", "6E9A")
    _ref_pkd     = _rl_summ_pre.get("ref_pkd", 0)
    _pocket_label = f"Pocket {_ref_pocket}"
    if _ref_pkd:
        _pocket_label += f" · pKd {_ref_pkd:.2f}"

    page_header(
        "RL Optimization",
        "Molecule Browser",
        "Explore generated candidates by reward, predicted affinity, and drug-likeness without losing the structures.",
        [
            ("REINFORCE", "amber"),
            ("GNN oracle", "blue"),
            (_pocket_label, "teal"),
            ("0.5 affinity + 0.2 QED + 0.2 SA + 0.1 MW", ""),
        ],
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    # Prefer JSON (live run results) over SQL seed data
    df = _rl_df()
    if df.empty:
        df = _db_query("""
            SELECT smiles, reward, r_affinity, r_qed, r_sa, r_mw, pred_pkd, step
            FROM dbo.rl_molecules
            WHERE experiment_id=(SELECT id FROM dbo.experiments WHERE name='rl_reinforce_egfr')
            ORDER BY reward DESC
        """)
    if "reward_total" in df.columns and "reward" not in df.columns:
        df = df.rename(columns={"reward_total":"reward"})

    if df.empty:
        st.info("No RL results — run Phase 3 notebook first.")
    else:
        display_df = df.copy()
        for col, default in [("reward", 0.0), ("pred_pkd", 0.0), ("r_qed", 0.0)]:
            if col not in display_df.columns:
                display_df[col] = default
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").fillna(default)

        st.markdown('<div class="section-label">Filter and rank generated molecules</div>', unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns([1.15, 1, 1, 1])
        sort_label = f1.selectbox(
            "Sort by",
            ["Reward", "Predicted pKd", "QED"],
            index=0,
            help="Rank generated molecules by the selected score.",
        )
        min_pkd = f2.slider("Minimum pKd", 0.0, 12.0, 0.0, 0.25)
        min_qed = f3.slider("Minimum QED", 0.0, 1.0, 0.0, 0.05)
        max_show = min(20, len(display_df))
        min_show = 1 if max_show < 4 else 4
        show_n = f4.slider("Show", min_show, max_show, min(10, max_show), 1)

        sort_col = {"Reward": "reward", "Predicted pKd": "pred_pkd", "QED": "r_qed"}[sort_label]
        filtered = display_df[
            (display_df["pred_pkd"] >= min_pkd) &
            (display_df["r_qed"] >= min_qed)
        ].sort_values(sort_col, ascending=False)

        if filtered.empty:
            st.markdown(
                '<div class="empty-state">No generated molecules match the current filters.</div>',
                unsafe_allow_html=True,
            )
            st.stop()

        top10 = filtered.head(show_n).reset_index(drop=True)
        st.caption(f"Showing {len(top10)} of {len(display_df)} generated molecules.")

        # ── Molecule card grid (2 columns) ────────────────────────────────────
        RANK_NUMS = ["①","②","③","④","⑤","⑥","⑦","⑧","⑨","⑩"]
        CARD_CSS = """
        <style>
        .mol-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.85) 0%, rgba(255,255,255,0.5) 100%);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.65);
            border-radius: 16px;
            padding: 8px 8px 18px 8px;
            box-shadow: 0 4px 20px rgba(25,58,59,0.06);
            transition: box-shadow 0.2s;
            margin-bottom: 1.2rem;
        }
        .mol-canvas {
            background: white;
            background-image: radial-gradient(#dde6e6 1.2px, transparent 1.2px);
            background-size: 18px 18px;
            border-radius: 10px;
            border: 1px solid #ece7de;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            min-height: 220px;
        }
        .mol-canvas svg { max-width: 100%; }
        .stat-badge {
            border-radius: 12px;
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .stat-badge .sb-label {
            font-size: 0.62rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .stat-badge .sb-value {
            font-family: ui-monospace, monospace;
            font-size: 1.15rem;
            font-weight: 700;
            line-height: 1;
        }
        .stat-teal  { background: linear-gradient(160deg,#e6f4f4,#f2fafa); border: 1px solid #cbe6e6; }
        .stat-teal  .sb-label { color: #257476; }
        .stat-teal  .sb-value { color: #0d4f50; }
        .stat-amber { background: linear-gradient(160deg,#fdf7ee,#fffaf3); border: 1px solid #f5e3cd; }
        .stat-amber .sb-label { color: #a16222; }
        .stat-amber .sb-value { color: #7c4815; }
        .stat-green { background: linear-gradient(160deg,#edf7f2,#f3f9f6); border: 1px solid #d3ecd8; }
        .stat-green .sb-label { color: #2b7c4d; }
        .stat-green .sb-value { color: #195a34; }
        .rank-badge {
            width: 28px; height: 28px;
            background: #115e59;
            color: white;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
            font-weight: 700;
            flex-shrink: 0;
        }
        </style>
        """
        st.markdown(CARD_CSS, unsafe_allow_html=True)

        col_a, col_b = st.columns(2, gap="large")
        for i, row in top10.iterrows():
            col = col_a if i % 2 == 0 else col_b
            smiles = str(row["smiles"])
            pkd_v  = row.get("pred_pkd", 0) or 0
            qed_v  = row.get("r_qed", 0) or 0
            rew_v  = row.get("reward", 0) or 0
            rank   = RANK_NUMS[i] if i < len(RANK_NUMS) else str(i+1)
            rank_bg = "#0a1f20" if i == 0 else "#115e59"

            svg_html, _ = mol_svg(smiles, w=380, h=220)
            struct_html = svg_html if svg_html else (
                f'<div style="font-family:monospace;font-size:0.7rem;color:#9aa4ae;padding:1rem;'
                f'word-break:break-all">{escape(smiles)}</div>'
            )

            smiles_short = smiles if len(smiles) <= 42 else smiles[:40] + "…"

            with col:
                st.markdown(f"""
                <div class="mol-card">
                  <div class="mol-canvas">{struct_html}</div>
                  <div style="padding:12px 6px 0 6px">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
                      <span class="rank-badge" style="background:{rank_bg}">{rank}</span>
                      <span style="font-family:ui-monospace,monospace;font-size:0.72rem;
                            color:#5f6b76;overflow:hidden;text-overflow:ellipsis;
                            white-space:nowrap;flex:1">{escape(smiles_short)}</span>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">
                      <div class="stat-badge stat-teal">
                        <span class="sb-label">pKd</span>
                        <span class="sb-value">{pkd_v:.2f}</span>
                      </div>
                      <div class="stat-badge stat-amber">
                        <span class="sb-label">QED</span>
                        <span class="sb-value">{qed_v:.3f}</span>
                      </div>
                      <div class="stat-badge stat-green">
                        <span class="sb-label">Reward</span>
                        <span class="sb-value">{rew_v:.3f}</span>
                      </div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — GNN vs Vina
# ═════════════════════════════════════════════════════════════════════════════
elif page == "GNN vs Vina":
    # ── pull top-20 RL molecules for pocket label ──────────────────────────
    _rl_pre   = _rl_df()
    if "reward_total" in _rl_pre.columns and "reward" not in _rl_pre.columns:
        _rl_pre = _rl_pre.rename(columns={"reward_total": "reward"})
    _rl_summ  = _rl_summary()
    _rl_pocket = _rl_summ.get("ref_pocket", "6E9A")
    _rl_n      = min(20, len(_rl_pre))

    page_header(
        "Benchmark View",
        "GNN Predictions vs AutoDock Vina",
        f"Compare top RL molecules across GNN pKd predictions and real AutoDock Vina docking scores.",
        [
            (f"Pocket {_rl_pocket}", "teal"),
            (f"Up to {_rl_n} RL molecules", "amber"),
            ("GNN oracle", "blue"),
            ("AutoDock Vina", ""),
        ],
    )

    df = _db_query("""
        SELECT smiles, pocket_pdb, vina_score, gnn_pred_pkd,
               ABS(vina_score - gnn_pred_pkd) AS abs_error
        FROM dbo.vina_benchmarks ORDER BY abs_error DESC
    """)
    if df.empty:
        # Use top-20 RL SMILES; run live GNN inference for real pKd values
        import torch

        max_dock = st.slider(
            "Molecules to dock with Vina",
            min_value=1,
            max_value=max(1, _rl_n),
            value=min(10, max(1, _rl_n)),
            help="Real docking is slower than the GNN oracle; keep this lower for quick UI iteration.",
        )
        exhaustiveness = st.slider(
            "Vina exhaustiveness",
            min_value=1,
            max_value=16,
            value=4,
            help="Higher values search more thoroughly and take longer.",
        )

        _rl_top = (_rl_pre.nlargest(max_dock, "reward")
                   .reset_index(drop=True) if not _rl_pre.empty else pd.DataFrame())

        _smiles_list = list(_rl_top["smiles"]) if not _rl_top.empty else []

        @st.cache_data(show_spinner="Running GNN inference on RL molecules…")
        def _vina_gnn_pkd(smiles_tuple):
            model, _ = _load_gnn()
            if model is None:
                return [None] * len(smiles_tuple)
            import torch
            model.eval()
            out = []
            with torch.no_grad():
                for smi in smiles_tuple:
                    g = _build_graph(smi, pocket_id=_rl_pocket.lower())
                    if g is None:
                        out.append(None)
                    else:
                        out.append(float(model(g)["affinity"].item()))
            return out

        _gnn_pkd = _vina_gnn_pkd(tuple(_smiles_list))

        # Filter out None (invalid SMILES / graph build failures)
        valid = [(s, p) for s, p in zip(_smiles_list, _gnn_pkd) if p is not None]
        if not valid:
            st.warning("No valid molecules to score.")
            st.stop()
        _smiles_list, _gnn_pkd = zip(*valid)

        @st.cache_data(show_spinner="Running AutoDock Vina docking…")
        def _vina_scores(smiles_tuple, pocket_pdb, exhaustiveness_value):
            from src.docking.vina_runner import dock_smiles_batch

            root = Path(__file__).parent.parent / "data"
            docked = dock_smiles_batch(
                smiles_tuple,
                pocket_pdb=pocket_pdb.lower(),
                data_root=root,
                exhaustiveness=int(exhaustiveness_value),
            )
            return [r.__dict__ for r in docked]

        _vina_results = _vina_scores(tuple(_smiles_list), _rl_pocket, exhaustiveness)
        _vina_by_smiles = {r["smiles"]: r for r in _vina_results}
        docked_rows = []
        failed = []
        for smi, pkd in zip(_smiles_list, _gnn_pkd):
            vr = _vina_by_smiles.get(smi, {})
            if vr.get("vina_score") is None:
                failed.append(vr.get("error", "Vina failed"))
                continue
            docked_rows.append({
                "smiles": smi,
                "pocket_pdb": _rl_pocket,
                "vina_score": float(vr["vina_score"]),
                "vina_energy_kcal_mol": float(vr["vina_energy_kcal_mol"]),
                "gnn_pred_pkd": float(pkd),
            })

        if not docked_rows:
            reason = failed[0] if failed else "No Vina scores returned."
            st.warning(f"AutoDock Vina comparison unavailable: {reason}")
            st.stop()

        df = pd.DataFrame({
            "smiles": [r["smiles"] for r in docked_rows],
            "pocket_pdb": [r["pocket_pdb"] for r in docked_rows],
            "vina_score": [r["vina_score"] for r in docked_rows],
            "vina_energy_kcal_mol": [r["vina_energy_kcal_mol"] for r in docked_rows],
            "gnn_pred_pkd": [r["gnn_pred_pkd"] for r in docked_rows],
        })
        df["abs_error"] = (df["vina_score"] - df["gnn_pred_pkd"]).abs()
        if failed:
            st.info(f"Docked {len(df)} molecules with AutoDock Vina; {len(failed)} failed during preparation or docking.")
        else:
            st.success(f"Docked {len(df)} molecules with AutoDock Vina.")

    if "vina_energy_kcal_mol" not in df.columns:
        df["vina_energy_kcal_mol"] = -df["vina_score"]

    from scipy.stats import pearsonr
    r = float(pearsonr(df["vina_score"], df["gnn_pred_pkd"])[0]) if len(df) >= 2 else float("nan")
    rmse  = float(np.sqrt(((df["vina_score"]-df["gnn_pred_pkd"])**2).mean()))
    n_good= int((df["abs_error"] < 1.0).sum())

    c1,c2,c3 = st.columns(3)
    r_color = "#15803d" if r>0.7 else "#d97706" if r>0.4 else "#b42318"
    r_display = f"{r:.3f}" if not np.isnan(r) else "—"
    c1.markdown(f"""<div class="metric-card teal">
        <div class="metric-label">Pearson r</div>
        <div class="metric-value" style="color:{r_color}">{r_display}</div>
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
            x=alt.X("vina_score:Q",   title="Vina affinity proxy (-kcal/mol)",
                    scale=alt.Scale(domain=[lmin,lmax]),
                    axis=alt.Axis(labelColor="#9aa4ae",titleColor="#9aa4ae",gridColor="#e8e0d5")),
            y=alt.Y("gnn_pred_pkd:Q", title="GNN Predicted pKd",
                    scale=alt.Scale(domain=[lmin,lmax]),
                    axis=alt.Axis(labelColor="#9aa4ae",titleColor="#9aa4ae",gridColor="#e8e0d5")),
            color=alt.Color("abs_error:Q", title="|Error|",
                            scale=alt.Scale(scheme="orangered"),
                            legend=alt.Legend(labelColor="#5f6b76",titleColor="#5f6b76")),
            tooltip=["pocket_pdb","vina_energy_kcal_mol","vina_score","gnn_pred_pkd","abs_error"],
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

    tdf = df.sort_values("abs_error", ascending=False).reset_index(drop=True)

    def _err_bar(v, maxv=3.0):
        pct = min(100, v / maxv * 100)
        color = "#b42318" if v > 1.5 else "#d97706" if v > 1.0 else "#15803d"
        return (f'<div style="display:flex;align-items:center;gap:8px">'
                f'<div style="flex:1;background:#ede8e1;border-radius:4px;height:6px;overflow:hidden">'
                f'<div style="width:{pct:.0f}%;height:100%;background:{color};border-radius:4px"></div></div>'
                f'<span style="font-family:ui-monospace,monospace;font-size:0.78rem;color:{color};'
                f'min-width:36px">{v:.2f}</span></div>')

    rows_html = ""
    for i, row in tdf.iterrows():
        bg = "rgba(255,255,255,0.55)" if i % 2 == 0 else "rgba(255,255,255,0.25)"
        pocket = row.get("pocket_pdb", "—")
        smi    = str(row.get("smiles", "—"))
        smi_short = smi if len(smi) <= 36 else smi[:34] + "…"
        vina   = f"{row['vina_score']:.2f}"
        gnn    = f"{row['gnn_pred_pkd']:.2f}"
        err    = float(row["abs_error"])
        err_cell = _err_bar(err)
        rows_html += f"""
        <tr style="background:{bg}">
          <td style="padding:9px 14px;font-family:ui-monospace,monospace;font-size:0.8rem;
                     color:#193a3b;font-weight:600">{i+1}</td>
          <td style="padding:9px 14px">
            <span style="background:#193a3b;color:#7efce1;font-size:0.68rem;
                  font-family:ui-monospace,monospace;padding:2px 8px;border-radius:4px;
                  font-weight:700">{escape(str(pocket))}</span>
          </td>
          <td style="padding:9px 14px;max-width:220px" title="{escape(smi)}">
            <span style="font-family:ui-monospace,monospace;font-size:0.75rem;color:#5f6b76;
                  display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{escape(smi_short)}</span>
          </td>
          <td style="padding:9px 14px;font-family:ui-monospace,monospace;font-size:0.85rem;
                     color:#0d4f50;font-weight:600">{vina}</td>
          <td style="padding:9px 14px;font-family:ui-monospace,monospace;font-size:0.85rem;
                     color:#193a3b;font-weight:600">{gnn}</td>
          <td style="padding:9px 14px;min-width:140px">{err_cell}</td>
        </tr>"""

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.5);backdrop-filter:blur(10px);
                border:1px solid rgba(255,255,255,0.6);border-radius:16px;
                overflow:hidden;box-shadow:0 4px 20px rgba(25,58,59,0.06)">
      <table style="width:100%;border-collapse:collapse">
        <thead>
          <tr style="background:rgba(25,58,59,0.06);border-bottom:1px solid #ddd8d0">
            <th style="padding:10px 14px;text-align:left;font-size:0.68rem;font-weight:700;
                       color:#7a8b8c;text-transform:uppercase;letter-spacing:0.08em">#</th>
            <th style="padding:10px 14px;text-align:left;font-size:0.68rem;font-weight:700;
                       color:#7a8b8c;text-transform:uppercase;letter-spacing:0.08em">Pocket</th>
            <th style="padding:10px 14px;text-align:left;font-size:0.68rem;font-weight:700;
                       color:#7a8b8c;text-transform:uppercase;letter-spacing:0.08em">Ligand SMILES</th>
            <th style="padding:10px 14px;text-align:left;font-size:0.68rem;font-weight:700;
                       color:#7a8b8c;text-transform:uppercase;letter-spacing:0.08em">Vina Score</th>
            <th style="padding:10px 14px;text-align:left;font-size:0.68rem;font-weight:700;
                       color:#7a8b8c;text-transform:uppercase;letter-spacing:0.08em">GNN pKd</th>
            <th style="padding:10px 14px;text-align:left;font-size:0.68rem;font-weight:700;
                       color:#7a8b8c;text-transform:uppercase;letter-spacing:0.08em">|Error|</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SQL Console
# ═════════════════════════════════════════════════════════════════════════════
elif page == "SQL Console":
    db_chip = "SQL Server connected" if _db_configured() else "Local demo mode"
    page_header(
        "Data Access",
        "SQL Console",
        "Run read-only queries against the experiment database, then inspect or export the result set.",
        [
            (db_chip, "teal" if _db_configured() else "amber"),
            ("SELECT only", "blue"),
        ],
    )

    QUERY_TEMPLATES = {
        "Top RL molecules": """SELECT TOP 20
    e.name        AS experiment,
    rl.smiles,
    rl.reward,
    rl.pred_pkd,
    rl.r_qed,
    rl.r_sa
FROM dbo.rl_molecules rl
JOIN dbo.experiments e ON e.id = rl.experiment_id
ORDER BY rl.reward DESC""",
        "Recent binding predictions": """SELECT TOP 20
    smiles,
    pocket_pdb,
    pred_pkd,
    pred_pose_prob,
    pred_select_prob,
    created_at
FROM dbo.binding_predictions
ORDER BY created_at DESC""",
        "Vina benchmark errors": """SELECT TOP 20
    smiles,
    pocket_pdb,
    vina_score,
    gnn_pred_pkd,
    ABS(vina_score - gnn_pred_pkd) AS abs_error
FROM dbo.vina_benchmarks
ORDER BY abs_error DESC""",
    }

    t1, t2 = st.columns([1.25, 2.75])
    template_name = t1.selectbox("Query template", list(QUERY_TEMPLATES.keys()))
    t2.markdown(
        '<div class="empty-state" style="padding:0.85rem 1rem">'
        'Queries are intentionally read-only in the UI. In local mode, the database calls return empty demo results.'
        '</div>',
        unsafe_allow_html=True,
    )

    sql = st.text_area("SQL Query", value=QUERY_TEMPLATES[template_name], height=230)

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
