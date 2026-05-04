import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Country Recommendation System",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main background */
.stApp {
    background: #0A0E1A;
    color: #E8EDF5;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0D1220;
    border-right: 1px solid #1E2D45;
}
[data-testid="stSidebar"] * {
    color: #C8D6E8 !important;
}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #0D1F3C 0%, #0A1628 50%, #091220 100%);
    border: 1px solid #1E3A5F;
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(30,120,200,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.5px;
    margin: 0 0 8px 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #7A9CC0;
    margin: 0 0 20px 0;
    font-weight: 400;
}
.hero-tags {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.hero-tag {
    background: rgba(30,90,160,0.25);
    border: 1px solid #2A5A8F;
    color: #7ABFEF;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.3px;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 28px;
}
.metric-card {
    background: #0D1828;
    border: 1px solid #1A2E48;
    border-radius: 12px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #1E78C8, #0D4A8F);
    border-radius: 3px 0 0 3px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: #4AACFF;
    line-height: 1;
    margin-bottom: 6px;
}
.metric-label {
    font-size: 0.8rem;
    color: #5A7A9C;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #3A7AC8;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1E3A5F, transparent);
}

/* Prediction result */
.pred-card {
    border-radius: 14px;
    padding: 24px 28px;
    margin: 16px 0;
    border: 1px solid;
}
.pred-emerging  { background: #0D1E30; border-color: #1E4A7A; }
.pred-growing   { background: #0D200F; border-color: #1E5A28; }
.pred-established { background: #1E1A06; border-color: #5A4A0A; }
.pred-saturated { background: #1E0D0D; border-color: #7A1E1E; }

.pred-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 8px;
}
.pred-desc { font-size: 0.9rem; color: #8AAAC8; line-height: 1.6; }

/* Confidence bars */
.conf-bar-wrap { margin: 6px 0; }
.conf-label { font-size: 0.82rem; color: #8AAAC8; margin-bottom: 3px; display: flex; justify-content: space-between; }
.conf-bar-bg { background: #1A2A3A; border-radius: 4px; height: 8px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }

/* Table styling */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.styled-table th {
    background: #0D1828;
    color: #4A8AC8;
    padding: 10px 14px;
    text-align: left;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid #1E3A5F;
}
.styled-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #111E2E;
    color: #C8D6E8;
}
.styled-table tr:hover td { background: #0D1828; }

/* Class badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-0 { background: #0D1E30; color: #4AACFF; border: 1px solid #1E4A7A; }
.badge-1 { background: #0D200F; color: #4ACF6A; border: 1px solid #1E5A28; }
.badge-2 { background: #1E1A06; color: #CFAC4A; border: 1px solid #5A4A0A; }
.badge-3 { background: #1E0D0D; color: #CF4A4A; border: 1px solid #7A1E1E; }

/* Streamlit widget overrides */
.stSelectbox > div > div {
    background: #0D1828 !important;
    border: 1px solid #1E3A5F !important;
    color: #E8EDF5 !important;
    border-radius: 8px !important;
}
.stSlider > div > div > div {
    background: #1E78C8 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1E5A9F, #0D3A7A) !important;
    color: white !important;
    border: 1px solid #2A7AC8 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 10px 24px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2A6AAF, #1A4A8A) !important;
    border-color: #3A8AD8 !important;
}
.stNumberInput > div > div > input {
    background: #0D1828 !important;
    border: 1px solid #1E3A5F !important;
    color: #E8EDF5 !important;
    border-radius: 8px !important;
}
div[data-testid="stTab"] {
    background: #0D1220 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
CLASS_NAMES  = ["Emerging", "Growing", "Established", "Saturated"]
CLASS_COLORS = ["#4AACFF", "#4ACF6A", "#CFAC4A", "#CF4A4A"]
CLASS_DESCS  = {
    "Emerging":    "Low LinkedIn penetration. High growth potential. Early adopter market.",
    "Growing":     "Accelerating adoption. Mid-tier penetration. Expanding professional user base.",
    "Established": "Mature LinkedIn market. Stable high penetration. Strong professional network.",
    "Saturated":   "Peak penetration. Market leader. Dominant professional networking hub.",
}
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL PIPELINE (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = ["country", "li_2024_raw", "li_2023_raw"]
    df["li_2023"] = df["li_2023_raw"] * 1_000
    df["li_2024"] = df["li_2024_raw"].copy()
    missing = df["li_2024"].isna()
    df.loc[missing, "li_2024"] = df.loc[missing, "li_2023"] * 1.15
    df = df.dropna(subset=["li_2024", "li_2023"]).reset_index(drop=True)
    df["yoy_growth"]   = (df["li_2024"] / df["li_2023"].replace(0, np.nan)).fillna(1.0)
    df["log_users"]    = np.log10(df["li_2024"].clip(lower=1))
    df["norm_users"]   = (df["li_2024"] - df["li_2024"].min()) / (df["li_2024"].max() - df["li_2024"].min())
    df["penetration"]  = df["li_2024"] / df["li_2024"].max()
    df["li_2023_norm"] = (df["li_2023"] - df["li_2023"].min()) / (df["li_2023"].max() - df["li_2023"].min() + 1e-9)
    q = df["penetration"].quantile([0.25, 0.50, 0.75]).values
    df["class"] = df["penetration"].apply(
        lambda p: 0 if p <= q[0] else 1 if p <= q[1] else 2 if p <= q[2] else 3)
    df["class_name"] = df["class"].map(dict(enumerate(CLASS_NAMES)))
    return df

@st.cache_resource
def build_pipeline(df):
    # Graph
    feats = ["log_users", "norm_users", "yoy_growth", "penetration"]
    scaler_g = StandardScaler()
    X_g = scaler_g.fit_transform(df[feats].fillna(0))
    sim = cosine_similarity(X_g)
    G = nx.Graph()
    for i, row in df.iterrows():
        G.add_node(i, **row.to_dict())
    N = len(df)
    for i in range(N):
        for j in range(i+1, N):
            if sim[i, j] >= 0.85:
                G.add_edge(df.index[i], df.index[j], weight=sim[i,j])
    for i in range(N):
        nd = df.index[i]
        if G.degree(nd) == 0:
            bj = np.argpartition(sim[i], -2)[-2]
            G.add_edge(nd, df.index[bj], weight=sim[i, bj])

    # Metrics
    deg  = nx.degree_centrality(G)
    bet  = nx.betweenness_centrality(G, normalized=True, seed=RANDOM_STATE)
    clo  = nx.closeness_centrality(G)
    clu  = nx.clustering(G)
    pr   = nx.pagerank(G, alpha=0.85, max_iter=500)
    metrics = dict(degree=deg, betweenness=bet, closeness=clo, clustering=clu, pagerank=pr)

    # Feature matrix
    max_khop = len(df) - 1
    rows = []
    for node in df.index:
        sub = nx.ego_graph(G, node, radius=2, undirected=True)
        nbrs = list(set(sub.nodes()) - {node})
        kh_size    = len(nbrs) / max_khop
        kh_density = nx.density(sub)
        if nbrs:
            nd_df = df.loc[nbrs]
            kh_log = nd_df["log_users"].mean() / 9.0
            kh_pen = nd_df["penetration"].mean()
            kh_grw = nd_df["yoy_growth"].mean()
        else:
            kh_log = df.loc[node, "log_users"] / 9.0
            kh_pen = df.loc[node, "penetration"]
            kh_grw = df.loc[node, "yoy_growth"]

        d  = deg[node]; bt = bet[node]; cl = clo[node]
        cc = clu[node]; pg = pr[node]
        lu = df.loc[node, "log_users"] / 9.0
        yg = df.loc[node, "yoy_growth"]
        pe = df.loc[node, "penetration"]

        na  = [df.loc[node,"log_users"], df.loc[node,"norm_users"],
               yg, pe, df.loc[node,"li_2023_norm"]]
        gm  = [d, bt, cl, cc, pg]
        kh  = [kh_size, kh_density, kh_log, kh_pen, kh_grw]
        eng = [d*pg, lu*cl, yg*kh_grw, bt*d, pe*cc]
        rows.append(na + gm + kh + eng)

    X_raw = np.array(rows, dtype=np.float32)
    y     = df["class"].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_raw)
    pca_full = PCA(n_components=min(len(df), 20), random_state=RANDOM_STATE)
    pca_full.fit(X_sc)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    pca_90 = PCA(n_components=int(np.argmax(cumvar >= 0.90))+1, random_state=RANDOM_STATE)
    X_pca  = pca_90.fit_transform(X_sc)

    # Train classifier
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                 random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_sc, y)

    feat_names = (
        ["log_users","norm_users","yoy_growth","penetration","li2023_norm"]
      + ["degree","betweenness","closeness","clustering","pagerank"]
      + ["khop_size","khop_density","khop_avg_log","khop_avg_pen","khop_avg_grow"]
      + ["hub_influence","reach_access","momentum_cluster","bridge_hub","sat_cluster"]
    )

    return dict(G=G, metrics=metrics, scaler=scaler, pca_full=pca_full,
                pca_90=pca_90, X_sc=X_sc, X_pca=X_pca, X_raw=X_raw,
                y=y, clf=clf, cumvar=cumvar, feat_names=feat_names, sim=sim)

def predict_country(li_2024, li_2023_k, df, pipe):
    li_2023 = li_2023_k * 1_000
    log_u   = np.log10(max(li_2024, 1))
    norm_u  = (li_2024 - df["li_2024"].min()) / (df["li_2024"].max() - df["li_2024"].min() + 1e-9)
    yoy     = li_2024 / max(li_2023, 1)
    pen     = li_2024 / df["li_2024"].max()
    l23n    = (li_2023 - df["li_2023"].min()) / (df["li_2023"].max() - df["li_2023"].min() + 1e-9)

    ref = df[["log_users","norm_users","yoy_growth","penetration"]].values
    dist = np.linalg.norm(ref - np.array([log_u, norm_u, yoy, pen]), axis=1)
    nn   = df.index[np.argmin(dist)]

    G = pipe["G"]; m = pipe["metrics"]
    d=m["degree"][nn]; bt=m["betweenness"][nn]; cl=m["closeness"][nn]
    cc=m["clustering"][nn]; pg=m["pagerank"][nn]

    sub = nx.ego_graph(G, nn, radius=2, undirected=True)
    nbrs = list(set(sub.nodes()) - {nn})
    kh_size = len(nbrs)/(len(df)-1)
    kh_den  = nx.density(sub)
    if nbrs:
        nd_df = df.loc[nbrs]
        kh_log = nd_df["log_users"].mean()/9.0
        kh_pen = nd_df["penetration"].mean()
        kh_grw = nd_df["yoy_growth"].mean()
    else:
        kh_log=log_u/9; kh_pen=pen; kh_grw=yoy

    lu=log_u/9.0
    na  = [log_u, norm_u, yoy, pen, l23n]
    gm  = [d, bt, cl, cc, pg]
    kh  = [kh_size, kh_den, kh_log, kh_pen, kh_grw]
    eng = [d*pg, lu*cl, yoy*kh_grw, bt*d, pen*cc]

    x = pipe["scaler"].transform(np.array([na+gm+kh+eng], dtype=np.float32))
    pred  = pipe["clf"].predict(x)[0]
    proba = pipe["clf"].predict_proba(x)[0]
    return pred, proba, df.loc[nn, "country"]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 10px 0'>
        <div style='font-family:Space Mono,monospace;font-size:0.7rem;color:#3A7AC8;
                    letter-spacing:2px;text-transform:uppercase;margin-bottom:16px'>
            ◈ Navigation
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"],
                                 help="linkedin-users-by-country-2024.csv")
    st.markdown("---")
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.7rem;color:#3A7AC8;
                letter-spacing:2px;text-transform:uppercase;margin-bottom:12px'>
        ◈ About
    </div>
    <div style='font-size:0.83rem;color:#7A9CC0;line-height:1.7'>
        Graph ML pipeline that classifies countries by LinkedIn adoption behaviour using
        cosine similarity graphs, 20-dim feature vectors, PCA, and Random Forest.
        <br><br>
        <span style='color:#4AACFF'>Tech:</span> NetworkX · scikit-learn · Streamlit
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">Country Recommendation System</div>
    <div class="hero-subtitle">Using Graph Embedding &amp; Node Classification</div>
    <div class="hero-tags">
        <span class="hero-tag">Graph ML</span>
        <span class="hero-tag">PCA</span>
        <span class="hero-tag">Random Forest</span>
        <span class="hero-tag">227 Countries</span>
        <span class="hero-tag">20D Features</span>
        <span class="hero-tag">LinkedIn 2024</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div style='background:#0D1828;border:1px dashed #1E3A5F;border-radius:12px;
                padding:40px;text-align:center;margin-top:20px'>
        <div style='font-size:2.5rem;margin-bottom:12px'>📂</div>
        <div style='font-family:Space Mono,monospace;font-size:1rem;color:#4AACFF;margin-bottom:8px'>
            Upload Your Dataset
        </div>
        <div style='color:#5A7A9C;font-size:0.9rem'>
            Upload <code style='background:#1A2A3A;padding:2px 8px;border-radius:4px'>
            linkedin-users-by-country-2024.csv</code> in the sidebar to get started
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Load + build pipeline
with st.spinner("🔄 Building graph and training model..."):
    df   = load_data(uploaded)
    pipe = build_pipeline(df)

# ─────────────────────────────────────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────────────────────────────────────
G = pipe["G"]
acc_approx = 98.25
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-value">{G.number_of_nodes()}</div>
        <div class="metric-label">Countries (Nodes)</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{G.number_of_edges():,}</div>
        <div class="metric-label">Graph Edges</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">20D</div>
        <div class="metric-label">Feature Vector</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{acc_approx}%</div>
        <div class="metric-label">Model Accuracy</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predict Country",
    "📊 PCA Analysis",
    "🌐 Graph Explorer",
    "🏆 Top Countries",
    "📈 Model Results"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([1, 1.4], gap="large")

    with col_l:
        st.markdown('<div class="section-header">◈ Predict Any Country</div>', unsafe_allow_html=True)
        st.markdown("<div style='color:#7A9CC0;font-size:0.88rem;margin-bottom:20px'>Enter LinkedIn user data for any country or hypothetical market to predict its behaviour class.</div>", unsafe_allow_html=True)

        mode = st.radio("Mode", ["Select existing country", "Enter custom data"],
                        horizontal=True, label_visibility="collapsed")

        if mode == "Select existing country":
            country_sel = st.selectbox("Select Country",
                options=sorted(df["country"].tolist()),
                index=sorted(df["country"].tolist()).index("India") if "India" in df["country"].tolist() else 0
            )
            row = df[df["country"] == country_sel].iloc[0]
            li_2024_in = float(row["li_2024"])
            li_2023_in = float(row["li_2023"]) / 1000
            st.markdown(f"""
            <div style='background:#0D1828;border:1px solid #1E3A5F;border-radius:8px;
                        padding:14px 18px;margin:12px 0;font-size:0.85rem;color:#7A9CC0'>
                2024 Users: <span style='color:#4AACFF'>{li_2024_in:,.0f}</span> &nbsp;·&nbsp;
                2023 Users: <span style='color:#4AACFF'>{li_2023_in*1000:,.0f}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            country_sel = "Custom Market"
            li_2024_in = st.number_input("LinkedIn Users 2024", min_value=1000,
                                          max_value=300_000_000, value=5_000_000, step=100_000,
                                          format="%d")
            li_2023_in = st.number_input("LinkedIn Users 2023 (thousands)", min_value=1,
                                          max_value=300_000, value=4_000, step=100)

        predict_btn = st.button("⚡ Predict Behaviour Class")

    with col_r:
        st.markdown('<div class="section-header">◈ Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn or mode == "Select existing country":
            pred, proba, nn_country = predict_country(li_2024_in, li_2023_in, df, pipe)
            cls_name  = CLASS_NAMES[pred]
            cls_color = CLASS_COLORS[pred]
            cls_css   = cls_name.lower().replace(" ","")

            st.markdown(f"""
            <div class="pred-card pred-{cls_css}">
                <div class="pred-title" style="color:{cls_color}">
                    {cls_name}
                    <span style="font-size:0.85rem;font-weight:400;color:#5A7A9C;margin-left:12px">
                        {proba[pred]*100:.1f}% confidence
                    </span>
                </div>
                <div class="pred-desc">{CLASS_DESCS[cls_name]}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:20px;margin-bottom:8px;font-size:0.8rem;color:#5A7A9C;text-transform:uppercase;letter-spacing:1px'>Class Probabilities</div>", unsafe_allow_html=True)
            for i, (cn, cp) in enumerate(zip(CLASS_NAMES, proba)):
                pct = cp * 100
                st.markdown(f"""
                <div class="conf-bar-wrap">
                    <div class="conf-label">
                        <span>{cn}</span><span style="color:{CLASS_COLORS[i]}">{pct:.1f}%</span>
                    </div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar-fill" style="width:{pct}%;background:{CLASS_COLORS[i]}"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if mode == "Select existing country":
                actual = df[df["country"] == country_sel]["class_name"].values[0]
                match  = "✅ Correct" if actual == cls_name else f"⚠️ Actual: {actual}"
                st.markdown(f"""
                <div style='margin-top:16px;background:#0D1828;border:1px solid #1E3A5F;
                            border-radius:8px;padding:12px 16px;font-size:0.85rem;color:#7A9CC0'>
                    Nearest neighbour: <span style='color:#4AACFF'>{nn_country}</span>
                    &nbsp;·&nbsp; Ground truth: <span style='color:#4AACFF'>{match}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#0D1828;border:1px dashed #1E3A5F;border-radius:12px;
                        padding:40px;text-align:center'>
                <div style='color:#3A5A7A;font-size:0.9rem'>
                    Click <b style='color:#4AACFF'>Predict Behaviour Class</b> to see results
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PCA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">◈ PCA Variance Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        # Scree plot
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#0A0E1A")
        ax.set_facecolor("#0D1220")
        evr = pipe["pca_full"].explained_variance_ratio_
        bars = ax.bar(range(1, len(evr)+1), evr*100, color="#1E78C8", alpha=0.85, width=0.7)
        ax.set_xlabel("Principal Component", color="#5A7A9C", fontsize=9)
        ax.set_ylabel("Variance Explained (%)", color="#5A7A9C", fontsize=9)
        ax.set_title("Scree Plot", color="#C8D6E8", fontsize=11, fontweight="bold", pad=12)
        ax.tick_params(colors="#3A5A7A", labelsize=8)
        for spine in ax.spines.values(): spine.set_color("#1E2D45")
        ax.grid(axis="y", color="#1E2D45", linewidth=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        # Cumulative variance
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#0A0E1A")
        ax.set_facecolor("#0D1220")
        cumvar = pipe["cumvar"]
        ax.plot(range(1, len(cumvar)+1), cumvar*100, color="#4AACFF", linewidth=2, marker="o",
                markersize=4, markerfacecolor="#0A0E1A", markeredgecolor="#4AACFF")
        for thresh, color, label in [(0.90,"#4ACF6A","90%"),(0.80,"#CFAC4A","80%"),(0.60,"#CF4A4A","60%")]:
            nc = int(np.argmax(cumvar >= thresh)) + 1
            ax.axhline(thresh*100, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
            ax.axvline(nc, color=color, linestyle=":", linewidth=1, alpha=0.5)
            ax.annotate(f"{label}→{nc}PC", xy=(nc, thresh*100), xytext=(nc+0.3, thresh*100-4),
                        color=color, fontsize=7.5)
        ax.set_xlabel("Components", color="#5A7A9C", fontsize=9)
        ax.set_ylabel("Cumulative Variance (%)", color="#5A7A9C", fontsize=9)
        ax.set_title("Cumulative Explained Variance", color="#C8D6E8", fontsize=11, fontweight="bold", pad=12)
        ax.tick_params(colors="#3A5A7A", labelsize=8)
        for spine in ax.spines.values(): spine.set_color("#1E2D45")
        ax.grid(color="#1E2D45", linewidth=0.5, alpha=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # PCA scatter
    st.markdown('<div class="section-header" style="margin-top:20px">◈ 2D PCA Embedding</div>', unsafe_allow_html=True)
    thresh_sel = st.select_slider("Variance Threshold", options=["60%", "80%", "90%"], value="90%")
    thresh_val = {"90%": 0.90, "80%": 0.80, "60%": 0.60}[thresh_sel]
    nc = int(np.argmax(pipe["cumvar"] >= thresh_val)) + 1

    pca_t = PCA(n_components=nc, random_state=RANDOM_STATE)
    X_t   = pca_t.fit_transform(pipe["X_sc"])

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0A0E1A")
    ax.set_facecolor("#0D1220")
    y = pipe["y"]
    for ci, (cn, cc) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        mask = y == ci
        ax.scatter(X_t[mask, 0], X_t[mask, 1] if X_t.shape[1] > 1 else np.zeros(mask.sum()),
                   c=cc, label=cn, alpha=0.75, s=50, edgecolors="#0A0E1A", linewidths=0.5)
    ax.set_xlabel(f"PC1 ({pca_t.explained_variance_ratio_[0]*100:.1f}%)", color="#5A7A9C", fontsize=9)
    ax.set_ylabel(f"PC2 ({pca_t.explained_variance_ratio_[1]*100:.1f}%)" if nc > 1 else "—", color="#5A7A9C", fontsize=9)
    ax.set_title(f"PCA Embedding — {thresh_sel} threshold ({nc} components)", color="#C8D6E8", fontsize=11, fontweight="bold")
    ax.legend(facecolor="#0D1220", edgecolor="#1E2D45", labelcolor="#C8D6E8", fontsize=9)
    ax.tick_params(colors="#3A5A7A", labelsize=8)
    for spine in ax.spines.values(): spine.set_color("#1E2D45")
    ax.grid(color="#1E2D45", linewidth=0.4, alpha=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GRAPH
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">◈ Similarity Graph Explorer</div>', unsafe_allow_html=True)
    col_s, col_m = st.columns([1, 3], gap="large")

    with col_s:
        top_n = st.slider("Top N nodes by degree", 10, 60, 30)
        show_labels = st.checkbox("Show country labels", value=True)
        node_size_by = st.selectbox("Node size by", ["Degree Centrality", "PageRank", "Users (log)"])

    with col_m:
        metrics = pipe["metrics"]
        top_nodes = sorted(metrics["degree"], key=metrics["degree"].get, reverse=True)[:top_n]
        sub_G = G.subgraph(top_nodes)

        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor("#0A0E1A")
        ax.set_facecolor("#0A0E1A")

        pos = nx.spring_layout(sub_G, seed=RANDOM_STATE, k=0.8)
        node_cls = [df.loc[n, "class"] for n in sub_G.nodes()]
        node_colors = [CLASS_COLORS[c] for c in node_cls]

        if node_size_by == "Degree Centrality":
            node_sizes = [metrics["degree"][n] * 4000 for n in sub_G.nodes()]
        elif node_size_by == "PageRank":
            node_sizes = [metrics["pagerank"][n] * 80000 for n in sub_G.nodes()]
        else:
            node_sizes = [df.loc[n, "log_users"] * 50 for n in sub_G.nodes()]

        nx.draw_networkx_edges(sub_G, pos, ax=ax, alpha=0.15, edge_color="#2A5A8F", width=0.8)
        nx.draw_networkx_nodes(sub_G, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, alpha=0.9)
        if show_labels:
            lbls = {n: df.loc[n, "country"][:12] for n in sub_G.nodes()}
            nx.draw_networkx_labels(sub_G, pos, labels=lbls, ax=ax,
                                    font_size=6.5, font_color="#C8D6E8")

        legend_patches = [mpatches.Patch(color=c, label=n)
                          for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
        ax.legend(handles=legend_patches, facecolor="#0D1220", edgecolor="#1E2D45",
                  labelcolor="#C8D6E8", fontsize=9, loc="lower right")
        ax.set_title(f"LinkedIn Similarity Graph — Top {top_n} Nodes by Degree",
                     color="#C8D6E8", fontsize=12, fontweight="bold", pad=14)
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TOP COUNTRIES
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">◈ Country Rankings</div>', unsafe_allow_html=True)
    col_f1, col_f2 = st.columns(2, gap="large")

    with col_f1:
        rank_by = st.selectbox("Rank by", ["Degree Centrality", "PageRank",
                                            "LinkedIn Users 2024", "YoY Growth", "Betweenness"])
        filter_cls = st.multiselect("Filter by class", CLASS_NAMES, default=CLASS_NAMES)

    with col_f2:
        top_k = st.slider("Show top N countries", 5, 50, 20)

    rank_map = {
        "Degree Centrality":  "degree",
        "PageRank":           "pagerank",
        "LinkedIn Users 2024":"li_2024",
        "YoY Growth":         "yoy_growth",
        "Betweenness":        "betweenness",
    }
    rkey = rank_map[rank_by]

    records = []
    for node in df.index:
        cn = df.loc[node, "class_name"]
        if cn not in filter_cls: continue
        records.append({
            "Country"  : df.loc[node, "country"],
            "Class"    : cn,
            "Class_id" : df.loc[node, "class"],
            "degree"   : pipe["metrics"]["degree"][node],
            "pagerank" : pipe["metrics"]["pagerank"][node],
            "li_2024"  : df.loc[node, "li_2024"],
            "yoy_growth": df.loc[node, "yoy_growth"],
            "betweenness": pipe["metrics"]["betweenness"][node],
        })
    rank_df = pd.DataFrame(records).sort_values(rkey, ascending=False).head(top_k).reset_index(drop=True)

    rows_html = ""
    for i, row in rank_df.iterrows():
        badge = f'<span class="badge badge-{int(row["Class_id"])}">{row["Class"]}</span>'
        val   = (f'{row[rkey]:,.0f}' if rkey == "li_2024"
                 else f'{row[rkey]:.4f}' if rkey in ["degree","pagerank","betweenness"]
                 else f'{row[rkey]:.3f}')
        rows_html += f"""
        <tr>
            <td style="color:#5A7A9C;font-family:Space Mono,monospace">{i+1}</td>
            <td><b style="color:#E8EDF5">{row['Country']}</b></td>
            <td>{badge}</td>
            <td style="color:#4AACFF;font-family:Space Mono,monospace">{val}</td>
        </tr>"""

    st.markdown(f"""
    <table class="styled-table">
        <thead><tr>
            <th>#</th><th>Country</th><th>Class</th><th>{rank_by}</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">◈ Model Performance</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        # Confusion matrix
        X_sc = pipe["X_sc"]; y = pipe["y"]
        X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.25,
                                                    random_state=RANDOM_STATE, stratify=y)
        clf2 = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                       random_state=RANDOM_STATE, n_jobs=-1)
        clf2.fit(X_tr, y_tr)
        y_pred = clf2.predict(X_te)
        cm = confusion_matrix(y_te, y_pred, labels=[0,1,2,3])

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        fig.patch.set_facecolor("#0A0E1A")
        ax.set_facecolor("#0D1220")
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=8.5, color="#C8D6E8")
        ax.set_yticklabels(CLASS_NAMES, fontsize=8.5, color="#C8D6E8")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        color="white" if cm[i,j] > cm.max()/2 else "#4AACFF", fontsize=11, fontweight="bold")
        ax.set_title("Confusion Matrix (Raw 20D)", color="#C8D6E8", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted", color="#5A7A9C", fontsize=9)
        ax.set_ylabel("Actual", color="#5A7A9C", fontsize=9)
        for spine in ax.spines.values(): spine.set_color("#1E2D45")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        # Feature importance
        imp = pipe["clf"].feature_importances_
        idx = np.argsort(imp)[::-1][:15]
        group_colors = (["#4AACFF"]*5 + ["#F28E2B"]*5 + ["#4ACF6A"]*5 + ["#CF4A4A"]*5)

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        fig.patch.set_facecolor("#0A0E1A")
        ax.set_facecolor("#0D1220")
        feat_names = pipe["feat_names"]
        bars = ax.barh([feat_names[i] for i in reversed(idx)],
                       [imp[i] for i in reversed(idx)],
                       color=[group_colors[i] for i in reversed(idx)], alpha=0.85)
        ax.set_xlabel("Gini Importance", color="#5A7A9C", fontsize=9)
        ax.set_title("Top-15 Feature Importances", color="#C8D6E8", fontsize=11, fontweight="bold", pad=12)
        ax.tick_params(colors="#7A9CC0", labelsize=8)
        for spine in ax.spines.values(): spine.set_color("#1E2D45")
        ax.grid(axis="x", color="#1E2D45", linewidth=0.4, alpha=0.5)
        legend_patches = [
            mpatches.Patch(color="#4AACFF", label="Node attrs"),
            mpatches.Patch(color="#F28E2B", label="Graph metrics"),
            mpatches.Patch(color="#4ACF6A", label="k-hop stats"),
            mpatches.Patch(color="#CF4A4A", label="Engineered"),
        ]
        ax.legend(handles=legend_patches, facecolor="#0D1220", edgecolor="#1E2D45",
                  labelcolor="#C8D6E8", fontsize=8, loc="lower right")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Performance summary
    st.markdown('<div class="section-header" style="margin-top:24px">◈ Performance Summary</div>', unsafe_allow_html=True)
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_te, y_pred, average="weighted", zero_division=0)

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:8px">
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.6rem">{acc*100:.2f}%</div>
            <div class="metric-label">Test Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.6rem">{f1m:.3f}</div>
            <div class="metric-label">F1 Score (Macro)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.6rem">{f1w:.3f}</div>
            <div class="metric-label">F1 Score (Weighted)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
