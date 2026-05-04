"""
=============================================================================
Course: Big Data Analytics (BDA)
Author: Aman (25MA60R31)
Group Members: [Add if any]
Instructor: Prof. Arindham Banerjee
=============================================================================

DESIGN CHOICES (see Section 0):
  - Similarity graph via cosine similarity threshold (avoids KNN bias)
  - 20-dim feature vector: 5 node attrs + 5 graph metrics + 5 k-hop stats + 5 engineered
  - 4 behavior classes assigned by user penetration rate (LinkedIn reach %)
  - PCA at 90/80/60% variance thresholds — compared on classifier accuracy
  - Random Forest chosen for robustness to correlated features and feature importance
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 ▸ INSTALL DEPENDENCIES (Colab-compatible)
# ─────────────────────────────────────────────────────────────────────────────
# Run this cell in Colab first:
# !pip install networkx scikit-learn pandas numpy matplotlib seaborn --quiet

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ▸ IMPORTS & GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)
from sklearn.pipeline import Pipeline

import itertools, io, os, sys, textwrap
from collections import defaultdict

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
PALETTE = sns.color_palette("husl", 4)

print("✅ Section 1 complete — imports loaded")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ▸ DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load the LinkedIn CSV and engineer the features we will use as
    node attributes.

    Raw columns:
      country                        – node identifier
      LinkedInUsers_Total_2024       – absolute user count (some NaN)
      LinkedInUsersInThousands2023   – prior-year count in thousands

    Engineered columns:
      li_2024        – 2024 users (NaN rows imputed from 2023 × 1.15 growth)
      li_2023        – 2023 users (thousands → absolute)
      yoy_growth     – year-on-year growth ratio
      log_users      – log10 of li_2024 (compresses 4-order-of-magnitude range)
      norm_users     – min-max normalised li_2024
      penetration    – proxy = li_2024 / li_2024.max()  (no population data)
    """
    df = pd.read_csv(filepath)
    df.columns = ["country", "li_2024_raw", "li_2023_raw"]

    # Convert 2023 thousands → absolute
    df["li_2023"] = df["li_2023_raw"] * 1_000

    # Impute missing 2024 values with 2023 × assumed 15 % YoY growth
    df["li_2024"] = df["li_2024_raw"].copy()
    missing_mask = df["li_2024"].isna()
    df.loc[missing_mask, "li_2024"] = df.loc[missing_mask, "li_2023"] * 1.15

    # Safety: drop rows where both years are NaN (edge case guard)
    df = df.dropna(subset=["li_2024", "li_2023"]).reset_index(drop=True)

    # Engineered features
    df["yoy_growth"]   = (df["li_2024"] / df["li_2023"].replace(0, np.nan)).fillna(1.0)
    df["log_users"]    = np.log10(df["li_2024"].clip(lower=1))
    df["norm_users"]   = (df["li_2024"] - df["li_2024"].min()) / (df["li_2024"].max() - df["li_2024"].min())
    df["penetration"]  = df["li_2024"] / df["li_2024"].max()   # relative penetration proxy

    print(f"✅ Loaded {len(df)} countries  |  Missing 2024 imputed: {missing_mask.sum()}")
    return df


def assign_behavior_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a node behaviour class (the prediction target) based on
    relative LinkedIn penetration quartiles.

      Class 0 – 'Emerging'   : Q1  (bottom 25 % penetration)
      Class 1 – 'Growing'    : Q2
      Class 2 – 'Established': Q3
      Class 3 – 'Saturated'  : Q4  (top 25 % penetration)

    Design note: quartile splits guarantee balanced classes (25 % each),
    which prevents the classifier from trivially predicting the majority
    class in a heavily skewed distribution.
    """
    quartiles = df["penetration"].quantile([0.25, 0.50, 0.75]).values

    def _label(p):
        if p <= quartiles[0]: return 0  # Emerging
        if p <= quartiles[1]: return 1  # Growing
        if p <= quartiles[2]: return 2  # Established
        return 3                         # Saturated

    df["class"]      = df["penetration"].apply(_label)
    df["class_name"] = df["class"].map({
        0: "Emerging", 1: "Growing", 2: "Established", 3: "Saturated"
    })
    print(f"✅ Class distribution:\n{df['class_name'].value_counts().to_string()}\n")
    return df


# ── Run data pipeline ─────────────────────────────────────────────────────────
DATA_PATH = "linkedin-users-by-country-2024.csv"   # ← update if needed in Colab

# In Colab you can upload the file and this path works directly.
# Or replace with the full Drive path:
# DATA_PATH = "/content/drive/MyDrive/linkedin-users-by-country-2024.csv"

df = load_and_clean(DATA_PATH)
df = assign_behavior_class(df)
print(df[["country", "li_2024", "li_2023", "yoy_growth", "log_users", "penetration", "class_name"]].head(10).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ▸ GRAPH CONSTRUCTION (Similarity Graph)
# ─────────────────────────────────────────────────────────────────────────────

def build_similarity_graph(
    df: pd.DataFrame,
    feature_cols: list,
    similarity_threshold: float = 0.85,
    min_edges_per_node: int = 1
) -> nx.Graph:
    """
    Construct an undirected weighted similarity graph where:
      - Nodes  = countries (indexed by DataFrame index)
      - Edges  = cosine similarity ≥ threshold between normalised feature vectors
      - Weight = cosine similarity value

    Design choices:
      • Cosine similarity is used because the feature vectors span very
        different magnitudes (log-users vs. growth ratio). It captures
        directional similarity regardless of scale.
      • Threshold 0.85 empirically produces a connected-enough graph
        for ~200 nodes without becoming a fully dense clique.
      • Nodes with no edge (isolated) are connected to their single
        most-similar neighbour to prevent disconnected components
        from breaking graph metrics.

    Args:
        df                  : DataFrame with node attributes
        feature_cols        : column names used for similarity
        similarity_threshold: cosine similarity cutoff [0, 1]
        min_edges_per_node  : fallback — add best edge if node is isolated

    Returns:
        G: nx.Graph with node attribute dict and edge weights
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Normalise features to unit vectors for cosine similarity
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].fillna(0))

    sim_matrix = cosine_similarity(X)          # (N × N)
    N = len(df)
    G = nx.Graph()

    # Add nodes with attributes
    for i, row in df.iterrows():
        G.add_node(i, **{
            "country"    : row["country"],
            "li_2024"    : row["li_2024"],
            "li_2023"    : row["li_2023"],
            "yoy_growth" : row["yoy_growth"],
            "log_users"  : row["log_users"],
            "penetration": row["penetration"],
            "class"      : row["class"],
        })

    # Add edges above threshold
    for i in range(N):
        for j in range(i + 1, N):
            if sim_matrix[i, j] >= similarity_threshold:
                G.add_edge(df.index[i], df.index[j], weight=sim_matrix[i, j])

    # Ensure every node has at least one edge (connectivity guarantee)
    for i in range(N):
        node = df.index[i]
        if G.degree(node) == 0:
            best_j = np.argpartition(sim_matrix[i], -2)[-2]   # second best (skip self)
            G.add_edge(node, df.index[best_j], weight=sim_matrix[i, best_j])

    n_nodes  = G.number_of_nodes()
    n_edges  = G.number_of_edges()
    density  = nx.density(G)
    n_comps  = nx.number_connected_components(G)
    print(f"✅ Graph built  |  Nodes: {n_nodes}  |  Edges: {n_edges}  "
          f"|  Density: {density:.4f}  |  Components: {n_comps}")
    return G, sim_matrix


GRAPH_FEATURE_COLS = ["log_users", "norm_users", "yoy_growth", "penetration"]
G, sim_matrix = build_similarity_graph(df, GRAPH_FEATURE_COLS, similarity_threshold=0.85)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ▸ K-HOP SUBGRAPH EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def get_khop_subgraph(G: nx.Graph, node, k: int = 2):
    """
    Return the ego subgraph containing all nodes within k hops of `node`.

    Uses nx.ego_graph which is O(k * avg_degree) — efficient for sparse graphs.

    Args:
        G    : the full graph
        node : central node ID
        k    : number of hops (radius)

    Returns:
        subgraph  : nx.Graph — k-hop ego subgraph
        neighbors : set     — all nodes within k hops (excluding center)
    """
    subgraph  = nx.ego_graph(G, node, radius=k, undirected=True)
    neighbors = set(subgraph.nodes()) - {node}
    return subgraph, neighbors


def compute_khop_stats(G: nx.Graph, node, df: pd.DataFrame, k: int = 2) -> dict:
    """
    Compute neighbourhood statistics for a node's k-hop subgraph.
    These become part of the 20-dim feature vector.

    Returns:
        khop_size     : number of nodes in k-hop neighbourhood
        khop_density  : edge density of subgraph (0 = star, 1 = clique)
        khop_avg_log  : mean log_users of neighbours
        khop_avg_pen  : mean penetration of neighbours
        khop_avg_grow : mean yoy_growth of neighbours
    """
    try:
        subgraph, neighbors = get_khop_subgraph(G, node, k=k)
    except Exception:
        # Edge case: isolated node (shouldn't happen after min_edges fix)
        return dict(khop_size=0, khop_density=0, khop_avg_log=0,
                    khop_avg_pen=0, khop_avg_grow=1)

    if len(neighbors) == 0:
        return dict(khop_size=0, khop_density=0,
                    khop_avg_log=df.loc[node, "log_users"],
                    khop_avg_pen=df.loc[node, "penetration"],
                    khop_avg_grow=df.loc[node, "yoy_growth"])

    neigh_list = list(neighbors)
    neigh_df   = df.loc[neigh_list]

    return {
        "khop_size"    : len(neighbors),
        "khop_density" : nx.density(subgraph),
        "khop_avg_log" : neigh_df["log_users"].mean(),
        "khop_avg_pen" : neigh_df["penetration"].mean(),
        "khop_avg_grow": neigh_df["yoy_growth"].mean(),
    }

print("✅ Section 4 ready — k-hop subgraph functions defined")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ▸ GRAPH METRICS COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_graph_metrics(G: nx.Graph) -> dict:
    """
    Compute five core graph centrality metrics for every node.

    Metrics selected:
      degree_centrality    – local connectivity (fast, O(n))
      betweenness_centrality – bridging role (O(n³) — manageable for n<1000)
      closeness_centrality – average path length (O(n²))
      clustering_coeff     – local clique-ness (O(n * avg_degree²))
      pagerank             – global influence via random walk (O(edges * iter))

    Returns dict of {metric_name: {node: value}}
    """
    print("  Computing degree centrality...")
    degree_c = nx.degree_centrality(G)

    print("  Computing betweenness centrality (may take ~10s for large graphs)...")
    betweenness_c = nx.betweenness_centrality(G, normalized=True, seed=RANDOM_STATE)

    print("  Computing closeness centrality...")
    # For disconnected graphs, closeness is computed per component — NetworkX handles this
    closeness_c = nx.closeness_centrality(G)

    print("  Computing clustering coefficients...")
    clustering_c = nx.clustering(G)

    print("  Computing PageRank...")
    pagerank_c = nx.pagerank(G, alpha=0.85, max_iter=500, tol=1e-6)

    print("✅ Graph metrics computed")
    return {
        "degree"      : degree_c,
        "betweenness" : betweenness_c,
        "closeness"   : closeness_c,
        "clustering"  : clustering_c,
        "pagerank"    : pagerank_c,
    }


metrics = compute_graph_metrics(G)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ▸ TOP-N NODES BY DEGREE CENTRALITY
# ─────────────────────────────────────────────────────────────────────────────

def select_top_nodes(metrics: dict, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Select top-N nodes by degree centrality for qualitative analysis.

    Design note: degree centrality highlights the most 'connected'
    countries in the similarity graph — i.e., countries whose LinkedIn
    penetration profile is most similar to many others. This surfaces
    representative 'hub' countries for each behavioural class.

    Args:
        metrics : output of compute_graph_metrics
        df      : raw DataFrame with country names
        top_n   : how many nodes to select

    Returns:
        DataFrame of top nodes with all centrality metrics
    """
    records = []
    for node, deg in metrics["degree"].items():
        records.append({
            "node"        : node,
            "country"     : df.loc[node, "country"],
            "class_name"  : df.loc[node, "class_name"],
            "degree_c"    : deg,
            "betweenness" : metrics["betweenness"][node],
            "closeness"   : metrics["closeness"][node],
            "clustering"  : metrics["clustering"][node],
            "pagerank"    : metrics["pagerank"][node],
        })
    top_df = pd.DataFrame(records).sort_values("degree_c", ascending=False).head(top_n)
    print(f"✅ Top {top_n} nodes by degree centrality:")
    print(top_df[["country", "class_name", "degree_c", "pagerank"]].to_string(index=False))
    return top_df


top_nodes_df = select_top_nodes(metrics, df, top_n=20)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 ▸ 20-DIMENSIONAL FEATURE VECTOR CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    G     : nx.Graph,
    df    : pd.DataFrame,
    metrics: dict,
    k     : int = 2
) -> tuple:
    """
    Build the (N × 20) feature matrix for every node.

    Feature breakdown (20 dimensions total):
    ──────────────────────────────────────────
    Dims  1–5   Node attributes (normalised):
                  log_users, norm_users, yoy_growth, penetration, li_2023_norm

    Dims  6–10  Graph metrics (already normalised by NetworkX):
                  degree, betweenness, closeness, clustering, pagerank

    Dims 11–15  k-hop neighbourhood statistics:
                  khop_size_norm, khop_density, khop_avg_log_norm,
                  khop_avg_pen, khop_avg_grow

    Dims 16–20  Engineered interaction features:
                  degree × pagerank        (hub influence score)
                  log_users × closeness    (reach-access interaction)
                  yoy_growth × khop_avg_grow (momentum cluster signal)
                  betweenness × degree     (bridge-hub dual role)
                  penetration × clustering (local market saturation cluster)

    Args:
        G       : graph
        df      : country DataFrame
        metrics : dict of centrality dicts
        k       : k-hop radius

    Returns:
        X       : np.ndarray (N × 20)
        y       : np.ndarray (N,) — class labels
        feature_names : list of 20 names
    """
    print(f"  Building {len(df)}-node × 20-dim feature matrix (k={k} hops)...")
    nodes = list(df.index)

    # Pre-normalise li_2023 for use as node attr
    li23_min, li23_max = df["li_2023"].min(), df["li_2023"].max()
    df["li_2023_norm"] = (df["li_2023"] - li23_min) / (li23_max - li23_min + 1e-9)

    # Pre-normalise khop_size (max possible = N-1)
    max_khop = len(df) - 1

    rows = []
    for node in nodes:
        # ── Dims 1–5: Node attributes ─────────────────────────────────────────
        na = [
            df.loc[node, "log_users"],
            df.loc[node, "norm_users"],
            df.loc[node, "yoy_growth"],
            df.loc[node, "penetration"],
            df.loc[node, "li_2023_norm"],
        ]

        # ── Dims 6–10: Graph metrics ──────────────────────────────────────────
        gm = [
            metrics["degree"][node],
            metrics["betweenness"][node],
            metrics["closeness"][node],
            metrics["clustering"][node],
            metrics["pagerank"][node],
        ]

        # ── Dims 11–15: k-hop stats ───────────────────────────────────────────
        kh = compute_khop_stats(G, node, df, k=k)
        khop_vec = [
            kh["khop_size"]     / max_khop,      # normalised
            kh["khop_density"],
            kh["khop_avg_log"]  / 9.0,           # log10(1B) ≈ 9
            kh["khop_avg_pen"],
            kh["khop_avg_grow"],
        ]

        # ── Dims 16–20: Interaction features ─────────────────────────────────
        d  = metrics["degree"][node]
        pr = metrics["pagerank"][node]
        lu = df.loc[node, "log_users"] / 9.0
        cl = metrics["closeness"][node]
        yg = df.loc[node, "yoy_growth"]
        kag= kh["khop_avg_grow"]
        bt = metrics["betweenness"][node]
        pe = df.loc[node, "penetration"]
        cc = metrics["clustering"][node]

        eng = [
            d  * pr,     # hub influence
            lu * cl,     # reach-access
            yg * kag,    # momentum cluster
            bt * d,      # bridge-hub
            pe * cc,     # local saturation cluster
        ]

        rows.append(na + gm + khop_vec + eng)

    X = np.array(rows, dtype=np.float32)
    y = df["class"].values

    feature_names = (
        ["log_users", "norm_users", "yoy_growth", "penetration", "li2023_norm"]     # 1-5
      + ["degree", "betweenness", "closeness", "clustering", "pagerank"]            # 6-10
      + ["khop_size", "khop_density", "khop_avg_log", "khop_avg_pen", "khop_avg_grow"]  # 11-15
      + ["hub_influence", "reach_access", "momentum_cluster", "bridge_hub", "sat_cluster"]  # 16-20
    )
    assert X.shape[1] == 20, f"Expected 20 features, got {X.shape[1]}"
    print(f"✅ Feature matrix shape: {X.shape}  |  Label distribution: {np.bincount(y)}")
    return X, y, feature_names


X_raw, y, feature_names = build_feature_matrix(G, df, metrics, k=2)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 ▸ PCA WITH VARIANCE THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

def run_pca_analysis(
    X        : np.ndarray,
    y        : np.ndarray,
    thresholds: list = [0.90, 0.80, 0.60],
    feature_names: list = None
) -> dict:
    """
    Apply PCA and find the minimum number of components to reach each
    explained-variance threshold.

    Design choices:
      • StandardScaler before PCA is mandatory — raw features span
        different numeric ranges (growth ratio ~1.0 vs. log_users ~3–9).
        Without scaling PCA would be dominated by high-variance dims.
      • Full PCA (n_components = min(N, 20)) is run once; thresholds are
        read from the cumulative explained variance array — efficient.

    Args:
        X           : raw (N × 20) feature matrix (unscaled)
        y           : class labels
        thresholds  : variance thresholds to evaluate
        feature_names: optional list of 20 feature names

    Returns:
        results dict with scaler, pca object, transformed X, component counts
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Full PCA — keep all components for variance analysis
    pca_full = PCA(n_components=min(X.shape[0], X.shape[1]), random_state=RANDOM_STATE)
    pca_full.fit(X_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    results = {
        "scaler"  : scaler,
        "pca_full": pca_full,
        "X_scaled": X_scaled,
        "cumvar"  : cumvar,
        "thresholds": {},
    }

    print("\n┌─────────────────────────────────────────────────────┐")
    print("│            PCA Variance Threshold Analysis           │")
    print("├──────────────┬───────────────────┬───────────────────┤")
    print("│  Threshold   │  Components Needed │  Actual Variance  │")
    print("├──────────────┼───────────────────┼───────────────────┤")

    for thresh in thresholds:
        n_comp = int(np.argmax(cumvar >= thresh)) + 1
        actual = cumvar[n_comp - 1]

        # Refit PCA with exactly n_comp components → get transformed X
        pca_t  = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        X_t    = pca_t.fit_transform(X_scaled)

        results["thresholds"][thresh] = {
            "n_components": n_comp,
            "actual_var"  : actual,
            "pca"         : pca_t,
            "X_pca"       : X_t,
        }
        print(f"│    {thresh*100:.0f}%       │        {n_comp:>2}           │      {actual*100:.2f}%        │")

    print("└──────────────┴───────────────────┴───────────────────┘\n")
    return results


pca_results = run_pca_analysis(X_raw, y, thresholds=[0.90, 0.80, 0.60],
                                feature_names=feature_names)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 ▸ CLASSIFIER TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["Emerging", "Growing", "Established", "Saturated"]

def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    label: str = "Full (20D)",
    n_estimators: int = 200
) -> dict:
    """
    Train a Random Forest and compute accuracy, F1, confusion matrix,
    and 5-fold stratified cross-validation score.

    Random Forest advantages for this task:
      • Robust to correlated features (many graph metrics are correlated)
      • Built-in feature importance (Gini-based)
      • No hyperparameter tuning required for a good baseline
      • Handles class imbalance reasonably well with balanced subsample

    Args:
        X_train, X_test, y_train, y_test : train/test splits
        label           : display label for this configuration
        n_estimators    : number of trees

    Returns:
        dict with accuracy, f1, cm, cv_scores
    """
    clf = RandomForestClassifier(
        n_estimators   = n_estimators,
        max_depth      = None,       # let trees grow fully
        class_weight   = "balanced", # handles minor class imbalance
        random_state   = RANDOM_STATE,
        n_jobs         = -1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_wt  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm     = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

    # 5-fold stratified CV on full train data
    cv_scores = cross_val_score(clf, X_train, y_train,
                                cv=StratifiedKFold(n_splits=5, shuffle=True,
                                                   random_state=RANDOM_STATE),
                                scoring="accuracy")

    print(f"\n{'─'*55}")
    print(f"  Configuration : {label}")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  F1 (macro)    : {f1_mac:.4f}")
    print(f"  F1 (weighted) : {f1_wt:.4f}")
    print(f"  CV Accuracy   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"{'─'*55}")

    return {
        "clf"      : clf,
        "accuracy" : acc,
        "f1_macro" : f1_mac,
        "f1_weighted": f1_wt,
        "cm"       : cm,
        "cv_scores": cv_scores,
        "y_pred"   : y_pred,
        "label"    : label,
    }


def run_all_experiments(pca_results, X_raw, y, test_size=0.25):
    """
    Train classifiers on:
      1. Raw 20-dim features (baseline)
      2. PCA @ 90% variance
      3. PCA @ 80% variance
      4. PCA @ 60% variance

    Uses the SAME train/test split for fair comparison.
    """
    X_scaled = pca_results["X_scaled"]

    # Common split indices — apply to all configurations
    idx = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    experiments = {}

    # Baseline: standardised raw 20-dim
    experiments["raw"] = train_and_evaluate(
        X_scaled[idx_train], X_scaled[idx_test], y_train, y_test,
        label="Raw 20D (scaled)"
    )

    # PCA variants
    for thresh, res in pca_results["thresholds"].items():
        X_pca   = res["X_pca"]
        n_comp  = res["n_components"]
        key     = f"pca_{int(thresh*100)}"
        experiments[key] = train_and_evaluate(
            X_pca[idx_train], X_pca[idx_test], y_train, y_test,
            label=f"PCA {int(thresh*100)}%  ({n_comp} dims)"
        )

    return experiments, idx_test, y_test


experiments, idx_test, y_test = run_all_experiments(pca_results, X_raw, y)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 ▸ VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_pca_variance_curve(pca_results):
    """Scree plot + cumulative variance with threshold markers."""
    pca_full = pca_results["pca_full"]
    cumvar   = pca_results["cumvar"]
    evr      = pca_full.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("PCA Variance Analysis", fontsize=14, fontweight="bold")

    # Scree plot
    axes[0].bar(range(1, len(evr)+1), evr * 100, color="#4E79A7", alpha=0.85)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Scree Plot — Individual Variance")
    axes[0].set_xticks(range(1, len(evr)+1))

    # Cumulative variance
    axes[1].plot(range(1, len(cumvar)+1), cumvar * 100,
                 marker="o", color="#E15759", linewidth=2)

    # Threshold lines
    thresh_colors = {0.90: "#59A14F", 0.80: "#F28E2B", 0.60: "#76B7B2"}
    for thresh, color in thresh_colors.items():
        n_comp = pca_results["thresholds"][thresh]["n_components"]
        axes[1].axhline(thresh * 100, color=color, linestyle="--", linewidth=1.5,
                        label=f"{int(thresh*100)}% threshold → {n_comp} PCs")
        axes[1].axvline(n_comp, color=color, linestyle=":", linewidth=1.2)

    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance (%)")
    axes[1].set_title("Cumulative Variance Curve")
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig("fig1_pca_variance.png", bbox_inches="tight")
    plt.show()
    print("✅ Figure 1 saved: fig1_pca_variance.png")


def plot_pca_embeddings(pca_results, y, df, thresh_list=[0.90, 0.80, 0.60]):
    """2D scatter plot of first two PCs for each variance threshold."""
    n_plots  = len(thresh_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    class_colors = {0: "#4E79A7", 1: "#F28E2B", 2: "#E15759", 3: "#59A14F"}
    class_labels = {0: "Emerging", 1: "Growing", 2: "Established", 3: "Saturated"}

    for ax, thresh in zip(axes, thresh_list):
        res  = pca_results["thresholds"][thresh]
        X_t  = res["X_pca"]
        n_c  = res["n_components"]

        for cls in [0, 1, 2, 3]:
            mask = y == cls
            ax.scatter(X_t[mask, 0], X_t[mask, 1] if X_t.shape[1] > 1 else np.zeros(mask.sum()),
                       c=[class_colors[cls]], label=class_labels[cls],
                       alpha=0.75, s=45, edgecolors="white", linewidths=0.4)

        # Annotate a few top countries
        top_idx = np.argsort(np.abs(X_t[:, 0]))[-5:]
        for i in top_idx:
            ax.annotate(df.loc[df.index[i], "country"][:10],
                        (X_t[i, 0], X_t[i, 1] if X_t.shape[1] > 1 else 0),
                        fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")

        pc2_label = "PC2" if X_t.shape[1] > 1 else "N/A (1D)"
        ax.set_xlabel(f"PC1 ({res['pca'].explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"{pc2_label}" + (
            f" ({res['pca'].explained_variance_ratio_[1]*100:.1f}%)"
            if X_t.shape[1] > 1 else ""))
        ax.set_title(f"PCA {int(thresh*100)}% — {n_c} component(s)")
        ax.legend(fontsize=8, markerscale=1.2)

    fig.suptitle("PCA 2D Embeddings by Variance Threshold", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("fig2_pca_embeddings.png", bbox_inches="tight")
    plt.show()
    print("✅ Figure 2 saved: fig2_pca_embeddings.png")


def plot_confusion_matrices(experiments):
    """4-panel confusion matrices for each experiment."""
    keys   = list(experiments.keys())
    n      = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4))
    if n == 1: axes = [axes]

    for ax, key in zip(axes, keys):
        exp = experiments[key]
        disp = ConfusionMatrixDisplay(
            confusion_matrix=exp["cm"],
            display_labels=CLASS_NAMES
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{exp['label']}\nAcc: {exp['accuracy']*100:.1f}%  F1: {exp['f1_macro']:.3f}",
                     fontsize=9)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Confusion Matrices — All Configurations", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("fig3_confusion_matrices.png", bbox_inches="tight")
    plt.show()
    print("✅ Figure 3 saved: fig3_confusion_matrices.png")


def plot_feature_importance(experiments, feature_names):
    """Top-20 feature importances from the raw-features Random Forest."""
    rf  = experiments["raw"]["clf"]
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(20), imp[idx], color=sns.color_palette("coolwarm", 20))
    ax.set_xticks(range(20))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Gini Importance")
    ax.set_title("Random Forest Feature Importance (Raw 20D)", fontweight="bold")

    # Colour bars by feature group
    group_colors = (["#4E79A7"]*5 + ["#F28E2B"]*5 + ["#59A14F"]*5 + ["#E15759"]*5)
    for bar, i in zip(bars, idx):
        bar.set_color(group_colors[i])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4E79A7", label="Node attributes (1-5)"),
        Patch(facecolor="#F28E2B", label="Graph metrics (6-10)"),
        Patch(facecolor="#59A14F", label="k-hop stats (11-15)"),
        Patch(facecolor="#E15759", label="Engineered (16-20)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig("fig4_feature_importance.png", bbox_inches="tight")
    plt.show()
    print("✅ Figure 4 saved: fig4_feature_importance.png")


def plot_cv_comparison(experiments):
    """Cross-validation accuracy comparison across all configurations."""
    labels = [exp["label"] for exp in experiments.values()]
    means  = [exp["cv_scores"].mean() * 100 for exp in experiments.values()]
    stds   = [exp["cv_scores"].std()  * 100 for exp in experiments.values()]

    fig, ax = plt.subplots(figsize=(8, 4))
    x   = np.arange(len(labels))
    colors = ["#4E79A7", "#59A14F", "#F28E2B", "#E15759"]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors[:len(labels)],
                  alpha=0.85, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("5-Fold CV Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Cross-Validation Accuracy — All Configurations", fontweight="bold")
    ax.axhline(25, color="grey", linestyle="--", linewidth=1, label="Random baseline (25%)")
    ax.legend(fontsize=8)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig("fig5_cv_comparison.png", bbox_inches="tight")
    plt.show()
    print("✅ Figure 5 saved: fig5_cv_comparison.png")


def plot_graph_overview(G, df, metrics, top_n=30):
    """
    Spring-layout visualisation of the top-30 highest-degree nodes,
    coloured by behaviour class.
    """
    # Subgraph of top-N by degree
    top_nodes  = sorted(metrics["degree"], key=metrics["degree"].get, reverse=True)[:top_n]
    sub_G      = G.subgraph(top_nodes)
    class_color_map = {0: "#4E79A7", 1: "#F28E2B", 2: "#E15759", 3: "#59A14F"}

    node_colors = [class_color_map[df.loc[n, "class"]] for n in sub_G.nodes()]
    node_sizes  = [metrics["degree"][n] * 3000 for n in sub_G.nodes()]
    labels      = {n: df.loc[n, "country"][:10] for n in sub_G.nodes()}

    fig, ax = plt.subplots(figsize=(13, 8))
    pos = nx.spring_layout(sub_G, seed=RANDOM_STATE, k=0.6)
    nx.draw_networkx_edges(sub_G, pos, ax=ax, alpha=0.25, edge_color="grey")
    nx.draw_networkx_nodes(sub_G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85)
    nx.draw_networkx_labels(sub_G, pos, labels=labels, ax=ax, font_size=7)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=c,
               markersize=10, label=l)
        for c, l in zip(class_color_map.values(),
                        ["Emerging", "Growing", "Established", "Saturated"])
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.set_title(f"Top-{top_n} Nodes by Degree Centrality — LinkedIn Similarity Graph",
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("fig6_graph_overview.png", bbox_inches="tight")
    plt.show()
    print("✅ Figure 6 saved: fig6_graph_overview.png")


# ── Run all visualisations ────────────────────────────────────────────────────
print("\n📊 Generating visualisations...\n")
plot_pca_variance_curve(pca_results)
plot_pca_embeddings(pca_results, y, df)
plot_confusion_matrices(experiments)
plot_feature_importance(experiments, feature_names)
plot_cv_comparison(experiments)
plot_graph_overview(G, df, metrics, top_n=30)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 ▸ FULL EVALUATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_full_report(experiments, pca_results, G, df, metrics):
    """Print a structured report summarising all results."""
    print("\n" + "═"*60)
    print("  FULL EVALUATION REPORT")
    print("═"*60)

    print("\n── Graph Statistics ─────────────────────────────────────")
    print(f"  Nodes   : {G.number_of_nodes()}")
    print(f"  Edges   : {G.number_of_edges()}")
    print(f"  Density : {nx.density(G):.4f}")
    print(f"  Components: {nx.number_connected_components(G)}")
    print(f"  Avg Clustering: {nx.average_clustering(G):.4f}")

    print("\n── PCA Summary ──────────────────────────────────────────")
    for thresh, res in pca_results["thresholds"].items():
        print(f"  {int(thresh*100)}% threshold → {res['n_components']} PCs "
              f"(actual: {res['actual_var']*100:.2f}%)")

    print("\n── Classifier Performance ───────────────────────────────")
    best_key = max(experiments, key=lambda k: experiments[k]["cv_scores"].mean())
    for key, exp in experiments.items():
        marker = " ◀ BEST" if key == best_key else ""
        print(f"  {exp['label']:30s}  "
              f"Acc: {exp['accuracy']*100:5.1f}%  "
              f"F1: {exp['f1_macro']:.3f}  "
              f"CV: {exp['cv_scores'].mean()*100:.1f}%{marker}")

    print("\n── Top-10 Most Connected Countries ──────────────────────")
    top10 = sorted(metrics["degree"].items(), key=lambda x: x[1], reverse=True)[:10]
    for node, deg in top10:
        print(f"  {df.loc[node,'country']:30s}  degree: {deg:.4f}  "
              f"class: {df.loc[node,'class_name']}")

    print("\n" + "═"*60)
    print("  ✅ Pipeline complete — all outputs saved")
    print("═"*60)


print_full_report(experiments, pca_results, G, df, metrics)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 ▸ INFERENCE ON NEW DATA
# ─────────────────────────────────────────────────────────────────────────────

def predict_new_country(
    country_name : str,
    li_2024      : float,
    li_2023_k    : float,          # in thousands
    experiments  : dict,
    pca_results  : dict,
    G            : nx.Graph,
    df           : pd.DataFrame,
    metrics      : dict,
    use_pca      : float = 0.90    # which PCA threshold to use
) -> str:
    """
    Predict the behaviour class for a hypothetical new country.

    Note: since the new node is not in the graph, we approximate its
    graph metrics using the mean of the closest existing node by feature
    distance — a common inductive extension approach.

    Args:
        country_name : display name
        li_2024      : LinkedIn users (absolute)
        li_2023_k    : LinkedIn users 2023 (thousands)
        use_pca      : PCA threshold to use (0.90 / 0.80 / 0.60)

    Returns:
        Predicted class label string
    """
    li_2023 = li_2023_k * 1_000

    # Approximate engineered features
    log_users   = np.log10(max(li_2024, 1))
    norm_users  = (li_2024 - df["li_2024"].min()) / (df["li_2024"].max() - df["li_2024"].min() + 1e-9)
    yoy_growth  = li_2024 / max(li_2023, 1)
    penetration = li_2024 / df["li_2024"].max()
    li_2023_norm= (li_2023 - df["li_2023"].min()) / (df["li_2023"].max() - df["li_2023"].min() + 1e-9)

    # Find nearest existing node by feature distance
    ref_feats = df[["log_users", "norm_users", "yoy_growth", "penetration"]].values
    new_feat  = np.array([log_users, norm_users, yoy_growth, penetration])
    dist      = np.linalg.norm(ref_feats - new_feat, axis=1)
    nearest   = df.index[np.argmin(dist)]

    # Approximate graph metrics from nearest neighbour
    d   = metrics["degree"][nearest]
    bt  = metrics["betweenness"][nearest]
    cl  = metrics["closeness"][nearest]
    cc  = metrics["clustering"][nearest]
    pr  = metrics["pagerank"][nearest]

    # Approximate k-hop stats from nearest neighbour's neighbourhood
    kh_stats = compute_khop_stats(G, nearest, df, k=2)

    # Build 20-dim vector
    na  = [log_users, norm_users, yoy_growth, penetration, li_2023_norm]
    gm  = [d, bt, cl, cc, pr]
    kh  = [
        kh_stats["khop_size"]     / (len(df) - 1),
        kh_stats["khop_density"],
        kh_stats["khop_avg_log"]  / 9.0,
        kh_stats["khop_avg_pen"],
        kh_stats["khop_avg_grow"],
    ]
    eng = [d*pr, (log_users/9.0)*cl, yoy_growth*kh_stats["khop_avg_grow"],
           bt*d, penetration*cc]

    x_new = np.array([na + gm + kh + eng], dtype=np.float32)

    # Scale and optionally apply PCA
    scaler  = pca_results["scaler"]
    x_scaled = scaler.transform(x_new)

    if use_pca in pca_results["thresholds"]:
        pca_t   = pca_results["thresholds"][use_pca]["pca"]
        x_input = pca_t.transform(x_scaled)
        clf_key = f"pca_{int(use_pca*100)}"
    else:
        x_input = x_scaled
        clf_key = "raw"

    clf    = experiments[clf_key]["clf"]
    pred   = clf.predict(x_input)[0]
    proba  = clf.predict_proba(x_input)[0]
    label  = CLASS_NAMES[pred]

    print(f"\n  Country   : {country_name}")
    print(f"  Prediction: {label} (class {pred})")
    print(f"  Confidence: {proba.max()*100:.1f}%")
    print(f"  Class probs: {dict(zip(CLASS_NAMES, [f'{p*100:.1f}%' for p in proba]))}")
    return label


# Example: predict for a hypothetical new market
print("\n── Inference Example ────────────────────────────────────")
predict_new_country(
    country_name="Hypothetical Market A",
    li_2024=5_000_000,
    li_2023_k=3_500,
    experiments=experiments,
    pca_results=pca_results,
    G=G,
    df=df,
    metrics=metrics,
    use_pca=0.90
)
