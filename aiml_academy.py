"""
🤖 AI/ML Academy — Interactive Learning Platform for Introductory AI Classes
Complete educational app covering classification, regression, clustering & neural nets.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Sklearn imports ─────────────────────────────────────────────────────────
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer,
    make_classification, make_regression, make_blobs, make_moons
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, silhouette_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🤖 AI/ML Academy",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── Global ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Hero Banner ─────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a4e 40%, #24243e 100%);
    border: 1px solid #3d3d8a;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,102,241,.35) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero-title span { color: #818cf8; }
.hero-sub {
    font-size: 1rem;
    color: #94a3b8;
    margin: 0;
}

/* ── Section Cards ───────────────────────────── */
.section-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 18px;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 6px;
}

/* ── Metric Chips ────────────────────────────── */
.metric-row {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    margin: 14px 0;
}
.metric-chip {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px 18px;
    text-align: center;
    min-width: 110px;
}
.metric-chip .val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #818cf8;
    display: block;
}
.metric-chip .lbl {
    font-size: .75rem;
    color: #64748b;
    display: block;
    margin-top: 2px;
}

/* ── Concept Boxes ───────────────────────────── */
.concept-box {
    background: linear-gradient(135deg, #0f172a, #1e1b4b);
    border-left: 4px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin: 14px 0;
    font-size: .92rem;
    color: #cbd5e1;
    line-height: 1.7;
}
.concept-box strong { color: #a5b4fc; }

/* ── Code Block ──────────────────────────────── */
.code-block {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: .8rem;
    color: #c9d1d9;
    line-height: 1.7;
    white-space: pre-wrap;
    margin: 10px 0;
}

/* ── Step Pills ──────────────────────────────── */
.step-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: .82rem;
    color: #94a3b8;
    margin: 4px 4px 4px 0;
}
.step-pill .num {
    background: #6366f1;
    color: white;
    width: 20px; height: 20px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: .72rem;
    font-weight: 700;
}

/* ── Algorithm Tag ───────────────────────────── */
.algo-tag {
    display: inline-block;
    background: rgba(99,102,241,.15);
    border: 1px solid rgba(99,102,241,.4);
    color: #a5b4fc;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: .78rem;
    font-weight: 600;
    margin: 2px;
}

/* ── Sidebar ─────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0a0f1e;
    border-right: 1px solid #1e293b;
}

/* ── Tabs ────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: #1e293b;
    border-radius: 8px;
    color: #64748b;
    font-weight: 500;
    border: 1px solid #334155;
}
.stTabs [aria-selected="true"] {
    background: #6366f1 !important;
    color: white !important;
    border-color: #6366f1 !important;
}

/* ── Plotly chart background ─────────────────── */
.stPlotlyChart { border-radius: 10px; overflow: hidden; }

/* ── Callout ─────────────────────────────────── */
.callout-green {
    background: rgba(16,185,129,.1);
    border: 1px solid rgba(16,185,129,.3);
    border-radius: 8px;
    padding: 12px 16px;
    color: #6ee7b7;
    font-size: .88rem;
    margin: 10px 0;
}
.callout-yellow {
    background: rgba(245,158,11,.1);
    border: 1px solid rgba(245,158,11,.3);
    border-radius: 8px;
    padding: 12px 16px;
    color: #fcd34d;
    font-size: .88rem;
    margin: 10px 0;
}
.callout-red {
    background: rgba(239,68,68,.1);
    border: 1px solid rgba(239,68,68,.3);
    border-radius: 8px;
    padding: 12px 16px;
    color: #fca5a5;
    font-size: .88rem;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS / PLOT UTILS
# ═══════════════════════════════════════════════════════════════════════════
DARK_TEMPLATE = dict(
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font=dict(color="#cbd5e1", family="Space Grotesk"),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
)

COLORS = ["#6366f1", "#10b981", "#f59e0b", "#ef4444",
          "#3b82f6", "#ec4899", "#8b5cf6", "#14b8a6"]


def dark_fig(fig):
    """Apply dark theme to any plotly figure."""
    fig.update_layout(**DARK_TEMPLATE, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def metric_html(metrics: dict):
    """Render metric chips."""
    chips = "".join(
        f'<div class="metric-chip"><span class="val">{v}</span><span class="lbl">{k}</span></div>'
        for k, v in metrics.items()
    )
    return f'<div class="metric-row">{chips}</div>'


def concept(text):
    st.markdown(f'<div class="concept-box">{text}</div>', unsafe_allow_html=True)


def codeblock(text):
    st.markdown(f'<div class="code-block">{text}</div>', unsafe_allow_html=True)


def algo_tags(*tags):
    html = "".join(f'<span class="algo-tag">{t}</span>' for t in tags)
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px;'>
        <div style='font-size:2.2rem'>🤖</div>
        <div style='font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-top:4px'>AI/ML Academy</div>
        <div style='font-size:.78rem; color:#475569; margin-top:2px'>Introductory Course</div>
    </div>
    <hr style='border-color:#1e293b; margin: 12px 0'>
    """, unsafe_allow_html=True)

    module = st.selectbox(
        "📚 Select Module",
        [
            "🏠  Home — Course Overview",
            "1️⃣  What is AI/ML?",
            "2️⃣  Classification",
            "3️⃣  Regression",
            "4️⃣  Clustering",
            "5️⃣  Neural Networks",
            "6️⃣  Model Comparison Lab",
            "7️⃣  Build Your Own Dataset",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#1e293b; margin: 12px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.78rem; color:#475569; padding: 0 4px; line-height:1.8'>
    <b style='color:#64748b'>📖 How to use</b><br>
    • Work through modules 1→7<br>
    • Adjust sliders to experiment<br>
    • Read the concept boxes<br>
    • Check the code examples<br><br>
    <b style='color:#64748b'>🛠 Tech Stack</b><br>
    Python · Scikit-learn<br>
    Plotly · Streamlit<br>
    NumPy · Pandas
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 0 — HOME
# ═══════════════════════════════════════════════════════════════════════════
if "Home" in module:
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Welcome to <span>AI/ML Academy</span> 🎓</p>
        <p class="hero-sub">An interactive, hands-on introduction to Artificial Intelligence and Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col, icon, title, desc in zip(
        [col1, col2, col3, col4],
        ["🎯", "📊", "🔍", "🧠"],
        ["Classification", "Regression", "Clustering", "Neural Nets"],
        ["Predict categories", "Predict numbers", "Find groups", "Deep learning"],
    ):
        with col:
            st.markdown(f"""
            <div class="section-card" style="text-align:center; cursor:pointer">
                <div style="font-size:2rem">{icon}</div>
                <div class="section-title">{title}</div>
                <div style="font-size:.82rem; color:#64748b">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 🗺️ Course Roadmap")
    roadmap_data = {
        "Module": ["1 – What is AI?", "2 – Classification", "3 – Regression",
                   "4 – Clustering", "5 – Neural Nets", "6 – Model Lab", "7 – Build Dataset"],
        "Concepts": [
            "AI vs ML, supervised vs unsupervised, ML pipeline",
            "Decision trees, kNN, SVM, Random Forest, accuracy metrics",
            "Linear regression, Ridge, Lasso, R², MSE",
            "K-Means, DBSCAN, PCA, silhouette score",
            "Perceptron, MLP, activation functions, backpropagation",
            "Side-by-side model benchmarking, bias-variance tradeoff",
            "Generate custom data, train any model, download results",
        ],
        "Difficulty": ["⭐", "⭐⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐", "⭐⭐"],
        "Time": ["10 min", "20 min", "15 min", "20 min", "20 min", "15 min", "15 min"],
    }
    df_road = pd.DataFrame(roadmap_data)
    st.dataframe(df_road, use_container_width=True, hide_index=True)

    st.markdown("### 💡 Key Takeaways You'll Learn")
    c1, c2 = st.columns(2)
    with c1:
        concept("""
        <strong>The ML Workflow:</strong> Every machine learning project follows the same pipeline:
        <br><br>
        🔹 <strong>Collect</strong> data → 🔹 <strong>Preprocess</strong> (clean, scale) →
        🔹 <strong>Split</strong> (train/test) → 🔹 <strong>Train</strong> the model →
        🔹 <strong>Evaluate</strong> (metrics) → 🔹 <strong>Tune</strong> hyperparameters →
        🔹 <strong>Deploy</strong>
        """)
    with c2:
        concept("""
        <strong>The Three ML Paradigms:</strong><br><br>
        🎯 <strong>Supervised Learning</strong> — learn from labeled examples (Classification, Regression)<br>
        🔍 <strong>Unsupervised Learning</strong> — find hidden patterns, no labels needed (Clustering)<br>
        🎮 <strong>Reinforcement Learning</strong> — learn by trial and reward (Games, Robotics)
        """)


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 1 — WHAT IS AI/ML?
# ═══════════════════════════════════════════════════════════════════════════
elif "What is AI" in module:
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Module 1: <span>What is AI & ML?</span></p>
        <p class="hero-sub">Understand the landscape — from Artificial Intelligence to Machine Learning to Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌍 The AI Landscape", "📐 The ML Pipeline", "🎲 Interactive Demo"])

    with tab1:
        c1, c2 = st.columns([3, 2])
        with c1:
            concept("""
            <strong>Artificial Intelligence (AI)</strong> is the broad field of making computers
            perform tasks that require human-like intelligence — reasoning, learning, problem-solving,
            understanding language, and perception.<br><br>
            <strong>Machine Learning (ML)</strong> is a <em>subset</em> of AI where computers
            <em>learn from data</em> rather than being explicitly programmed. Instead of writing
            rules, you show the algorithm examples and it finds the patterns itself.<br><br>
            <strong>Deep Learning (DL)</strong> is a <em>subset</em> of ML using multi-layer
            neural networks inspired by the human brain. It powers image recognition, ChatGPT,
            AlphaFold, and self-driving cars.
            """)

            st.markdown("#### Real-World Examples")
            examples = {
                "📧 Email Spam Filter": "ML classification — trained on millions of spam/not-spam emails",
                "🎵 Spotify Recommendations": "ML clustering + collaborative filtering on listening habits",
                "👁️ Face Unlock": "Deep learning on millions of face images",
                "💬 ChatGPT": "Large Language Model — transformer neural network with 175B parameters",
                "🏥 Cancer Detection": "Deep learning on medical scans, often outperforms radiologists",
                "🚗 Tesla Autopilot": "Computer vision + RL trained on billions of miles of driving data",
            }
            for k, v in examples.items():
                st.markdown(f"**{k}** — {v}")

        with c2:
            # Venn diagram as plotly
            fig = go.Figure()
            # AI circle
            theta = np.linspace(0, 2*np.pi, 100)
            r_ai, r_ml, r_dl = 160, 110, 65
            cx_ai, cy_ai = 200, 200
            fig.add_trace(go.Scatter(
                x=cx_ai + r_ai*np.cos(theta), y=cy_ai + r_ai*np.sin(theta),
                fill='toself', fillcolor='rgba(99,102,241,0.18)',
                line=dict(color='#6366f1', width=2),
                name='AI', mode='lines'
            ))
            fig.add_trace(go.Scatter(
                x=cx_ai + r_ml*np.cos(theta), y=cy_ai - 20 + r_ml*np.sin(theta),
                fill='toself', fillcolor='rgba(16,185,129,0.22)',
                line=dict(color='#10b981', width=2),
                name='ML', mode='lines'
            ))
            fig.add_trace(go.Scatter(
                x=cx_ai + r_dl*np.cos(theta), y=cy_ai - 25 + r_dl*np.sin(theta),
                fill='toself', fillcolor='rgba(245,158,11,0.28)',
                line=dict(color='#f59e0b', width=2),
                name='Deep Learning', mode='lines'
            ))
            for txt, x, y, size, col in [
                ("Artificial Intelligence", 200, 80, 13, "#a5b4fc"),
                ("Machine Learning", 200, 160, 11, "#6ee7b7"),
                ("Deep\nLearning", 200, 200, 10, "#fcd34d"),
            ]:
                fig.add_annotation(text=txt, x=x, y=y, showarrow=False,
                    font=dict(size=size, color=col, family="Space Grotesk"), align="center")
            fig.update_layout(
                **DARK_TEMPLATE, height=340, showlegend=False,
                xaxis=dict(visible=False, range=[0, 400]),
                yaxis=dict(visible=False, range=[0, 400]),
                title=dict(text="AI Ecosystem", font=dict(size=14, color="#94a3b8")),
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### The Standard ML Pipeline")
        steps = [
            ("1", "📦 Collect Data", "Gather raw data from sensors, databases, APIs, web scraping, etc."),
            ("2", "🧹 Preprocess", "Handle missing values, encode categories, remove outliers"),
            ("3", "✂️ Feature Engineering", "Select/create informative features; normalize/standardize"),
            ("4", "🔀 Split Dataset", "Train (70-80%) / Validation (10-15%) / Test (10-15%)"),
            ("5", "🏋️ Train Model", "Feed training data to the algorithm; it learns internal parameters"),
            ("6", "📏 Evaluate", "Measure performance on test set using accuracy, R², etc."),
            ("7", "🎛️ Tune", "Adjust hyperparameters using cross-validation to improve performance"),
            ("8", "🚀 Deploy", "Serve predictions via API, web app, or embed in a product"),
        ]
        for num, title, desc in steps:
            c1, c2 = st.columns([1, 5])
            with c1:
                st.markdown(f"""<div style='background:#6366f1;color:white;width:36px;height:36px;
                border-radius:50%;display:flex;align-items:center;justify-content:center;
                font-weight:700;font-size:.9rem;margin-top:4px'>{num}</div>""",
                unsafe_allow_html=True)
            with c2:
                st.markdown(f"**{title}** — {desc}")
            if num != "8":
                st.markdown("<div style='border-left:2px dashed #334155;height:12px;margin-left:17px'></div>",
                    unsafe_allow_html=True)

        st.markdown("#### Supervised vs Unsupervised Learning")
        codeblock(
"""# SUPERVISED LEARNING — you provide labels
X_train = [[1.2, 3.4], [2.1, 1.8], [3.5, 2.2]]   # features (inputs)
y_train = [0, 1, 0]                                 # labels (answers you provide)
model.fit(X_train, y_train)                          # algorithm learns the mapping

# UNSUPERVISED LEARNING — no labels, find structure
X_data  = [[1.2, 3.4], [2.1, 1.8], [3.5, 2.2]]    # features only, no labels
model.fit(X_data)                                    # algorithm finds hidden patterns
clusters = model.predict(X_data)                     # assigns cluster membership"""
        )

    with tab3:
        st.markdown("#### 🎲 See ML in Action — Coin Bias Estimator")
        concept("""
        A simple <strong>Bayesian learning demo</strong>: a biased coin has an unknown probability
        of heads. As we flip it more times, our estimate of the bias becomes more accurate.
        This is the essence of ML — <em>learning from data</em>.
        """)
        c1, c2 = st.columns([1, 2])
        with c1:
            true_bias = st.slider("True hidden bias (p of heads)", 0.1, 0.9, 0.7, 0.05)
            n_flips   = st.slider("Number of coin flips", 5, 500, 50)
            seed      = st.number_input("Random seed", 1, 100, 42)
        with c2:
            rng     = np.random.default_rng(int(seed))
            flips   = rng.random(n_flips) < true_bias
            ns      = np.arange(1, n_flips + 1)
            running = np.cumsum(flips) / ns

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ns, y=running, mode='lines',
                line=dict(color='#6366f1', width=2.5), name='Estimated bias'))
            fig.add_hline(y=true_bias, line=dict(color='#10b981', dash='dash', width=2),
                annotation_text=f"True bias = {true_bias}", annotation_font_color="#10b981")
            fig.update_layout(**DARK_TEMPLATE, height=280,
                title="Coin Bias Estimation vs. # Flips",
                xaxis_title="Flips seen so far", yaxis_title="Estimated P(Heads)",
                yaxis=dict(range=[0, 1], gridcolor="#1e293b"), showlegend=True,
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""
        <div class="callout-green">
        ✅ After <b>{n_flips}</b> flips: estimated bias = <b>{running[-1]:.3f}</b>,
        true bias = <b>{true_bias:.3f}</b>, error = <b>{abs(running[-1]-true_bias):.3f}</b><br>
        <em>More data → better estimates. This is the core principle of machine learning!</em>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 2 — CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════
elif "Classification" in module:
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Module 2: <span>Classification</span> 🎯</p>
        <p class="hero-sub">Predict which category an input belongs to — spam vs. not spam, cat vs. dog, disease vs. healthy</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 Concepts", "🔬 Train & Evaluate", "🗺️ Decision Boundaries", "📊 Metrics Deep Dive"
    ])

    # ── Sidebar config for classification
    with st.sidebar:
        st.markdown("---\n### ⚙️ Classification Settings")
        clf_dataset = st.selectbox("Dataset", ["Iris Flowers", "Wine Quality", "Breast Cancer", "Moons (synthetic)"])
        clf_algo = st.selectbox("Algorithm", [
            "Logistic Regression", "Decision Tree", "Random Forest",
            "K-Nearest Neighbors", "Support Vector Machine", "Naive Bayes",
            "Gradient Boosting", "Neural Network (MLP)"
        ])
        test_size = st.slider("Test set size (%)", 10, 40, 20)
        st.markdown("**Algorithm Hyperparams**")
        max_depth = st.slider("Max depth (Tree/Forest/GB)", 1, 20, 5)
        n_estim   = st.slider("N estimators (Forest/GB)", 10, 300, 100, 10)
        k_val     = st.slider("K neighbors (kNN)", 1, 25, 5)

    @st.cache_data
    def get_clf_data(name):
        if name == "Iris Flowers":
            d = load_iris()
            return pd.DataFrame(d.data, columns=d.feature_names), pd.Series(d.target), list(d.target_names)
        elif name == "Wine Quality":
            d = load_wine()
            return pd.DataFrame(d.data, columns=d.feature_names), pd.Series(d.target), list(d.target_names)
        elif name == "Breast Cancer":
            d = load_breast_cancer()
            return pd.DataFrame(d.data, columns=d.feature_names), pd.Series(d.target), list(d.target_names)
        else:
            X, y = make_moons(n_samples=400, noise=0.25, random_state=42)
            return pd.DataFrame(X, columns=["Feature 1", "Feature 2"]), pd.Series(y), ["Class 0", "Class 1"]

    @st.cache_resource
    def train_clf(algo, dataset, ts, md, ne, kv):
        X, y, names = get_clf_data(dataset)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts/100, random_state=42, stratify=y)
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        clf_map = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree":       DecisionTreeClassifier(max_depth=md, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=ne, max_depth=md, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=kv),
            "Support Vector Machine": SVC(probability=True, random_state=42),
            "Naive Bayes":         GaussianNB(),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=ne, max_depth=md, random_state=42),
            "Neural Network (MLP)":MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        }
        model = clf_map[algo]
        model.fit(Xtr_s, ytr)
        ypred = model.predict(Xte_s)
        acc   = accuracy_score(yte, ypred)
        cv    = cross_val_score(model, scaler.fit_transform(X), y, cv=5).mean()
        cm    = confusion_matrix(yte, ypred)
        report= classification_report(yte, ypred, target_names=names, output_dict=True)
        return model, scaler, Xtr, Xte, ytr, yte, ypred, acc, cv, cm, report, names

    result = train_clf(clf_algo, clf_dataset, test_size, max_depth, n_estim, k_val)
    model, scaler, Xtr, Xte, ytr, yte, ypred, acc, cv, cm, report, names = result

    with tab1:
        concept("""
        <strong>Classification</strong> is a supervised learning task where the goal is to predict
        a <em>discrete label/category</em> for each input. The model learns a decision boundary
        that separates the classes.<br><br>
        Examples: spam detection, image recognition, sentiment analysis, disease diagnosis.
        """)
        algos = {
            "Logistic Regression": "Finds a linear boundary using a sigmoid function. Simple, fast, interpretable. Works well when classes are linearly separable.",
            "Decision Tree": "Learns a flowchart of yes/no questions. Highly interpretable — you can literally read the rules. Prone to overfitting.",
            "Random Forest": "Ensemble of 100s of decision trees, each trained on a random subset of data/features. Robust, accurate, less overfitting.",
            "K-Nearest Neighbors": "For a new point, look at the K closest training examples and vote. No training phase — all computation at prediction time.",
            "Support Vector Machine": "Finds the maximum-margin hyperplane separating classes. Works in high dimensions. Uses kernel trick for non-linear boundaries.",
            "Naive Bayes": "Applies Bayes' theorem assuming features are independent. Very fast, works well for text classification.",
            "Gradient Boosting": "Builds trees sequentially, each one correcting the errors of the previous. State-of-the-art on tabular data.",
            "Neural Network (MLP)": "Layers of interconnected neurons. Can learn arbitrary complex patterns given enough data.",
        }
        for alg, desc in algos.items():
            with st.expander(f"**{alg}**"):
                st.write(desc)
        codeblock(
"""# Standard classification workflow in scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Load and split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Scale features (crucial for kNN, SVM, LR, MLP)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)        # use same scaler!

# 3. Train
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))"""
        )

    with tab2:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test Accuracy", f"{acc:.1%}")
        c2.metric("CV Score (5-fold)", f"{cv:.1%}")
        c3.metric("Train Samples", len(Xtr))
        c4.metric("Test Samples", len(Xte))

        cA, cB = st.columns(2)
        with cA:
            # Confusion matrix
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual"),
                x=names, y=names, title="Confusion Matrix")
            fig.update_layout(**DARK_TEMPLATE, height=340,
                coloraxis_showscale=False,
                xaxis=dict(side="bottom"),
                font=dict(size=12))
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

        with cB:
            # Per-class metrics
            rows = []
            for cls in names:
                if cls in report:
                    rows.append({
                        "Class": cls,
                        "Precision": round(report[cls]["precision"], 3),
                        "Recall":    round(report[cls]["recall"], 3),
                        "F1-Score":  round(report[cls]["f1-score"], 3),
                        "Support":   int(report[cls]["support"]),
                    })
            df_rep = pd.DataFrame(rows)
            fig2 = go.Figure()
            for metric, color in [("Precision","#6366f1"),("Recall","#10b981"),("F1-Score","#f59e0b")]:
                fig2.add_trace(go.Bar(name=metric, x=df_rep["Class"], y=df_rep[metric],
                    marker_color=color))
            fig2.update_layout(**DARK_TEMPLATE, height=340,
                title="Per-Class Performance", barmode="group",
                yaxis=dict(range=[0,1.05], gridcolor="#1e293b"),
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            dark_fig(fig2)
            st.plotly_chart(fig2, use_container_width=True)

        # Prediction on new sample
        st.markdown("#### 🔮 Make a Live Prediction")
        X_all, y_all, _ = get_clf_data(clf_dataset)
        sample_idx = st.slider("Select a sample from test set", 0, len(Xte)-1, 0)
        sample = Xte.iloc[sample_idx]
        scaled = scaler.transform(sample.values.reshape(1,-1))
        pred   = model.predict(scaled)[0]
        true   = yte.iloc[sample_idx]
        proba  = model.predict_proba(scaled)[0] if hasattr(model, "predict_proba") else None

        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(sample.to_frame("Value"), use_container_width=True)
        with c2:
            st.markdown(f"""
            <div style='background:#1e293b;border-radius:10px;padding:18px'>
            <div style='color:#94a3b8;font-size:.85rem'>Predicted Class</div>
            <div style='color:#818cf8;font-size:1.6rem;font-weight:700'>{names[pred]}</div>
            <div style='color:#94a3b8;font-size:.85rem;margin-top:10px'>Actual Class</div>
            <div style='color:{"#10b981" if pred==true else "#ef4444"};font-size:1.4rem;font-weight:600'>{names[true]}</div>
            <div style='margin-top:6px;color:{"#6ee7b7" if pred==true else "#fca5a5"};font-size:.9rem'>
            {"✅ Correct prediction!" if pred==true else "❌ Wrong prediction"}</div>
            </div>
            """, unsafe_allow_html=True)
            if proba is not None:
                fig_p = go.Figure(go.Bar(x=names, y=proba, marker_color=COLORS[:len(names)]))
                fig_p.update_layout(**DARK_TEMPLATE, height=200, title="Class Probabilities",
                    yaxis=dict(range=[0,1]), margin=dict(l=10,r=10,t=36,b=10))
                dark_fig(fig_p)
                st.plotly_chart(fig_p, use_container_width=True)

    with tab3:
        if clf_dataset == "Moons (synthetic)":
            X2, y2, _ = get_clf_data(clf_dataset)
            h = 0.05
            x_min, x_max = X2.iloc[:,0].min()-0.5, X2.iloc[:,0].max()+0.5
            y_min, y_max = X2.iloc[:,1].min()-0.5, X2.iloc[:,1].max()+0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            grid = np.c_[xx.ravel(), yy.ravel()]
            grid_s = scaler.transform(grid)
            Z = model.predict(grid_s).reshape(xx.shape)

            fig = go.Figure()
            fig.add_trace(go.Contour(x=np.arange(x_min,x_max,h), y=np.arange(y_min,y_max,h),
                z=Z, colorscale=[[0,"rgba(99,102,241,.25)"],[1,"rgba(16,185,129,.25)"]],
                showscale=False, contours=dict(coloring='fill'), line_width=0))
            for cls_val, col, nm in [(0,"#6366f1","Class 0"),(1,"#10b981","Class 1")]:
                mask = y2 == cls_val
                fig.add_trace(go.Scatter(
                    x=X2[mask].iloc[:,0], y=X2[mask].iloc[:,1],
                    mode='markers', marker=dict(color=col, size=6, opacity=0.8),
                    name=nm))
            fig.update_layout(**DARK_TEMPLATE, height=420, title=f"Decision Boundary — {clf_algo}",
                xaxis_title="Feature 1", yaxis_title="Feature 2",
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 Switch algorithms in the sidebar to see how different models draw boundaries!")
        else:
            st.info("🎯 Switch dataset to **Moons (synthetic)** in the sidebar to see 2D decision boundaries!")
            X2, y2, n2 = get_clf_data(clf_dataset)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(scaler.fit_transform(X2))
            fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=[n2[i] for i in y2],
                title=f"{clf_dataset} — PCA 2D Projection",
                labels={"x":"PC1","y":"PC2","color":"Class"},
                color_discrete_sequence=COLORS)
            fig.update_layout(**DARK_TEMPLATE, height=420)
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        concept("""
        <strong>Classification Metrics Explained:</strong><br><br>
        🎯 <strong>Accuracy</strong> = (correct predictions) / (total predictions). Misleading on imbalanced datasets.<br>
        📐 <strong>Precision</strong> = (true positives) / (predicted positives). "When I say Yes, how often am I right?"<br>
        📡 <strong>Recall</strong> = (true positives) / (actual positives). "Of all actual Yes cases, how many did I catch?"<br>
        ⚖️ <strong>F1-Score</strong> = harmonic mean of precision and recall. Best single metric for imbalanced classes.<br>
        🧮 <strong>Confusion Matrix</strong> = table showing TP, FP, TN, FN counts per class.
        """)

        st.markdown("#### Interactive: Threshold Effect on Metrics")
        st.write("For binary classifiers, changing the decision threshold trades off precision vs recall:")
        if hasattr(model, "predict_proba") and len(names) == 2:
            X2, y2, _ = get_clf_data(clf_dataset)
            Xs = scaler.transform(X2)
            probs = model.predict_proba(Xs)[:,1]
            thresholds = np.linspace(0.05, 0.95, 50)
            precisions, recalls, f1s, accs = [], [], [], []
            for t in thresholds:
                yp = (probs >= t).astype(int)
                from sklearn.metrics import precision_score, recall_score, f1_score
                precisions.append(precision_score(y2, yp, zero_division=0))
                recalls.append(recall_score(y2, yp, zero_division=0))
                f1s.append(f1_score(y2, yp, zero_division=0))
                accs.append(accuracy_score(y2, yp))
            fig = go.Figure()
            for vals, name, col in [
                (precisions,"Precision","#6366f1"),
                (recalls,"Recall","#10b981"),
                (f1s,"F1-Score","#f59e0b"),
                (accs,"Accuracy","#ef4444"),
            ]:
                fig.add_trace(go.Scatter(x=thresholds, y=vals, name=name,
                    line=dict(color=col, width=2.5), mode='lines'))
            fig.update_layout(**DARK_TEMPLATE, height=360,
                title="Metric vs. Decision Threshold",
                xaxis_title="Decision Threshold", yaxis_title="Score",
                yaxis=dict(range=[0,1.05], gridcolor="#1e293b"),
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Choose a binary dataset (Breast Cancer) for threshold analysis.")


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 3 — REGRESSION
# ═══════════════════════════════════════════════════════════════════════════
elif "Regression" in module:
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Module 3: <span>Regression</span> 📈</p>
        <p class="hero-sub">Predict a continuous numeric value — house prices, temperature, stock returns, patient age</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---\n### ⚙️ Regression Settings")
        reg_algo = st.selectbox("Algorithm", [
            "Linear Regression", "Ridge Regression", "Lasso Regression",
            "Decision Tree", "Random Forest", "Support Vector Regression",
        ])
        n_samples = st.slider("# Training samples", 50, 500, 150)
        noise_lvl = st.slider("Noise level", 0.0, 1.0, 0.3, 0.05)
        reg_alpha  = st.slider("Regularization (α)", 0.01, 10.0, 1.0, 0.01)
        reg_depth  = st.slider("Max depth (Tree/Forest)", 1, 15, 4)

    tab1, tab2, tab3 = st.tabs(["📖 Concepts", "🔬 Train & Evaluate", "🔎 Feature Importance"])

    with tab1:
        concept("""
        <strong>Regression</strong> predicts a <em>continuous numeric output</em> rather than
        a class label. The model learns a function f(X) → y that minimizes the prediction error.<br><br>
        Examples: predicting house prices, forecasting energy demand, estimating patient survival time,
        predicting crop yield from weather data.
        """)
        codeblock(
"""# Linear Regression — the foundation of regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso

model = LinearRegression()       # y = w0 + w1*x1 + w2*x2 + ... + wn*xn
model = Ridge(alpha=1.0)         # Same + L2 penalty: |w|^2  — shrinks all weights
model = Lasso(alpha=1.0)         # Same + L1 penalty: |w|    — zeroes out weights (feature selection!)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Key regression metrics
from sklearn.metrics import mean_squared_error, r2_score
mse  = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5                # Root MSE — in same units as y
r2   = r2_score(y_test, y_pred) # 1.0 = perfect, 0 = predicts mean, <0 = terrible"""
        )

        c1, c2 = st.columns(2)
        with c1:
            concept("""
            <strong>Underfitting (High Bias)</strong><br>
            Model is too simple; misses real patterns in the data.<br>
            Fix: use a more complex model, more features, less regularization.
            """)
        with c2:
            concept("""
            <strong>Overfitting (High Variance)</strong><br>
            Model memorizes noise in training data; fails on new data.<br>
            Fix: more data, regularization (Ridge/Lasso), simpler model, dropout.
            """)

    with tab2:
        @st.cache_data
        def make_reg_data(n, noise, seed=42):
            np.random.seed(seed)
            X = np.sort(np.random.rand(n, 1) * 10, axis=0)
            y = 2*X[:,0] + 1.5*np.sin(X[:,0]) - 0.5*(X[:,0]-5)**2 * 0.3 + np.random.randn(n)*noise*3
            return X, y

        X_r, y_r = make_reg_data(n_samples, noise_lvl)
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)

        reg_map = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression":  Ridge(alpha=reg_alpha),
            "Lasso Regression":  Lasso(alpha=reg_alpha),
            "Decision Tree":     DecisionTreeRegressor(max_depth=reg_depth, random_state=42),
            "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=reg_depth, random_state=42),
            "Support Vector Regression": SVR(C=reg_alpha, kernel='rbf'),
        }
        reg_model = reg_map[reg_algo]
        reg_model.fit(X_train_r, y_train_r)
        y_pred_r = reg_model.predict(X_test_r)

        mse  = mean_squared_error(y_test_r, y_pred_r)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test_r, y_pred_r)
        train_r2 = r2_score(y_train_r, reg_model.predict(X_train_r))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test R²",  f"{r2:.4f}")
        c2.metric("Train R²", f"{train_r2:.4f}")
        c3.metric("RMSE",     f"{rmse:.3f}")
        c4.metric("Overfit gap", f"{train_r2-r2:.4f}")

        X_line = np.linspace(0, 10, 300).reshape(-1,1)
        y_line  = reg_model.predict(X_line)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_train_r[:,0], y=y_train_r, mode='markers',
            marker=dict(color='#6366f1', size=6, opacity=0.7), name='Train data'))
        fig.add_trace(go.Scatter(x=X_test_r[:,0], y=y_test_r, mode='markers',
            marker=dict(color='#f59e0b', size=8, symbol='diamond', opacity=0.9), name='Test data'))
        fig.add_trace(go.Scatter(x=X_line[:,0], y=y_line, mode='lines',
            line=dict(color='#10b981', width=2.5), name=reg_algo))
        fig.update_layout(**DARK_TEMPLATE, height=400,
            title=f"{reg_algo} Fit  |  R² = {r2:.3f}  |  RMSE = {rmse:.3f}",
            xaxis_title="Feature", yaxis_title="Target",
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        # Residuals plot
        resid = y_test_r - y_pred_r
        fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Residuals vs Predicted", "Residual Distribution"))
        fig2.add_trace(go.Scatter(x=y_pred_r, y=resid, mode='markers',
            marker=dict(color='#6366f1', size=7, opacity=0.7)), row=1, col=1)
        fig2.add_hline(y=0, line=dict(color='#10b981', dash='dash'), row=1, col=1)
        fig2.add_trace(go.Histogram(x=resid, nbinsx=20,
            marker_color='#6366f1', opacity=0.75, name='Residuals'), row=1, col=2)
        fig2.update_layout(**DARK_TEMPLATE, height=300, showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)

        if abs(train_r2 - r2) > 0.15:
            st.markdown('<div class="callout-yellow">⚠️ Large gap between train R² and test R² suggests <strong>overfitting</strong>. Try reducing max depth or adding regularization.</div>', unsafe_allow_html=True)
        elif r2 < 0.5:
            st.markdown('<div class="callout-red">❌ Low R² suggests <strong>underfitting</strong>. Try a more complex model or increase data.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="callout-green">✅ Good balance between training and test performance!</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown("#### Effect of Regularization on Coefficients")
        concept("""
        <strong>Regularization</strong> adds a penalty for large weights to prevent overfitting.
        <strong>Ridge (L2)</strong> shrinks all weights toward zero.
        <strong>Lasso (L1)</strong> drives some weights exactly to zero — automatic feature selection!
        """)
        # Multi-feature synthetic dataset
        X_mf, y_mf = make_regression(n_samples=200, n_features=10, n_informative=5, noise=20, random_state=42)
        alphas = np.logspace(-3, 2, 50)
        ridge_coefs = [Ridge(alpha=a).fit(X_mf, y_mf).coef_ for a in alphas]
        lasso_coefs = [Lasso(alpha=a, max_iter=5000).fit(X_mf, y_mf).coef_ for a in alphas]

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Ridge (L2) Coefficients", "Lasso (L1) Coefficients"))
        for i, (coef_list, col) in enumerate([(ridge_coefs, 1), (lasso_coefs, 2)]):
            for j in range(10):
                vals = [c[j] for c in coef_list]
                color = COLORS[j % len(COLORS)]
                fig.add_trace(go.Scatter(x=alphas, y=vals, mode='lines',
                    line=dict(color=color, width=1.5), showlegend=(i==0),
                    name=f"Feature {j+1}"), row=1, col=col)
        fig.update_xaxes(type="log", title_text="Alpha (regularization strength)")
        fig.update_yaxes(title_text="Coefficient value")
        fig.update_layout(**DARK_TEMPLATE, height=380,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)))
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 Notice how Lasso drives coefficients exactly to 0 — it performs automatic feature selection!")


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 4 — CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════
elif "Clustering" in module:
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Module 4: <span>Clustering</span> 🔍</p>
        <p class="hero-sub">Unsupervised learning — discover hidden structure and groups in unlabeled data</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---\n### ⚙️ Clustering Settings")
        clust_algo = st.selectbox("Algorithm", ["K-Means", "DBSCAN", "Agglomerative"])
        n_clusters = st.slider("Number of clusters (K-Means / Agg)", 2, 10, 3)
        eps_val    = st.slider("Epsilon (DBSCAN)", 0.1, 2.0, 0.5, 0.05)
        min_samp   = st.slider("Min samples (DBSCAN)", 2, 20, 5)
        blob_std   = st.slider("Cluster spread", 0.3, 2.5, 0.8, 0.1)
        n_pts      = st.slider("Data points", 100, 600, 300, 50)

    tab1, tab2, tab3 = st.tabs(["📖 Concepts", "🔬 Live Clustering", "📉 Elbow & PCA"])

    with tab1:
        concept("""
        <strong>Clustering</strong> is an <em>unsupervised</em> learning task — there are no labels.
        The algorithm finds groups (clusters) of similar data points purely based on their features.<br><br>
        Applications: customer segmentation, anomaly detection, document grouping,
        gene expression analysis, image compression, social network community detection.
        """)
        for algo, desc in {
            "K-Means": "Assigns each point to the nearest of K centroids, then updates centroids iteratively. Fast, scalable. Assumes spherical clusters of similar size.",
            "DBSCAN": "Density-based: finds core points with many neighbors, expands clusters from them. Handles arbitrary shapes, detects outliers (noise points). No need to specify K.",
            "Agglomerative": "Hierarchically merges the two closest clusters bottom-up. Produces a dendrogram showing the merge history. Flexible linkage criteria.",
        }.items():
            with st.expander(f"**{algo}**"):
                st.write(desc)
        codeblock(
"""from sklearn.cluster import KMeans, DBSCAN

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# DBSCAN — no need to specify K!
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)     # label=-1 means noise/outlier

# Silhouette score: measures cluster quality (-1 bad, 0 overlapping, 1 perfect)
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)"""
        )

    with tab2:
        X_cl, y_true = make_blobs(n_samples=n_pts, centers=n_clusters,
                                   cluster_std=blob_std, random_state=42)

        if clust_algo == "K-Means":
            cmodel = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif clust_algo == "DBSCAN":
            cmodel = DBSCAN(eps=eps_val, min_samples=min_samp)
        else:
            cmodel = AgglomerativeClustering(n_clusters=n_clusters)

        labels_pred = cmodel.fit_predict(X_cl)
        n_found = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
        n_noise = np.sum(labels_pred == -1)

        try:
            sil = silhouette_score(X_cl, labels_pred) if n_found > 1 else 0
        except Exception:
            sil = 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters Found", n_found)
        c2.metric("Silhouette Score", f"{sil:.3f}")
        c3.metric("Noise Points", n_noise)

        cA, cB = st.columns(2)
        with cA:
            label_names = [f"Cluster {l}" if l >= 0 else "Noise" for l in labels_pred]
            fig = px.scatter(x=X_cl[:,0], y=X_cl[:,1], color=label_names,
                color_discrete_sequence=COLORS + ["#ffffff"],
                title=f"{clust_algo} Result",
                labels={"x":"Feature 1","y":"Feature 2","color":"Cluster"})
            if hasattr(cmodel, 'cluster_centers_'):
                cc = cmodel.cluster_centers_
                fig.add_trace(go.Scatter(x=cc[:,0], y=cc[:,1], mode='markers',
                    marker=dict(symbol='x', size=14, color='white', line_width=2),
                    name='Centroids'))
            fig.update_layout(**DARK_TEMPLATE, height=380,
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

        with cB:
            true_names = [f"True Group {l}" for l in y_true]
            fig2 = px.scatter(x=X_cl[:,0], y=X_cl[:,1], color=true_names,
                color_discrete_sequence=COLORS,
                title="Ground Truth Groups",
                labels={"x":"Feature 1","y":"Feature 2","color":"True Label"})
            fig2.update_layout(**DARK_TEMPLATE, height=380,
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            dark_fig(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("#### 📉 Elbow Method — Finding the Best K")
        concept("""
        For K-Means, plot the <strong>inertia</strong> (sum of squared distances to nearest centroid)
        for different values of K. The "elbow" — where inertia stops decreasing rapidly — is a good choice for K.
        """)
        X_elb, _ = make_blobs(n_samples=300, centers=4, random_state=42)
        ks = range(1, 11)
        inertias  = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_elb).inertia_ for k in ks]
        sil_scores = [silhouette_score(X_elb, KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_elb))
                      if k > 1 else 0 for k in ks]

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Inertia (Elbow Method)", "Silhouette Score"))
        fig.add_trace(go.Scatter(x=list(ks), y=inertias, mode='lines+markers',
            line=dict(color='#6366f1', width=2.5), marker=dict(size=8)), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(ks), y=sil_scores, mode='lines+markers',
            line=dict(color='#10b981', width=2.5), marker=dict(size=8)), row=1, col=2)
        fig.add_vline(x=4, line=dict(color='#f59e0b', dash='dash'), row=1, col=1)
        fig.add_vline(x=4, line=dict(color='#f59e0b', dash='dash'), row=1, col=2)
        fig.update_xaxes(title_text="Number of Clusters K")
        fig.update_layout(**DARK_TEMPLATE, height=320, showlegend=False)
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🔻 PCA: Visualizing High-Dimensional Data")
        concept("""
        <strong>PCA (Principal Component Analysis)</strong> reduces high-dimensional data to 2D or 3D
        for visualization, while preserving the most important variance.
        """)
        d_iris = load_iris()
        X_iris = StandardScaler().fit_transform(d_iris.data)
        pca3   = PCA(n_components=3)
        X_3d   = pca3.fit_transform(X_iris)
        var_exp = pca3.explained_variance_ratio_

        fig3 = px.scatter_3d(x=X_3d[:,0], y=X_3d[:,1], z=X_3d[:,2],
            color=[d_iris.target_names[i] for i in d_iris.target],
            color_discrete_sequence=COLORS,
            labels={"x":f"PC1 ({var_exp[0]:.1%})","y":f"PC2 ({var_exp[1]:.1%})",
                    "z":f"PC3 ({var_exp[2]:.1%})","color":"Species"},
            title=f"Iris Dataset — 3D PCA ({sum(var_exp):.1%} variance explained)")
        fig3.update_layout(**DARK_TEMPLATE, height=440,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            scene=dict(
                xaxis=dict(backgroundcolor="#0f172a", gridcolor="#1e293b"),
                yaxis=dict(backgroundcolor="#0f172a", gridcolor="#1e293b"),
                zaxis=dict(backgroundcolor="#0f172a", gridcolor="#1e293b"),
            ))
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 5 — NEURAL NETWORKS
# ═══════════════════════════════════════════════════════════════════════════
elif "Neural Networks" in module:
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Module 5: <span>Neural Networks</span> 🧠</p>
        <p class="hero-sub">From the perceptron to deep learning — how artificial neurons learn</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 Concepts", "🔬 Train MLP", "🧪 Activation Functions"])

    with tab1:
        concept("""
        A <strong>Neural Network</strong> is a stack of layers, each containing neurons.
        Each neuron computes a weighted sum of its inputs, applies an <em>activation function</em>,
        and passes the result to the next layer.<br><br>
        <strong>Backpropagation</strong> computes gradients of the loss with respect to every weight
        using the chain rule, then <strong>gradient descent</strong> updates the weights to minimize loss.
        """)

        # ASCII-style network diagram via plotly
        fig = go.Figure()
        layer_sizes = [4, 6, 6, 3]
        layer_names = ["Input\nLayer", "Hidden\nLayer 1", "Hidden\nLayer 2", "Output\nLayer"]
        layer_colors = ["#6366f1", "#10b981", "#10b981", "#f59e0b"]
        node_positions = {}

        for li, (size, color) in enumerate(zip(layer_sizes, layer_colors)):
            xs = [li * 2] * size
            ys = [(j - size/2) * 1.2 for j in range(size)]
            for j, (x, y) in enumerate(zip(xs, ys)):
                node_positions[(li, j)] = (x, y)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers',
                marker=dict(size=28, color=color, line=dict(color='white', width=1.5)),
                showlegend=False, hoverinfo='skip'))
            fig.add_annotation(text=layer_names[li].replace('\n','<br>'),
                x=li*2, y=min(ys)-0.85, showarrow=False,
                font=dict(size=10, color="#94a3b8"), align='center')

        # Draw connections (subset for clarity)
        for li in range(len(layer_sizes)-1):
            for j in range(layer_sizes[li]):
                for k in range(layer_sizes[li+1]):
                    x0, y0 = node_positions[(li, j)]
                    x1, y1 = node_positions[(li+1, k)]
                    fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                        line=dict(color='rgba(99,102,241,0.18)', width=1),
                        showlegend=False, hoverinfo='skip'))

        fig.update_layout(**DARK_TEMPLATE, height=360,
            title="Multi-Layer Perceptron (MLP) Architecture",
            xaxis=dict(visible=False, range=[-0.5, 6.5]),
            yaxis=dict(visible=False, range=[-4.5, 4.5]),
            margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        codeblock(
"""from sklearn.neural_network import MLPClassifier

# Architecture: 2 hidden layers with 64 and 32 neurons
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),   # 2 hidden layers: 64 → 32 neurons
    activation='relu',              # Activation: relu, tanh, logistic
    solver='adam',                  # Optimizer: adam (adaptive gradient)
    alpha=0.0001,                   # L2 regularization
    learning_rate_init=0.001,       # Initial learning rate
    max_iter=500,                   # Max training epochs
    random_state=42
)
model.fit(X_train, y_train)

# The loss curve
loss_curve = model.loss_curve_     # List of loss values per epoch"""
        )

    with tab2:
        with st.sidebar:
            st.markdown("---\n### ⚙️ Neural Net Settings")
            nn_dataset = st.selectbox("Dataset", ["Iris", "Wine", "Breast Cancer", "Moons"])
            h1 = st.slider("Hidden layer 1 neurons", 4, 128, 64, 4)
            h2 = st.slider("Hidden layer 2 neurons", 0, 128, 32, 4)
            activation = st.selectbox("Activation", ["relu", "tanh", "logistic"])
            lr = st.select_slider("Learning rate", [0.0001, 0.001, 0.01, 0.1], value=0.001)
            nn_alpha = st.slider("L2 reg (alpha)", 0.0001, 1.0, 0.0001, 0.0001, format="%.4f")

        @st.cache_data
        def get_nn_data(name):
            if name == "Iris":    d = load_iris()
            elif name == "Wine":  d = load_wine()
            elif name == "Breast Cancer": d = load_breast_cancer()
            else:
                X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
                return pd.DataFrame(X, columns=["F1","F2"]), pd.Series(y), ["C0","C1"]
            return pd.DataFrame(d.data, columns=d.feature_names), pd.Series(d.target), list(d.target_names)

        @st.cache_resource
        def train_nn(ds, h1_, h2_, act, lr_, alph):
            X, y, names = get_nn_data(ds)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            sc = StandardScaler()
            layers = (h1_,) if h2_ == 0 else (h1_, h2_)
            nn = MLPClassifier(hidden_layer_sizes=layers, activation=act,
                               learning_rate_init=lr_, alpha=alph,
                               max_iter=500, random_state=42)
            nn.fit(sc.fit_transform(Xtr), ytr)
            yp = nn.predict(sc.transform(Xte))
            return nn, sc, Xtr, Xte, ytr, yte, yp, names

        nn_model, nn_sc, Xtr, Xte, ytr, yte, yp, nn_names = train_nn(
            nn_dataset, h1, h2, activation, lr, nn_alpha)

        nn_acc = accuracy_score(yte, yp)
        c1, c2, c3 = st.columns(3)
        c1.metric("Test Accuracy", f"{nn_acc:.1%}")
        c2.metric("Architecture", f"{'→'.join(str(s) for s in nn_model.hidden_layer_sizes)}")
        c3.metric("Epochs run", len(nn_model.loss_curve_))

        cA, cB = st.columns(2)
        with cA:
            # Loss curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=nn_model.loss_curve_, mode='lines',
                line=dict(color='#6366f1', width=2.5), name='Training Loss'))
            fig.update_layout(**DARK_TEMPLATE, height=300,
                title="Training Loss Curve",
                xaxis_title="Epoch", yaxis_title="Loss",
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        with cB:
            cm_nn = confusion_matrix(yte, yp)
            fig2 = px.imshow(cm_nn, text_auto=True, color_continuous_scale="Blues",
                x=nn_names, y=nn_names, title="Confusion Matrix")
            fig2.update_layout(**DARK_TEMPLATE, height=300, coloraxis_showscale=False)
            dark_fig(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("#### 🧪 Activation Functions Visualized")
        concept("""
        The <strong>activation function</strong> introduces non-linearity into the network.
        Without it, stacking linear layers would still only produce a linear function —
        we couldn't learn any complex patterns.
        """)
        x_act = np.linspace(-4, 4, 300)
        activations = {
            "ReLU":    np.maximum(0, x_act),
            "Sigmoid": 1 / (1 + np.exp(-x_act)),
            "Tanh":    np.tanh(x_act),
            "Leaky ReLU": np.where(x_act > 0, x_act, 0.1 * x_act),
            "Swish":   x_act * (1 / (1 + np.exp(-x_act))),
        }
        fig = go.Figure()
        for (name_a, vals), col in zip(activations.items(), COLORS):
            fig.add_trace(go.Scatter(x=x_act, y=vals, mode='lines', name=name_a,
                line=dict(color=col, width=2.5)))
        fig.add_vline(x=0, line=dict(color='#334155', width=1))
        fig.add_hline(y=0, line=dict(color='#334155', width=1))
        fig.update_layout(**DARK_TEMPLATE, height=380,
            title="Common Activation Functions",
            xaxis_title="Input z", yaxis_title="Activation f(z)",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(range=[-1.5, 4], gridcolor="#1e293b"))
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        act_notes = {
            "Sigmoid": "Output: (0,1). Great for binary output layer. Vanishing gradient problem for deep nets.",
            "Tanh": "Output: (-1,1). Zero-centered — better than sigmoid for hidden layers. Still suffers vanishing gradient.",
            "ReLU": "Most popular. Fast to compute. Doesn't saturate for positive inputs. Can cause 'dying ReLU' issue.",
            "Leaky ReLU": "Fixes dying ReLU by allowing small negative slope. Usually works better than plain ReLU.",
            "Swish": "Google's smooth activation. Often outperforms ReLU in deep networks. Used in EfficientNet.",
        }
        for name_a, note in act_notes.items():
            st.markdown(f"**{name_a}** — {note}")


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 6 — MODEL COMPARISON LAB
# ═══════════════════════════════════════════════════════════════════════════
elif "Comparison" in module:
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Module 6: <span>Model Comparison Lab</span> 🔬</p>
        <p class="hero-sub">Benchmark every classifier on the same dataset — understand the bias-variance tradeoff</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---\n### ⚙️ Benchmark Settings")
        bench_ds = st.selectbox("Dataset", ["Iris", "Wine", "Breast Cancer", "Moons"])

    @st.cache_data
    def get_bench_data(name):
        if name == "Iris":    d = load_iris()
        elif name == "Wine":  d = load_wine()
        elif name == "Breast Cancer": d = load_breast_cancer()
        else:
            X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
            return np.array(X), np.array(y), ["C0","C1"]
        return d.data, d.target, list(d.target_names)

    @st.cache_data
    def benchmark_all(ds):
        X, y, names = get_bench_data(ds)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        Xtr, Xte, ytr, yte = train_test_split(X_s, y, test_size=0.2, stratify=y, random_state=42)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
            "kNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
            "SVM (RBF)":           SVC(probability=True, random_state=42),
            "Naive Bayes":         GaussianNB(),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
            "MLP Neural Net":      MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42),
        }
        results = []
        for name_m, model in models.items():
            model.fit(Xtr, ytr)
            yp    = model.predict(Xte)
            acc   = accuracy_score(yte, yp)
            cv    = cross_val_score(model, X_s, y, cv=5)
            results.append({
                "Model":       name_m,
                "Test Acc":    round(acc, 4),
                "CV Mean":     round(cv.mean(), 4),
                "CV Std":      round(cv.std(), 4),
                "Train Acc":   round(accuracy_score(ytr, model.predict(Xtr)), 4),
            })
        return pd.DataFrame(results).sort_values("CV Mean", ascending=False)

    with st.spinner("Benchmarking all models..."):
        df_bench = benchmark_all(bench_ds)

    best = df_bench.iloc[0]
    st.markdown(f"""
    <div class="callout-green">
    🏆 <strong>Best model: {best['Model']}</strong> — CV accuracy {best['CV Mean']:.1%} ± {best['CV Std']:.1%}
    </div>
    """, unsafe_allow_html=True)

    cA, cB = st.columns(2)
    with cA:
        st.markdown("#### 📊 Model Rankings")
        df_display = df_bench.copy()
        df_display["Overfit Gap"] = (df_display["Train Acc"] - df_display["CV Mean"]).round(4)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    with cB:
        fig = go.Figure()
        sorted_df = df_bench.sort_values("CV Mean")
        fig.add_trace(go.Bar(
            x=sorted_df["CV Mean"], y=sorted_df["Model"],
            orientation='h', name="CV Mean Accuracy",
            marker_color=COLORS[0],
            error_x=dict(type='data', array=sorted_df["CV Std"], color='#64748b'),
        ))
        fig.add_trace(go.Bar(
            x=sorted_df["Test Acc"], y=sorted_df["Model"],
            orientation='h', name="Test Accuracy",
            marker_color=COLORS[1], opacity=0.7,
        ))
        fig.update_layout(**DARK_TEMPLATE, height=380,
            title="Accuracy Comparison (all models)",
            xaxis=dict(range=[0.5, 1.02], title="Accuracy", gridcolor="#1e293b"),
            barmode='group',
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        dark_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown("#### 🕸️ Model Radar Chart — Multi-Metric Comparison")
    categories = ["Test Acc", "CV Mean", "Low Overfit"]
    df_bench["Low Overfit"] = 1 - (df_bench["Train Acc"] - df_bench["Test Acc"]).clip(0, 1)

    fig_r = go.Figure()
    for _, row in df_bench.head(5).iterrows():
        vals = [row["Test Acc"], row["CV Mean"], row["Low Overfit"]]
        vals += [vals[0]]
        cats = categories + [categories[0]]
        fig_r.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill='toself', name=row["Model"],
            opacity=0.7
        ))
    fig_r.update_layout(**DARK_TEMPLATE, height=420,
        polar=dict(
            bgcolor="#0f172a",
            radialaxis=dict(range=[0.5,1.05], gridcolor="#1e293b", color="#64748b"),
            angularaxis=dict(gridcolor="#1e293b", color="#94a3b8"),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("#### 💡 Key Takeaways: Bias-Variance Tradeoff")
    c1, c2 = st.columns(2)
    with c1:
        concept("""
        <strong>Bias-Variance Tradeoff:</strong><br>
        Every model makes a tradeoff between two error sources:<br>
        🔹 <strong>Bias</strong> — error from wrong assumptions (too simple model → underfitting)<br>
        🔹 <strong>Variance</strong> — error from sensitivity to noise (too complex model → overfitting)<br><br>
        Ensemble methods (Random Forest, Gradient Boosting) reduce variance through averaging.
        """)
    with c2:
        concept("""
        <strong>No Free Lunch Theorem:</strong><br>
        No single algorithm is best for all problems. Performance depends on:<br>
        🔹 Size and quality of training data<br>
        🔹 Dimensionality of features<br>
        🔹 Class balance<br>
        🔹 Signal-to-noise ratio in the data<br><br>
        Always benchmark multiple models on your specific dataset!
        """)


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 7 — BUILD YOUR OWN DATASET
# ═══════════════════════════════════════════════════════════════════════════
elif "Build" in module:
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Module 7: <span>Build Your Own Dataset</span> 🛠️</p>
        <p class="hero-sub">Generate custom data, choose your model, train, evaluate, and download results</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---\n### 🛠️ Data Generator")
        task_type  = st.radio("Task Type", ["Classification", "Regression"])
        n_samp     = st.slider("Samples", 100, 2000, 500, 50)
        n_feat     = st.slider("Features", 2, 20, 5)
        n_info     = st.slider("Informative features", 1, min(n_feat, 15), min(3, n_feat))
        noise_d    = st.slider("Noise", 0.0, 1.0, 0.2, 0.05)
        test_s     = st.slider("Test split %", 10, 40, 20)

        st.markdown("---\n### 🤖 Model")
        if task_type == "Classification":
            n_cl = st.slider("Number of classes", 2, 5, 2)
            chosen_model = st.selectbox("Model", [
                "Random Forest", "Gradient Boosting", "Logistic Regression",
                "Decision Tree", "K-Nearest Neighbors", "SVM", "MLP Neural Net"
            ])
        else:
            chosen_model = st.selectbox("Model", [
                "Random Forest", "Gradient Boosting", "Linear Regression",
                "Ridge Regression", "Decision Tree", "SVR"
            ])

    @st.cache_data
    def generate_data(task, n, nf, ni, noise, seed=42, n_classes=2):
        if task == "Classification":
            X, y = make_classification(n_samples=n, n_features=nf, n_informative=ni,
                n_redundant=max(0, nf-ni-1), n_classes=n_classes,
                n_clusters_per_class=1, flip_y=noise, random_state=seed)
        else:
            X, y = make_regression(n_samples=n, n_features=nf, n_informative=ni,
                noise=noise*50, random_state=seed)
        return X, y

    X_gen, y_gen = generate_data(task_type, n_samp, n_feat, n_info, noise_d,
                                  n_classes=(3 if task_type=="Classification" else 2))

    st.markdown(f"#### 📦 Generated Dataset: {n_samp} samples × {n_feat} features")
    df_gen = pd.DataFrame(X_gen, columns=[f"feature_{i+1}" for i in range(n_feat)])
    df_gen["target"] = y_gen

    tab1, tab2, tab3 = st.tabs(["📊 Explore Data", "🔬 Train & Results", "⬇️ Export"])

    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            # PCA 2D
            pca2 = PCA(n_components=2)
            X_p2 = pca2.fit_transform(StandardScaler().fit_transform(X_gen))
            fig = px.scatter(x=X_p2[:,0], y=X_p2[:,1],
                color=y_gen.astype(str),
                title=f"Dataset — PCA 2D View  ({pca2.explained_variance_ratio_.sum():.1%} variance)",
                labels={"x":"PC1","y":"PC2","color":"Target"},
                color_discrete_sequence=COLORS)
            fig.update_layout(**DARK_TEMPLATE, height=380,
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(df_gen.describe().round(3), use_container_width=True)

        # Feature correlations
        corr = df_gen.corr()
        fig_c = px.imshow(corr, color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix", zmin=-1, zmax=1)
        fig_c.update_layout(**DARK_TEMPLATE, height=420)
        dark_fig(fig_c)
        st.plotly_chart(fig_c, use_container_width=True)

    with tab2:
        X_d, y_d = X_gen, y_gen
        sc = StandardScaler()
        Xtr, Xte, ytr, yte = train_test_split(X_d, y_d, test_size=test_s/100, random_state=42)
        Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

        if task_type == "Classification":
            m_map = {
                "Random Forest":      RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting":  GradientBoostingClassifier(n_estimators=100, random_state=42),
                "Logistic Regression":LogisticRegression(max_iter=1000),
                "Decision Tree":      DecisionTreeClassifier(max_depth=8, random_state=42),
                "K-Nearest Neighbors":KNeighborsClassifier(n_neighbors=5),
                "SVM":                SVC(probability=True, random_state=42),
                "MLP Neural Net":     MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42),
            }
            the_model = m_map[chosen_model]
            the_model.fit(Xtr_s, ytr)
            yp = the_model.predict(Xte_s)
            cv = cross_val_score(the_model, sc.fit_transform(X_d), y_d, cv=5)

            acc = accuracy_score(yte, yp)
            c1, c2, c3 = st.columns(3)
            c1.metric("Test Accuracy", f"{acc:.1%}")
            c2.metric("CV Mean ± Std", f"{cv.mean():.1%} ± {cv.std():.2%}")
            c3.metric("Train Accuracy", f"{accuracy_score(ytr, the_model.predict(Xtr_s)):.1%}")

            cA, cB = st.columns(2)
            with cA:
                n_cls_found = len(np.unique(y_d))
                cls_names = [str(i) for i in range(n_cls_found)]
                cm_g = confusion_matrix(yte, yp)
                fig_cm = px.imshow(cm_g, text_auto=True, color_continuous_scale="Blues",
                    title="Confusion Matrix")
                fig_cm.update_layout(**DARK_TEMPLATE, height=320, coloraxis_showscale=False)
                dark_fig(fig_cm)
                st.plotly_chart(fig_cm, use_container_width=True)
            with cB:
                if hasattr(the_model, "feature_importances_"):
                    fi_vals = the_model.feature_importances_
                    fi_df = pd.DataFrame({"feature":[f"f_{i+1}" for i in range(n_feat)], "importance":fi_vals})
                    fi_df = fi_df.sort_values("importance", ascending=False).head(10)
                    fig_fi = px.bar(fi_df, x="importance", y="feature", orientation='h',
                        title="Top Feature Importances", color="importance",
                        color_continuous_scale="Blues")
                    fig_fi.update_layout(**DARK_TEMPLATE, height=320, coloraxis_showscale=False,
                        yaxis=dict(autorange="reversed"))
                    dark_fig(fig_fi)
                    st.plotly_chart(fig_fi, use_container_width=True)

        else:  # Regression
            r_map = {
                "Random Forest":   RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting":GradientBoostingClassifier(n_estimators=100, random_state=42) if False
                                    else __import__('sklearn.ensemble', fromlist=['GradientBoostingRegressor']).GradientBoostingRegressor(n_estimators=100, random_state=42),
                "Linear Regression": LinearRegression(),
                "Ridge Regression":  Ridge(alpha=1.0),
                "Decision Tree":   DecisionTreeRegressor(max_depth=8, random_state=42),
                "SVR":             SVR(C=1.0),
            }
            the_model = r_map[chosen_model]
            the_model.fit(Xtr_s, ytr)
            yp = the_model.predict(Xte_s)

            r2   = r2_score(yte, yp)
            rmse = np.sqrt(mean_squared_error(yte, yp))
            c1, c2, c3 = st.columns(3)
            c1.metric("Test R²", f"{r2:.4f}")
            c2.metric("RMSE", f"{rmse:.3f}")
            c3.metric("Train R²", f"{r2_score(ytr, the_model.predict(Xtr_s)):.4f}")

            fig = px.scatter(x=yte, y=yp, title="Actual vs Predicted",
                labels={"x":"Actual","y":"Predicted"},
                color_discrete_sequence=["#6366f1"])
            mn, mx = min(yte.min(), yp.min()), max(yte.max(), yp.max())
            fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
                line=dict(color='#10b981', dash='dash'), name='Perfect'))
            fig.update_layout(**DARK_TEMPLATE, height=380,
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            dark_fig(fig)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### ⬇️ Download Your Results")
        col1, col2 = st.columns(2)
        with col1:
            csv_data = df_gen.to_csv(index=False)
            st.download_button("📥 Download Dataset (CSV)", csv_data,
                "my_dataset.csv", "text/csv",
                help="Download the generated dataset")
        with col2:
            if task_type == "Classification":
                pred_df = pd.DataFrame({"actual": yte, "predicted": yp})
            else:
                pred_df = pd.DataFrame({"actual": yte, "predicted": yp,
                                        "residual": yte - yp})
            pred_csv = pred_df.to_csv(index=False)
            st.download_button("📥 Download Predictions (CSV)", pred_csv,
                "predictions.csv", "text/csv",
                help="Download model predictions vs actuals")

        codeblock(
f"""# Replicate this experiment in your own Python environment
import numpy as np
from sklearn.datasets import make_{'classification' if task_type=='Classification' else 'regression'}
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Generate data
X, y = make_{'classification' if task_type=='Classification' else 'regression'}(
    n_samples={n_samp}, n_features={n_feat},
    n_informative={n_info}, random_state=42
)

# Preprocess
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={test_s/100:.2f})
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Train {chosen_model}
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"Accuracy: {{(predictions == y_test).mean():.3f}}")"""
        )

# Footer
st.markdown("""
<hr style='border-color:#1e293b; margin: 48px 0 16px'>
<div style='text-align:center; color:#334155; font-size:.82rem; padding-bottom:16px'>
    🤖 AI/ML Academy &nbsp;·&nbsp; Built with Streamlit & Scikit-learn &nbsp;·&nbsp;
    For educational use in introductory AI/ML courses &nbsp;·&nbsp;
    <span style='color:#475569'>Apache 2.0</span>
</div>
""", unsafe_allow_html=True)
