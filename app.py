import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎮 GameStat ML Pipeline",
    layout="wide",
    page_icon="🕹️"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Retro Arcade / Neon CRT aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117;
}
[data-testid="stHeader"] { background: transparent; }

.game-title {
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(16px, 2vw, 24px);
    text-align: center;
    color: #00e5b0;
    padding: 24px 10px 6px;
    letter-spacing: 3px;
}
.game-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: #888;
    text-align: center;
    margin-bottom: 16px;
    letter-spacing: 3px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 3px;
    border-bottom: 1px solid #00e5b033;
    padding: 4px 4px 0;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: #555;
    background: transparent;
    border: 1px solid #ffffff11;
    padding: 6px 10px;
    border-radius: 3px 3px 0 0;
}
.stTabs [aria-selected="true"] {
    color: #0d1117 !important;
    background: #00e5b0 !important;
}

h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: #00e5b0 !important;
}
h2 { font-size: 16px !important; letter-spacing: 1px; }
h3 { font-size: 13px !important; color: #e060a0 !important; }

p, label, .stMarkdown, div[data-testid="stText"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: #aac8b8 !important;
    font-size: 13px !important;
}

.stButton > button {
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    background: transparent;
    color: #00e5b0;
    border: 1px solid #00e5b0;
    border-radius: 3px;
    padding: 8px 18px;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #00e5b0;
    color: #0d1117;
}

.stSuccess { background: rgba(0,229,176,0.08) !important; border-left: 3px solid #00e5b0 !important; }
.stInfo    { background: rgba(0,100,255,0.08) !important; border-left: 3px solid #3080ff !important; }
.stWarning { background: rgba(255,180,0,0.08) !important; border-left: 3px solid #ffb400 !important; }
.stError   { background: rgba(255,60,80,0.08) !important; border-left: 3px solid #ff3c50 !important; }

.pixel-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00e5b066, transparent);
    margin: 14px 0;
}

.stat-card {
    background: rgba(0,229,176,0.05);
    border: 1px solid #00e5b033;
    border-radius: 4px;
    padding: 14px;
    text-align: center;
}
.stat-number {
    font-family: 'Orbitron', sans-serif;
    font-size: 24px;
    color: #00e5b0;
    display: block;
}
.stat-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: #778899;
    margin-top: 4px;
    display: block;
}
</style>

<div class="game-title">GAMESTAT ML PIPELINE</div>
<div class="game-subtitle">VIDEO GAME SALES ANALYSIS</div>
<div class="pixel-divider"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET COLUMNS INFO (for context)
# COLS: Name, Platform, Year_of_Release, Genre, Publisher, NA_Sales, EU_Sales,
#       JP_Sales, Other_Sales, Global_Sales, Critic_Score, Critic_Count,
#       User_Score, User_Count, Developer, Rating
# ─────────────────────────────────────────────────────────────────────────────
DATASET_URL = "Video_Games_Sales_as_at_22_Dec_2016.csv"
NUMERIC_COLS = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales",
                "Critic_Score", "Critic_Count", "User_Score", "User_Count", "Year_of_Release"]
CATEGORICAL_COLS = ["Platform", "Genre", "Publisher", "Developer", "Rating"]
SUGGESTED_TARGETS = {
    "Classification": ["Genre", "Rating", "Platform"],
    "Regression":     ["Global_Sales", "NA_Sales", "Critic_Score", "User_Score"]
}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
defaults = dict(df=None, target=None, task_type="Classification",
                model_choice=None, split_data=None)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: load & preprocess
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_vg_data(file):
    df = pd.read_csv(file)
    # Fix User_Score (sometimes 'tbd' string)
    df["User_Score"] = pd.to_numeric(df["User_Score"], errors="coerce")
    return df


def encode_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
_AXIS_STYLE = dict(gridcolor="rgba(0,229,176,0.1)", linecolor="rgba(0,229,176,0.2)")

PLOT_THEME = dict(
    paper_bgcolor="rgba(13,17,23,0)",
    plot_bgcolor="rgba(13,17,23,0.6)",
    font=dict(family="Share Tech Mono", color="#aac8b8"),
    title_font=dict(family="Orbitron", color="#00e5b0"),
    xaxis=_AXIS_STYLE,
    yaxis=_AXIS_STYLE,
    colorway=["#00e5b0", "#e060a0", "#f0b400", "#4080ff", "#a855f7", "#f97316"],
)

PLOT_THEME_NO_AXIS = {k: v for k, v in PLOT_THEME.items() if k not in ("xaxis", "yaxis")}
PLOT_TEMPLATE = {"layout": PLOT_THEME}  # legacy alias

NEON_COLORS = ["#00e5b0","#e060a0","#f0b400","#4080ff","#a855f7","#f97316"]
HEATMAP_SCALE = [[0, "#e060a0"], [0.5, "#0d1117"], [1, "#00e5b0"]]

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "1. Problem Type", "2. Data Input & PCA", "3. EDA",
    "4. Data Cleaning", "5. Feature Selection", "6. Data Split",
    "7. Model Selection", "8. Training & Validation", "9. Metrics", "10. Tuning"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Problem Type
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("1. SELECT GAME MODE")
    st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <p>Choose your <span style='color:#00ffcc;font-weight:bold;'>mission objective</span> for the Video Game Sales dataset.</p>
    <ul>
        <li><b style='color:#00ffcc;'>Classification</b> → Predict Genre, Platform, or Age Rating of a game</li>
        <li><b style='color:#ff0080;'>Regression</b> → Predict Global/Regional Sales, Critic Score, or User Score</li>
    </ul>
    """, unsafe_allow_html=True)

    st.session_state.task_type = st.radio(
        "Mission type:",
        ("Classification", "Regression"),
        horizontal=True
    )

    task = st.session_state.task_type
    emoji = "🎯" if task == "Classification" else "📈"
    color = "#00ffcc" if task == "Classification" else "#ff0080"
    st.markdown(f"""
    <div style='margin-top:20px; padding:16px; border:2px solid {color};
                background:rgba(255,255,255,0.03); box-shadow: 0 0 20px {color}44;
                font-family:Share Tech Mono; color:{color}; font-size:14px; letter-spacing:2px;'>
        {emoji} GAME MODE LOCKED IN: <b>{task.upper()}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <p>💡 <b>Suggested targets for {task}:</b> {', '.join(SUGGESTED_TARGETS[task])}</p>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Data Input & PCA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("2. LOAD SAVE FILE & VISUALIZE")
    st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

    st.info("📂 Upload the Video Game Sales CSV dataset (Video_Games_Sales_as_at_22_Dec_2016.csv)")
    uploaded_file = st.file_uploader("Insert Cartridge 🎮", type=["csv"])

    if uploaded_file is not None:
        raw_df = load_vg_data(uploaded_file)
        df_encoded = encode_df(raw_df)
        st.session_state.df = df_encoded
        st.session_state.raw_df = raw_df

        st.markdown("### 📋 RAW DATA PREVIEW")
        st.dataframe(raw_df.head(10))
        st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

        # Quick stats
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-card"><span class="stat-number">{raw_df.shape[0]:,}</span><span class="stat-label">TOTAL GAMES</span></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><span class="stat-number">{raw_df["Platform"].nunique()}</span><span class="stat-label">PLATFORMS</span></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-card"><span class="stat-number">{raw_df["Genre"].nunique()}</span><span class="stat-label">GENRES</span></div>', unsafe_allow_html=True)
        with c4:
            val = raw_df["Global_Sales"].sum()
            st.markdown(f'<div class="stat-card"><span class="stat-number">{val:.0f}M</span><span class="stat-label">GLOBAL SALES</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

        # Target selection — filtered to suggested
        st.markdown("### 🎯 SELECT TARGET FEATURE")
        task = st.session_state.task_type
        suggested = SUGGESTED_TARGETS[task]
        available_suggested = [c for c in suggested if c in df_encoded.columns]
        st.session_state.target = st.selectbox(
            f"Target (for {task}):",
            df_encoded.columns.tolist(),
            index=df_encoded.columns.tolist().index(available_suggested[0]) if available_suggested else 0
        )

        st.markdown("### 🔭 PCA WARP VISUALIZATION")
        numeric_feats = [c for c in NUMERIC_COLS if c in df_encoded.columns and c != st.session_state.target]
        features = st.multiselect("Select features for PCA:", numeric_feats, default=numeric_feats[:5])

        if len(features) >= 2:
            X_pca = df_encoded[features].dropna()
            pca = PCA(n_components=2)
            result = pca.fit_transform(StandardScaler().fit_transform(X_pca))
            pca_df = pd.DataFrame(result, columns=["PC1", "PC2"])

            # Color by Genre (raw) for readability
            if "Genre" in raw_df.columns:
                pca_df["Genre"] = raw_df["Genre"].iloc[X_pca.index].values
                color_col = "Genre"
            else:
                pca_df[st.session_state.target] = df_encoded[st.session_state.target].iloc[X_pca.index].values
                color_col = st.session_state.target

            fig = px.scatter(
                pca_df, x="PC1", y="PC2", color=color_col,
                title="HYPERSPACE SCAN — 2D PCA of Selected Features",
                color_discrete_sequence=NEON_COLORS,
                opacity=0.7,
                template="plotly_dark"
            )
            fig.update_layout(**PLOT_THEME)
            fig.update_traces(marker=dict(size=4))
            st.plotly_chart(fig, use_container_width=True)

            exp_var = pca.explained_variance_ratio_
            st.markdown(f"🔋 **Explained Variance:** PC1 = `{exp_var[0]:.2%}` | PC2 = `{exp_var[1]:.2%}`")
        else:
            st.warning("Pick at least 2 features to fire up PCA.")
    else:
        st.info("Waiting for cartridge… Upload the CSV to begin! 🕹️")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("3. INTEL BRIEFING — EDA")
    st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

    if st.session_state.df is not None:
        # Always use the current (possibly cleaned) encoded df + the raw_df for labels
        df = st.session_state.df
        raw_df = st.session_state.get("raw_df", df)

        # Build a working df with original string labels where possible,
        # aligned to the current df index after any cleaning/outlier removal
        working_raw = raw_df.loc[raw_df.index.isin(df.index)].reindex(df.index)

        n_rows = len(df)
        st.info(f"📊 Currently working with **{n_rows:,} rows** (reflects any cleaning done in Tab 4)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Statistics Summary")
            num_cols_present = [c for c in NUMERIC_COLS if c in df.columns]
            st.dataframe(df[num_cols_present].describe().round(2))

        with col2:
            st.markdown("### Correlation Heatmap")
            num_df = df[num_cols_present].dropna()
            corr = num_df.corr()
            fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                            color_continuous_scale=HEATMAP_SCALE,
                            title="Feature Correlation Matrix",
                            template="plotly_dark")
            fig.update_layout(**PLOT_THEME)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("### Sales by Genre")
            if "Genre" in working_raw.columns and "Global_Sales" in working_raw.columns:
                genre_sales = working_raw.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False)
                fig2 = px.bar(genre_sales, x=genre_sales.index, y=genre_sales.values,
                              labels={"x": "Genre", "y": "Global Sales (M)"},
                              title=f"Sales by Genre  ({n_rows:,} records)",
                              color=genre_sales.values,
                              color_continuous_scale=["#e060a0", "#00e5b0"],
                              template="plotly_dark")
                fig2.update_layout(**PLOT_THEME)
                st.plotly_chart(fig2, use_container_width=True)

        with col4:
            st.markdown("### Top Platforms")
            if "Platform" in working_raw.columns and "Global_Sales" in working_raw.columns:
                plat_sales = working_raw.groupby("Platform")["Global_Sales"].sum().nlargest(12)
                fig3 = px.bar(plat_sales, x=plat_sales.values, y=plat_sales.index,
                              orientation="h",
                              labels={"x": "Sales (M)", "y": "Platform"},
                              title=f"Top 12 Platforms  ({n_rows:,} records)",
                              color=plat_sales.values,
                              color_continuous_scale=["#e060a0", "#f0b400", "#00e5b0"],
                              template="plotly_dark")
                fig3.update_layout(**PLOT_THEME)
                st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

        st.markdown("### Yearly Release Timeline")
        if "Year_of_Release" in working_raw.columns and "Global_Sales" in working_raw.columns:
            year_df = working_raw.dropna(subset=["Year_of_Release"]).copy()
            year_df["Year_of_Release"] = year_df["Year_of_Release"].astype(int)
            year_sales = year_df.groupby("Year_of_Release").agg(
                Total_Sales=("Global_Sales", "sum"),
                Num_Games=("Name", "count") if "Name" in year_df.columns else ("Global_Sales", "count")
            ).reset_index()
            year_sales = year_sales[year_sales["Year_of_Release"].between(1980, 2017)]

            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=year_sales["Year_of_Release"], y=year_sales["Total_Sales"],
                                      name="Global Sales (M)", line=dict(color="#00e5b0", width=2),
                                      fill="tozeroy", fillcolor="rgba(0,229,176,0.1)"))
            fig4.add_trace(go.Bar(x=year_sales["Year_of_Release"], y=year_sales["Num_Games"],
                                  name="# Titles Released", yaxis="y2",
                                  marker_color="rgba(224,96,160,0.4)"))
            fig4.update_layout(
                title=f"Sales & Releases Per Year  ({n_rows:,} records)",
                yaxis=dict(title="Global Sales (M)", color="#00e5b0",
                           gridcolor="rgba(0,229,176,0.1)", linecolor="rgba(0,229,176,0.2)"),
                yaxis2=dict(title="# Games Released", overlaying="y", side="right", color="#e060a0"),
                **PLOT_THEME_NO_AXIS
            )
            st.plotly_chart(fig4, use_container_width=True)

    else:
        st.info("Upload the dataset in Tab 2 to see analysis here.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Data Cleaning
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("4. BUG FIXES & PATCH NOTES")
    st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

    if st.session_state.df is not None:
        df = st.session_state.df

        # Missing value heatmap
        st.markdown("### 🕳️ MISSING DATA MAP")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
        missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing %", ascending=False)
        if not missing_df.empty:
            fig_m = px.bar(missing_df, x=missing_df.index, y="Missing %",
                           title="Missing Values per Column (%)",
                           color="Missing %", color_continuous_scale=["#00ffcc", "#ffc800", "#ff0080"],
                           template="plotly_dark")
            fig_m.update_layout(**PLOT_THEME)
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.success("✅ No missing values detected! Clean save file!")

        st.markdown("### 🔧 IMPUTATION PATCH")
        impute_method = st.selectbox("Imputation Strategy:", ("mean", "median", "most_frequent"))
        if st.button("⚙️ APPLY PATCH"):
            imputer = SimpleImputer(strategy=impute_method)
            df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            st.session_state.df = df_imputed
            st.success(f"✅ Patch applied using `{impute_method}` strategy!")

        st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🚨 ANOMALY DETECTION — OUTLIER BOSS FIGHT")
        outlier_method = st.selectbox("Detection Method:", ["None", "IQR", "Isolation Forest", "DBSCAN", "OPTICS"])

        if outlier_method != "None":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outliers = np.zeros(len(df), dtype=bool)

            if outlier_method == "IQR":
                Q1 = df[numeric_cols].quantile(0.25)
                Q3 = df[numeric_cols].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
            elif outlier_method == "Isolation Forest":
                iso = IsolationForest(contamination=0.05, random_state=42)
                outliers = iso.fit_predict(df[numeric_cols].fillna(0)) == -1
            elif outlier_method == "DBSCAN":
                db = DBSCAN(eps=0.5, min_samples=5)
                outliers = db.fit_predict(StandardScaler().fit_transform(df[numeric_cols].fillna(0))) == -1
            elif outlier_method == "OPTICS":
                optics = OPTICS(min_samples=5)
                outliers = optics.fit_predict(StandardScaler().fit_transform(df[numeric_cols].fillna(0))) == -1

            n_out = int(outliers.sum())
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f'<div class="stat-card"><span class="stat-number" style="color:#ff0080;">{n_out}</span><span class="stat-label">OUTLIERS DETECTED</span></div>', unsafe_allow_html=True)
            with col_b:
                st.markdown(f'<div class="stat-card"><span class="stat-number">{len(df) - n_out}</span><span class="stat-label">CLEAN RECORDS</span></div>', unsafe_allow_html=True)

            if st.button("💥 DEFEAT OUTLIERS"):
                st.session_state.df = df[~outliers].reset_index(drop=True)
                st.success(f"💥 {n_out} outliers defeated! {len(st.session_state.df)} records remain.")
    else:
        st.info("⬆️ Upload data in Tab 2 first!")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Feature Selection
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("5. POWER-UP SELECTION")
    st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

    if st.session_state.df is not None:
        df = st.session_state.df
        target = st.session_state.target

        if target and target in df.columns:
            X = df.drop(columns=[target])
            y = df[target]

            st.markdown(f"📌 **Target:** `{target}` | **Features available:** `{X.shape[1]}`")
            fs_method = st.selectbox("Feature Selection Weapon:",
                                     ["Variance Threshold", "Correlation Drop", "Information Gain"])

            if st.button("⚡ EQUIP FEATURES"):
                if fs_method == "Variance Threshold":
                    vt = VarianceThreshold(threshold=0.1)
                    vt.fit(X)
                    cols_to_keep = X.columns[vt.get_support()]
                elif fs_method == "Information Gain":
                    if st.session_state.task_type == "Classification":
                        ig = mutual_info_classif(X.fillna(0), y)
                    else:
                        ig = mutual_info_regression(X.fillna(0), y)
                    cols_to_keep = X.columns[ig > 0.01]
                elif fs_method == "Correlation Drop":
                    corr_matrix = X.corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
                    cols_to_keep = [c for c in X.columns if c not in to_drop]

                df_selected = pd.concat([X[cols_to_keep], y], axis=1)
                st.session_state.df = df_selected
                st.success(f"⚡ Features reduced: `{X.shape[1]}` → `{len(cols_to_keep)}`")
                st.write("🎒 **Equipped Features:**", list(cols_to_keep))

                # Feature importance bar (Information Gain only)
                if fs_method == "Information Gain":
                    ig_series = pd.Series(ig, index=X.columns).sort_values(ascending=False).head(15)
                    fig_ig = px.bar(ig_series, x=ig_series.values, y=ig_series.index,
                                    orientation="h", title="Feature Power Levels (Info Gain)",
                                    color=ig_series.values,
                                    color_continuous_scale=["#ff0080", "#ffc800", "#00ffcc"],
                                    template="plotly_dark")
                    fig_ig.update_layout(**PLOT_THEME)
                    st.plotly_chart(fig_ig, use_container_width=True)
        else:
            st.warning("Select a target in Tab 2 first!")
    else:
        st.info("⬆️ Upload data in Tab 2 first!")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Data Split
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("6. TEAM SPLIT — TRAIN vs TEST")
    st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

    if st.session_state.df is not None:
        df = st.session_state.df
        target = st.session_state.target

        st.markdown("⚔️ Divide your dataset into **Training Squad** and **Test Squad**.")
        test_size = st.slider("Test Size %", 10, 50, 20, help="Percentage of data for testing")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f'<div class="stat-card"><span class="stat-number">{100 - test_size}%</span><span class="stat-label">TRAINING SQUAD</span></div>', unsafe_allow_html=True)
        with col_r:
            st.markdown(f'<div class="stat-card"><span class="stat-number" style="color:#ff0080;">{test_size}%</span><span class="stat-label">TEST SQUAD</span></div>', unsafe_allow_html=True)

        if st.button("⚔️ SPLIT INTO TEAMS"):
            if target and target in df.columns:
                X = df.drop(columns=[target])
                y = df[target]
                # Drop rows where target is NaN before splitting
                mask = y.notna()
                X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size / 100, random_state=42
                )
                st.session_state.split_data = (X_train, X_test, y_train, y_test)
                st.success(f"✅ Split complete! Train: `{X_train.shape[0]}` | Test: `{X_test.shape[0]}` | Features: `{X_train.shape[1]}`")
            else:
                st.error("No target selected! Go back to Tab 2.")
    else:
        st.info("⬆️ Upload data in Tab 2 first!")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Model Selection
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("7. CHOOSE YOUR CHARACTER")
    st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

    task = st.session_state.task_type

    model_info = {
        "Classification": {
            "Logistic Regression":       ("⚡ Fast & linear. Great baseline.", "#00ffcc"),
            "SVC (Linear)":              ("🛡️ Solid linear boundary fighter.", "#0080ff"),
            "SVC (RBF)":                 ("🔮 Non-linear magic. Slower but powerful.", "#a855f7"),
            "Random Forest Classifier":  ("🌲 Ensemble of decision trees. Reliable.", "#ffc800"),
        },
        "Regression": {
            "Linear Regression":         ("📏 Simple & fast linear predictor.", "#00ffcc"),
            "SVR (Linear)":              ("🛡️ Linear SVR — robust to noise.", "#0080ff"),
            "SVR (RBF)":                 ("🔮 Non-linear regression powerhouse.", "#a855f7"),
            "Random Forest Regressor":   ("🌲 Best-in-class ensemble regressor.", "#ffc800"),
        }
    }

    model_list = list(model_info[task].keys())
    model_choice = st.selectbox("Select your fighter:", model_list)
    st.session_state.model_choice = model_choice

    desc, clr = model_info[task][model_choice]
    st.markdown(f"""
    <div style='margin-top:16px; padding:14px; border:2px solid {clr};
                background:rgba(255,255,255,0.03); font-family:Share Tech Mono;
                color:{clr}; font-size:13px; box-shadow: 0 0 15px {clr}44;'>
        {desc}
    </div>
    """, unsafe_allow_html=True)




# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — Training & Validation
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("8. TRAINING ARC — K-FOLD VALIDATION")
    st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

    k_folds = st.number_input("K (Number of training rounds):", min_value=2, max_value=20, value=5)
    st.session_state.k_folds = k_folds

    st.markdown(f"""
    <p>🎮 Your model will be trained <b style='color:#00ffcc;'>{k_folds} times</b> on different data folds.
    This makes it battle-hardened and less likely to overfit (a.k.a. <i>memorise</i> instead of <i>learn</i>).</p>
    """, unsafe_allow_html=True)

    if 'split_data' in st.session_state and st.session_state.split_data:
        X_train, _, y_train, _ = st.session_state.split_data
        st.markdown(f"📦 **Training samples:** `{X_train.shape[0]}` | **Features:** `{X_train.shape[1]}`")
    else:
        st.info("⬆️ Split data in Tab 6 to see training stats.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — Metrics
# ══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.header("9. SCOREBOARD — PERFORMANCE METRICS")
    st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

    MODEL_MAP = {
        "Linear Regression":        lambda: LinearRegression(),
        "Logistic Regression":      lambda: LogisticRegression(max_iter=1000),
        "SVR (Linear)":             lambda: SVR(kernel="linear"),
        "SVR (RBF)":                lambda: SVR(kernel="rbf"),
        "SVC (Linear)":             lambda: SVC(kernel="linear"),
        "SVC (RBF)":                lambda: SVC(kernel="rbf"),
        "Random Forest Regressor":  lambda: RandomForestRegressor(random_state=42),
        "Random Forest Classifier": lambda: RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_leaf=5,
    min_samples_split=10,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
),
    }

    if "split_data" in st.session_state and st.session_state.split_data:
        X_train, X_test, y_train, y_test = st.session_state.split_data
        choice = st.session_state.get("model_choice", "")

        if st.button("🚀 START BATTLE — TRAIN & EVALUATE"):
            if not choice:
                st.error("Select a model in Tab 7 first!")
            else:
                model_fn = MODEL_MAP.get(choice)
                model = model_fn()

                if choice == "K-Means (Clustering)":
                    st.warning("K-Means is unsupervised — it finds clusters, not predictions. No accuracy score.")
                    model.fit(X_train.fillna(0))
                    labels = model.predict(X_test.fillna(0))
                    pca = PCA(n_components=2)
                    X_vis = pca.fit_transform(StandardScaler().fit_transform(X_test.fillna(0)))
                    cluster_df = pd.DataFrame(X_vis, columns=["D1", "D2"])
                    cluster_df["Cluster"] = labels.astype(str)
                    fig_c = px.scatter(cluster_df, x="D1", y="D2", color="Cluster",
                                       title="K-Means Cluster Map (PCA)",
                                       color_discrete_sequence=NEON_COLORS, template="plotly_dark")
                    fig_c.update_layout(**PLOT_THEME)
                    st.plotly_chart(fig_c, use_container_width=True)
                else:
                    k = getattr(st.session_state, "k_folds", 5)
                    kf = KFold(n_splits=k, shuffle=True, random_state=42)
                    scoring = "accuracy" if st.session_state.task_type == "Classification" else "r2"

                    with st.spinner("Training in progress… ⚔️"):
                        cv_results = cross_validate(model, X_train.fillna(0), y_train,
                                                    cv=kf, return_train_score=True, scoring=scoring)

                    train_score = cv_results["train_score"].mean()
                    val_score = cv_results["test_score"].mean()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f'<div class="stat-card"><span class="stat-number">{train_score:.3f}</span><span class="stat-label">TRAIN SCORE</span></div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="stat-card"><span class="stat-number">{val_score:.3f}</span><span class="stat-label">VAL SCORE ({k}-FOLD)</span></div>', unsafe_allow_html=True)
                    with col3:
                        gap = train_score - val_score
                        gclr = "#ff0080" if gap > 0.1 else "#00ffcc"
                        st.markdown(f'<div class="stat-card"><span class="stat-number" style="color:{gclr};">{gap:.3f}</span><span class="stat-label">OVERFIT GAP</span></div>', unsafe_allow_html=True)

                    st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

                    if train_score > val_score + 0.1:
                        st.error("🚨 OVERFITTING DETECTED — Your model memorised the training data like a tryhard!")
                    elif train_score < 0.5:
                        st.warning("⚠️ UNDERFITTING — Model is struggling. Try a more powerful character!")
                    else:
                        st.success("✅ Model generalised well! Balanced fighter detected.")

                    # Final test evaluation
                    model.fit(X_train.fillna(0), y_train)
                    y_pred = model.predict(X_test.fillna(0))

                    st.markdown("### 🏆 FINAL BOSS SCORE — TEST SET")
                    if st.session_state.task_type == "Classification":
                        acc = accuracy_score(y_test, y_pred)
                        st.markdown(f'<div class="stat-card" style="max-width:300px;margin:auto;"><span class="stat-number">{acc:.4f}</span><span class="stat-label">TEST ACCURACY</span></div>', unsafe_allow_html=True)
                    else:
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        c_r, c_m = st.columns(2)
                        with c_r:
                            st.markdown(f'<div class="stat-card"><span class="stat-number">{r2:.4f}</span><span class="stat-label">R² SCORE</span></div>', unsafe_allow_html=True)
                        with c_m:
                            st.markdown(f'<div class="stat-card"><span class="stat-number">{mse:.4f}</span><span class="stat-label">MSE</span></div>', unsafe_allow_html=True)

                    # Fold scores chart
                    fold_df = pd.DataFrame({
                        "Fold": list(range(1, k+1)),
                        "Train": cv_results["train_score"],
                        "Validation": cv_results["test_score"]
                    })
                    fig_fold = go.Figure()
                    fig_fold.add_trace(go.Scatter(x=fold_df["Fold"], y=fold_df["Train"],
                                                  name="Train", line=dict(color="#00ffcc", width=2),
                                                  mode="lines+markers"))
                    fig_fold.add_trace(go.Scatter(x=fold_df["Fold"], y=fold_df["Validation"],
                                                  name="Validation", line=dict(color="#ff0080", width=2),
                                                  mode="lines+markers"))
                    fig_fold.update_layout(title="K-Fold Score per Round",
                                           xaxis_title="Fold", yaxis_title="Score",
                                           **PLOT_THEME)
                    st.plotly_chart(fig_fold, use_container_width=True)
    else:
        st.info("⬆️ Complete Tabs 2 → 6 to unlock the scoreboard!")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — Hyperparameter Tuning
# ══════════════════════════════════════════════════════════════════════════════
with tabs[9]:
    st.header("10. POWER LEVEL UP — HYPERPARAMETER TUNING")
    st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)

    if "split_data" in st.session_state and st.session_state.split_data:
        X_train, X_test, y_train, y_test = st.session_state.split_data
        choice = st.session_state.get("model_choice", "")

        tune_method = st.radio("Tuning Strategy:", ("GridSearchCV", "RandomizedSearchCV"), horizontal=True)

        st.markdown("""
        <p>🔬 <b>Random Forest only</b> for this demo — but the <code>param_grid</code> dict in the code
        is easy to extend for SVM or Linear models. Go hack it! 🛠️</p>
        """, unsafe_allow_html=True)

        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }

        st.markdown("**Param Grid being tested:**")
        for k, v in param_grid.items():
            st.markdown(f"- `{k}`: {v}")

        if st.button("⚗️ ACTIVATE HYPER TUNE"):
            if "Random Forest" not in choice:
                st.warning("⚠️ Tuning demo only works with Random Forest. Select it in Tab 7!")
            else:
                model = (RandomForestClassifier(random_state=42)
                         if st.session_state.task_type == "Classification"
                         else RandomForestRegressor(random_state=42))

                search_cls = GridSearchCV if tune_method == "GridSearchCV" else RandomizedSearchCV
                kwargs = dict(cv=3, scoring="accuracy" if st.session_state.task_type == "Classification" else "r2")
                if tune_method == "RandomizedSearchCV":
                    kwargs["n_iter"] = 6
                search = search_cls(model, param_grid, **kwargs)

                with st.spinner("Tuning in progress… hold on tight ⚙️"):
                    search.fit(X_train.fillna(0), y_train)

                st.success(f"🏆 Best parameters found!")
                for pk, pv in search.best_params_.items():
                    st.markdown(f"- `{pk}` = **{pv}**")

                st.markdown(f"🎯 **Best CV Score:** `{search.best_score_:.4f}`")

                y_pred_tuned = search.best_estimator_.predict(X_test.fillna(0))
                if st.session_state.task_type == "Classification":
                    final = accuracy_score(y_test, y_pred_tuned)
                    label = "Tuned Test Accuracy"
                else:
                    final = r2_score(y_test, y_pred_tuned)
                    label = "Tuned Test R²"

                st.markdown(f'<div class="stat-card" style="max-width:300px;margin:20px auto;"><span class="stat-number">{final:.4f}</span><span class="stat-label">{label.upper()}</span></div>', unsafe_allow_html=True)

                # Param importance heatmap from cv_results
                cv_df = pd.DataFrame(search.cv_results_)
                if "param_n_estimators" in cv_df.columns and "param_max_depth" in cv_df.columns:
                    pivot = cv_df.pivot_table(
                        values="mean_test_score",
                        index="param_max_depth",
                        columns="param_n_estimators",
                        aggfunc="mean"
                    )
                    fig_hp = px.imshow(pivot, text_auto=".3f",
                                       title="Score Heatmap: max_depth vs n_estimators",
                                       color_continuous_scale=HEATMAP_SCALE,
                                       template="plotly_dark")
                    fig_hp.update_layout(**PLOT_THEME)
                    st.plotly_chart(fig_hp, use_container_width=True)
    else:
        st.info("⬆️ Complete earlier tabs to unlock tuning!")

# Footer
st.markdown('<div class="pixel-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; font-family: Press Start 2P, monospace; font-size:9px;
            color:#00ffcc44; padding:20px; letter-spacing:2px;'>
    GAMESTAT ML PIPELINE v1.0 
</div>
""", unsafe_allow_html=True)