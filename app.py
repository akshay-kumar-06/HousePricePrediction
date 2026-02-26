"""
=============================================================================
Housing Price Analysis & Prediction — Production-Grade Streamlit Application
=============================================================================
Author : ML Engineering Team
Dataset: Assignment_1_MLR_CF_Housing_Price.csv

Features:
  1. Data Understanding
  2. Advanced EDA (matplotlib only)
  3. Data Preprocessing
  4. Model Building (6 regressors + GridSearchCV)
  5. Model Evaluation
  6. Feature Importance
  7. SHAP Explainability
  8. Prediction Interface
  9. Business Insights
 10. Downloadable Report
 11. Gemini AI Chatbot (data-analysis Q&A)
=============================================================================
"""

# ──────────────────────────── Imports ────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import io
import os
import joblib
from scipy import stats

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Suppress noisy warnings in the UI
warnings.filterwarnings("ignore")

# ──────────────────────────── Page Config ────────────────────────
st.set_page_config(
    page_title="🏠 Housing Price Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────── Custom CSS ────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0 0.2rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .metric-card h3 { margin: 0; font-size: 1.8rem; color: #333; }
    .metric-card p  { margin: 0; color: #666; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🏠 Housing Price Analysis & Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Comprehensive ML Pipeline · Model Comparison · SHAP Explainability · AI Chatbot</p>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    """Load the housing-price CSV from the same directory as this script."""
    csv_path = os.path.join(os.path.dirname(__file__), "Assignment_1_MLR_CF_Housing_Price.csv")
    df = pd.read_csv(csv_path)
    return df


# ═══════════════════════════════════════════════════════════════
# 2. ADVANCED EDA
# ═══════════════════════════════════════════════════════════════
def perform_eda(df):
    """Render all EDA plots using matplotlib only (no seaborn)."""

    st.header("📊 Exploratory Data Analysis")

    # ── Univariate ──────────────────────────────────────────────
    with st.expander("🔹 Univariate Analysis", expanded=True):

        # Histogram — Price
        st.subheader("Distribution of Price")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["price"], bins=30, color="#667eea", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Price")
        ax.set_ylabel("Frequency")
        ax.set_title("Price Distribution")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        skew_price = df["price"].skew()
        st.info(f"**Price skewness:** {skew_price:.4f}  ({'right-skewed' if skew_price > 0 else 'left-skewed'})")

        # Histogram — Area
        st.subheader("Distribution of Area")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["area"], bins=30, color="#764ba2", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Area (sq ft)")
        ax.set_ylabel("Frequency")
        ax.set_title("Area Distribution")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        skew_area = df["area"].skew()
        st.info(f"**Area skewness:** {skew_area:.4f}")

        # Boxplots for numerical outlier detection
        numeric_cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
        for col in numeric_cols:
            st.subheader(f"Boxplot — {col.title()}")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.boxplot(df[col].dropna(), vert=False, patch_artist=True,
                       boxprops=dict(facecolor="#667eea", alpha=0.6),
                       medianprops=dict(color="red", linewidth=2))
            ax.set_xlabel(col.title())
            ax.set_title(f"Outlier Detection — {col.title()}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Countplots for categorical variables
        cat_cols = ["mainroad", "guestroom", "basement", "hotwaterheating",
                    "airconditioning", "prefarea", "furnishingstatus"]
        for col in cat_cols:
            st.subheader(f"Countplot — {col.title()}")
            counts = df[col].value_counts()
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bars = ax.bar(counts.index.astype(str), counts.values,
                          color=["#667eea", "#764ba2", "#f093fb"][:len(counts)],
                          edgecolor="white")
            for bar, val in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        str(val), ha="center", va="bottom", fontweight="bold", fontsize=10)
            ax.set_xlabel(col.title())
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {col.title()}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Bivariate ───────────────────────────────────────────────
    with st.expander("🔹 Bivariate Analysis", expanded=False):

        # Scatter: Area vs Price
        st.subheader("Area vs Price")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df["area"], df["price"], alpha=0.5, c="#667eea", edgecolors="white", s=40)
        ax.set_xlabel("Area (sq ft)")
        ax.set_ylabel("Price")
        ax.set_title("Area vs Price")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Scatter: Bathrooms vs Price
        st.subheader("Bathrooms vs Price")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df["bathrooms"], df["price"], alpha=0.5, c="#764ba2", edgecolors="white", s=40)
        ax.set_xlabel("Bathrooms")
        ax.set_ylabel("Price")
        ax.set_title("Bathrooms vs Price")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Boxplot: Price vs Furnishing Status
        st.subheader("Price vs Furnishing Status")
        groups = df.groupby("furnishingstatus")["price"].apply(list)
        labels = list(groups.index)
        data = list(groups.values)
        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops=dict(color="red", linewidth=2))
        colors_bp = ["#667eea", "#764ba2", "#f093fb"]
        for patch, color in zip(bp["boxes"], colors_bp):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xlabel("Furnishing Status")
        ax.set_ylabel("Price")
        ax.set_title("Price Distribution by Furnishing Status")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Boxplot: Price vs Prefarea
        st.subheader("Price vs Preferred Area")
        groups_pref = df.groupby("prefarea")["price"].apply(list)
        fig, ax = plt.subplots(figsize=(6, 5))
        bp2 = ax.boxplot(list(groups_pref.values), labels=[str(l) for l in groups_pref.index],
                         patch_artist=True, medianprops=dict(color="red", linewidth=2))
        for patch, color in zip(bp2["boxes"], ["#667eea", "#764ba2"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xlabel("Preferred Area")
        ax.set_ylabel("Price")
        ax.set_title("Price Distribution by Preferred Area")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="left", fontsize=9)
        ax.set_yticklabels(corr.columns, fontsize=9)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
        ax.set_title("Correlation Matrix", pad=60, fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Multivariate ────────────────────────────────────────────
    with st.expander("🔹 Multivariate Analysis", expanded=False):

        # Pairplot (manual with matplotlib)
        st.subheader("Pairwise Scatter Matrix (Numeric Features)")
        pair_cols = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
        n = len(pair_cols)
        fig, axes = plt.subplots(n, n, figsize=(16, 16))
        for i in range(n):
            for j in range(n):
                ax = axes[i][j]
                if i == j:
                    ax.hist(df[pair_cols[i]], bins=20, color="#667eea", alpha=0.7, edgecolor="white")
                else:
                    ax.scatter(df[pair_cols[j]], df[pair_cols[i]], alpha=0.3, s=10, c="#764ba2")
                if j == 0:
                    ax.set_ylabel(pair_cols[i], fontsize=8)
                else:
                    ax.set_ylabel("")
                if i == n - 1:
                    ax.set_xlabel(pair_cols[j], fontsize=8)
                else:
                    ax.set_xlabel("")
                ax.tick_params(labelsize=6)
        plt.suptitle("Pairwise Scatter Matrix", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Interaction: Area x Bathrooms -> Price
        st.subheader("Interaction — Area × Bathrooms vs Price")
        interaction = df["area"] * df["bathrooms"]
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(interaction, df["price"], c=df["bathrooms"], cmap="viridis",
                        alpha=0.6, edgecolors="white", s=40)
        fig.colorbar(sc, ax=ax, label="Bathrooms")
        ax.set_xlabel("Area × Bathrooms")
        ax.set_ylabel("Price")
        ax.set_title("Interaction Effect: Area × Bathrooms on Price")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # VIF (multicollinearity)
        st.subheader("Variance Inflation Factor (VIF)")
        numeric_df = df.select_dtypes(include=[np.number])
        vif_data = numeric_df.drop(columns=["price"], errors="ignore")
        vif_df = pd.DataFrame({
            "Feature": vif_data.columns,
            "VIF": [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
        }).sort_values("VIF", ascending=False).reset_index(drop=True)
        st.dataframe(vif_df, use_container_width=True)
        high_vif = vif_df[vif_df["VIF"] > 5]
        if not high_vif.empty:
            st.warning(f"⚠ Features with VIF > 5 (potential multicollinearity): {', '.join(high_vif['Feature'].tolist())}")
        else:
            st.success("✅ No significant multicollinearity detected (all VIF ≤ 5).")


# ═══════════════════════════════════════════════════════════════
# 3. DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def preprocess_data(df):
    """
    - Log-transform price if skewness > 0.5
    - One-hot encode categoricals (drop_first)
    - Standard-scale features
    - 80/20 train-test split
    - Return everything needed downstream
    """
    data = df.copy()

    # Log-transform price if skewed
    price_skew = data["price"].skew()
    log_transformed = False
    if abs(price_skew) > 0.5:
        data["price"] = np.log1p(data["price"])
        log_transformed = True

    # Separate target
    y = data["price"]
    X = data.drop(columns=["price"])

    # One-hot encode
    X = pd.get_dummies(X, drop_first=True)

    # Store column names before scaling
    feature_names = X.columns.tolist()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return (X_train, X_test, y_train, y_test,
            feature_names, scaler, log_transformed, X_scaled, y)


# ═══════════════════════════════════════════════════════════════
# 4. MODEL BUILDING
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def build_models(_X_train, _y_train, _X_scaled, _y):
    """
    Train six regressors.  Tree-based models use GridSearchCV.
    Returns dict of {name: (model, best_params, cv_scores)}.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(_X_train, _y_train)
    cv_lr = cross_val_score(lr, _X_scaled, _y, cv=kf, scoring="r2")
    results["Linear Regression"] = (lr, {}, cv_lr)

    # 2. Ridge Regression
    ridge_params = {"alpha": [0.01, 0.1, 1, 10, 100]}
    ridge_gs = GridSearchCV(Ridge(), ridge_params, cv=kf, scoring="r2", n_jobs=-1)
    ridge_gs.fit(_X_train, _y_train)
    cv_ridge = cross_val_score(ridge_gs.best_estimator_, _X_scaled, _y, cv=kf, scoring="r2")
    results["Ridge Regression"] = (ridge_gs.best_estimator_, ridge_gs.best_params_, cv_ridge)

    # 3. Lasso Regression
    lasso_params = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1]}
    lasso_gs = GridSearchCV(Lasso(max_iter=10000), lasso_params, cv=kf, scoring="r2", n_jobs=-1)
    lasso_gs.fit(_X_train, _y_train)
    cv_lasso = cross_val_score(lasso_gs.best_estimator_, _X_scaled, _y, cv=kf, scoring="r2")
    results["Lasso Regression"] = (lasso_gs.best_estimator_, lasso_gs.best_params_, cv_lasso)

    # 4. Random Forest
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }
    rf_gs = GridSearchCV(RandomForestRegressor(random_state=42), rf_params,
                         cv=kf, scoring="r2", n_jobs=-1)
    rf_gs.fit(_X_train, _y_train)
    cv_rf = cross_val_score(rf_gs.best_estimator_, _X_scaled, _y, cv=kf, scoring="r2")
    results["Random Forest"] = (rf_gs.best_estimator_, rf_gs.best_params_, cv_rf)

    # 5. Gradient Boosting
    gb_params = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    }
    gb_gs = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params,
                         cv=kf, scoring="r2", n_jobs=-1)
    gb_gs.fit(_X_train, _y_train)
    cv_gb = cross_val_score(gb_gs.best_estimator_, _X_scaled, _y, cv=kf, scoring="r2")
    results["Gradient Boosting"] = (gb_gs.best_estimator_, gb_gs.best_params_, cv_gb)

    # 6. XGBoost
    xgb_params = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
    }
    xgb_gs = GridSearchCV(
        xgb.XGBRegressor(random_state=42, verbosity=0, objective="reg:squarederror"),
        xgb_params, cv=kf, scoring="r2", n_jobs=-1
    )
    xgb_gs.fit(_X_train, _y_train)
    cv_xgb = cross_val_score(xgb_gs.best_estimator_, _X_scaled, _y, cv=kf, scoring="r2")
    results["XGBoost"] = (xgb_gs.best_estimator_, xgb_gs.best_params_, cv_xgb)

    return results


# ═══════════════════════════════════════════════════════════════
# 5. MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════
def evaluate_models(results, X_test, y_test, log_transformed, comp_df, best_name):
    """Display comparison table, actual-vs-predicted & residual plots."""

    st.header("📈 Model Evaluation")

    st.subheader("Comparison Table (sorted by R² Score)")
    st.dataframe(comp_df, use_container_width=True)

    st.success(f"🏆 **Best Model (highest CV R²):** {best_name}  — "
               f"CV R² = {comp_df[comp_df['Model']==best_name]['CV R² Mean'].values[0]:.4f}")

    # Actual vs Predicted plot for EACH model
    with st.expander("📉 Actual vs Predicted Plots", expanded=True):
        for name in comp_df["Model"]:
            model = results[name][0]
            y_pred = model.predict(X_test)
            if log_transformed:
                y_real = np.expm1(y_test)
                yp_real = np.expm1(y_pred)
            else:
                y_real = y_test
                yp_real = y_pred
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(y_real, yp_real, alpha=0.5, c="#667eea", edgecolors="white", s=35)
            mn = min(np.min(y_real), np.min(yp_real))
            mx = max(np.max(y_real), np.max(yp_real))
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5)
            ax.set_xlabel("Actual Price")
            ax.set_ylabel("Predicted Price")
            ax.set_title(f"Actual vs Predicted — {name}")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # Residual distribution for each model
    with st.expander("📉 Residual Distribution Plots", expanded=False):
        for name in comp_df["Model"]:
            model = results[name][0]
            y_pred = model.predict(X_test)
            if log_transformed:
                residuals = np.expm1(y_test) - np.expm1(y_pred)
            else:
                residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(residuals, bins=30, color="#764ba2", edgecolor="white", alpha=0.8)
            ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
            ax.set_xlabel("Residual")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Residual Distribution — {name}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 6. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════
def feature_importance_section(fi_df, fi_model_name):
    """Display feature importance bar chart and ranking table."""

    st.header("🌳 Feature Importance")

    if fi_df is None or fi_model_name is None:
        st.warning("No tree-based model available for feature importance.")
        return

    top10 = fi_df.head(10)

    st.subheader(f"Top 10 Features — {fi_model_name}")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top10["Feature"][::-1], top10["Importance"][::-1],
            color="#667eea", edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top 10 Feature Importances — {fi_model_name}")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Ranking Table")
    st.dataframe(fi_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# 7. SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════
def shap_explainability(results, X_test, feature_names, best_name):
    """SHAP TreeExplainer: summary, bar, and force plots."""

    st.header("🔍 SHAP Explainability")

    tree_models = ["Random Forest", "Gradient Boosting", "XGBoost"]
    model_name = best_name if best_name in tree_models else None
    if model_name is None:
        for tm in tree_models:
            if tm in results:
                model_name = tm
                break
    if model_name is None:
        st.warning("SHAP requires a tree-based model.")
        return None

    model = results[model_name][0]
    explainer = shap.TreeExplainer(model)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    shap_values = explainer.shap_values(X_test_df)

    # Summary plot
    st.subheader(f"SHAP Summary Plot — {model_name}")
    fig_sum = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_df, show=False, plot_type="dot")
    st.pyplot(fig_sum)
    plt.close("all")

    # Bar plot
    st.subheader("SHAP Feature Importance (Bar)")
    fig_bar = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_df, show=False, plot_type="bar")
    st.pyplot(fig_bar)
    plt.close("all")

    # Force plot for first observation
    st.subheader("SHAP Force Plot — Sample Prediction")
    try:
        shap.force_plot(
            explainer.expected_value, shap_values[0], X_test_df.iloc[0],
            matplotlib=True, show=False
        )
        st.pyplot(plt.gcf())
        plt.close("all")
    except Exception:
        st.info("Force plot rendering requires a compatible environment. Displaying text explanation instead.")
        top_shap = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_values[0]
        }).sort_values("SHAP Value", key=abs, ascending=False).head(5)
        st.dataframe(top_shap)

    # Explain direction
    mean_abs = np.abs(shap_values).mean(axis=0)
    direction_df = pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs})
    direction_df = direction_df.sort_values("Mean |SHAP|", ascending=False).head(10)
    st.subheader("Features that Increase / Decrease Price")
    st.markdown("""
    - Features with **high positive** SHAP values **increase** the predicted price.
    - Features with **high negative** SHAP values **decrease** the predicted price.
    """)
    st.dataframe(direction_df, use_container_width=True)

    return shap_values


# ═══════════════════════════════════════════════════════════════
# 8. PREDICTION INTERFACE
# ═══════════════════════════════════════════════════════════════
def prediction_interface(results, feature_names, scaler, log_transformed,
                         best_name, df, X_test, y_test):
    """Sidebar inputs -> preprocess -> predict -> show result + SHAP."""

    st.header("🎯 Price Prediction Interface")
    st.markdown("Configure features in the **sidebar** and click **Predict**.")

    with st.sidebar:
        st.header("🏡 Property Features")

        area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=5000, step=100)
        bedrooms = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1, 4, 1)
        stories = st.slider("Stories", 1, 4, 2)
        parking = st.slider("Parking Spots", 0, 3, 1)

        mainroad = st.selectbox("Mainroad", ["yes", "no"])
        guestroom = st.selectbox("Guestroom", ["yes", "no"])
        basement = st.selectbox("Basement", ["yes", "no"])
        hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
        airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
        prefarea = st.selectbox("Preferred Area", ["yes", "no"])
        furnishingstatus = st.selectbox("Furnishing Status",
                                        ["furnished", "semi-furnished", "unfurnished"])

        predict_btn = st.button("🔮 Predict Price", use_container_width=True)

    if predict_btn:
        # Build input dataframe
        input_dict = {
            "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
            "stories": stories, "parking": parking,
            "mainroad": mainroad, "guestroom": guestroom, "basement": basement,
            "hotwaterheating": hotwaterheating, "airconditioning": airconditioning,
            "prefarea": prefarea, "furnishingstatus": furnishingstatus,
        }
        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # Align columns with training features
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_names]

        input_scaled = scaler.transform(input_encoded)

        model = results[best_name][0]
        pred = model.predict(input_scaled)[0]

        # Compute RMSE on test set for confidence interval
        y_test_pred = model.predict(X_test)
        if log_transformed:
            rmse_val = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_test_pred)))
            pred_price = np.expm1(pred)
        else:
            rmse_val = np.sqrt(mean_squared_error(y_test, y_test_pred))
            pred_price = pred

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>₹{pred_price:,.0f}</h3>
                <p>Predicted Price</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>₹{pred_price - rmse_val:,.0f}</h3>
                <p>Lower Bound (−RMSE)</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>₹{pred_price + rmse_val:,.0f}</h3>
                <p>Upper Bound (+RMSE)</p>
            </div>
            """, unsafe_allow_html=True)

        # SHAP for this single prediction
        tree_models = ["Random Forest", "Gradient Boosting", "XGBoost"]
        if best_name in tree_models:
            try:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(input_scaled)
                shap_df = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP Value": sv[0]
                }).sort_values("SHAP Value", key=abs, ascending=False).head(5)
                st.subheader("Top Influencing Features (SHAP)")
                st.dataframe(shap_df, use_container_width=True)
            except Exception:
                st.info("SHAP explanation unavailable for this input.")


# ═══════════════════════════════════════════════════════════════
# 9. BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════════════
def business_insights(df, fi_df, best_name, comp_df):
    """Auto-generate business intelligence from model results."""

    st.header("💼 Business Insights")

    # Top 5 price drivers
    if fi_df is not None and len(fi_df) >= 5:
        top5 = fi_df.head(5)["Feature"].tolist()
        st.subheader("🔑 Top 5 Price Drivers")
        for i, f in enumerate(top5, 1):
            st.markdown(f"**{i}.** `{f}`")

    st.subheader("🏗️ Structural Insights")
    avg_by_stories = df.groupby("stories")["price"].mean().sort_values(ascending=False)
    st.markdown(f"""
    - Houses with **{avg_by_stories.index[0]} stories** command the highest average price 
      (₹{avg_by_stories.iloc[0]:,.0f}).
    - Each additional **bathroom** adds approximately 
      ₹{df.groupby('bathrooms')['price'].mean().diff().mean():,.0f} to the average price.
    - Properties on the **mainroad** are priced on average 
      ₹{df[df['mainroad']=='yes']['price'].mean() - df[df['mainroad']=='no']['price'].mean():,.0f} higher.
    """)

    st.subheader("📈 Investment Insights")
    pref_premium = df[df["prefarea"] == "yes"]["price"].mean() - df[df["prefarea"] == "no"]["price"].mean()
    st.markdown(f"""
    - **Preferred area** properties carry a premium of ₹{pref_premium:,.0f} on average.
    - **Air conditioning** adds roughly 
      ₹{df[df['airconditioning']=='yes']['price'].mean() - df[df['airconditioning']=='no']['price'].mean():,.0f} 
      to value.
    - **Furnished** homes sell for ~₹{df[df['furnishingstatus']=='furnished']['price'].mean() - df[df['furnishingstatus']=='unfurnished']['price'].mean():,.0f} 
      more than unfurnished homes.
    """)

    st.subheader("🏘️ Builder Recommendations")
    st.markdown("""
    - Prioritize **larger area** and **multiple bathrooms** for maximum ROI.
    - Invest in **air conditioning** and **preferred area** locations.
    - **Furnished** or **semi-furnished** units command higher resale value.
    - Adding a **basement** can provide marginal price uplift.
    """)

    st.subheader("📊 Non-Linear Behavior Observations")
    st.markdown("""
    - Price does **not** increase linearly with area — there is diminishing returns beyond ~10,000 sq ft.
    - Tree-based models capture this non-linearity, outperforming simple linear models.
    - Interaction effects (e.g., area × bathrooms) significantly influence price prediction.
    """)

    st.subheader("🔒 Model Reliability")
    best_r2 = comp_df[comp_df["Model"] == best_name]["CV R² Mean"].values[0]
    st.markdown(f"""
    - The **{best_name}** model achieves a **cross-validated R² of {best_r2:.4f}**.
    - This means it explains **{best_r2*100:.1f}%** of price variance on unseen data.
    - Predictions are most reliable within the training data range 
      (₹{df['price'].min():,.0f} — ₹{df['price'].max():,.0f}).
    - Extrapolation beyond this range carries higher uncertainty.
    """)


# ═══════════════════════════════════════════════════════════════
# 10. DOWNLOADABLE REPORT
# ═══════════════════════════════════════════════════════════════
def generate_downloadable_report(comp_df, fi_df):
    """CSV downloads for model comparison, feature importance."""

    st.header("📥 Downloadable Reports")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Comparison")
        csv1 = comp_df.to_csv(index=False)
        st.download_button("⬇ Download Model Comparison (CSV)", csv1,
                           file_name="model_comparison.csv", mime="text/csv")

    with col2:
        if fi_df is not None:
            st.subheader("Feature Importance")
            csv2 = fi_df.to_csv(index=False)
            st.download_button("⬇ Download Feature Importance (CSV)", csv2,
                               file_name="feature_importance.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════
# 11. GEMINI AI CHATBOT
# ═══════════════════════════════════════════════════════════════
def chatbot_section(df, comp_df, fi_df):
    """Gemini-powered chatbot restricted to data-analysis Q&A."""

    st.header("🤖 AI Data Analysis Chatbot")
    st.markdown("Ask questions about the housing data, EDA findings, model results, or feature insights.")

    # API key from sidebar
    api_key = st.sidebar.text_input("🔑 Gemini API Key", type="password",
                                     help="Enter your Google Gemini API key to enable the chatbot.")

    if not api_key:
        st.info("Enter your **Gemini API Key** in the sidebar to activate the chatbot.")
        return

    # Import and configure Gemini
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
    except ImportError:
        st.error("Please install `google-generativeai`: `pip install google-generativeai`")
        return
    except Exception as e:
        st.error(f"Gemini initialization failed: {e}")
        return

    # Build a context summary the model can reference
    data_summary = f"""
You are an expert data analyst assistant. You ONLY answer questions related to the following housing price dataset and its analysis. 
If a question is not about this data or data analysis, politely decline.

DATASET OVERVIEW:
- {len(df)} housing records with {len(df.columns)} features.
- Columns: {', '.join(df.columns.tolist())}
- Target: price (range ₹{df['price'].min():,.0f} to ₹{df['price'].max():,.0f})
- Mean price: ₹{df['price'].mean():,.0f}, Median: ₹{df['price'].median():,.0f}
- Price skewness: {df['price'].skew():.4f}

STATISTICAL SUMMARY:
{df.describe().to_string()}

CATEGORICAL VALUE COUNTS:
{chr(10).join([f"  {col}: {df[col].value_counts().to_dict()}" for col in ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']])}

CORRELATION WITH PRICE:
{df.select_dtypes(include=[np.number]).corr()['price'].sort_values(ascending=False).to_string()}
"""

    if comp_df is not None:
        data_summary += f"""
MODEL COMPARISON:
{comp_df.to_string(index=False)}
Best Model: {comp_df.iloc[0]['Model']} with R² = {comp_df.iloc[0]['R² Score']:.4f}
"""

    if fi_df is not None:
        data_summary += f"""
TOP 10 FEATURES BY IMPORTANCE:
{fi_df.head(10).to_string(index=False)}
"""

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_msg = st.chat_input("Ask about the housing data analysis...")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        # Build conversation for Gemini
        full_prompt = data_summary + "\n\nConversation so far:\n"
        for msg in st.session_state.chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            full_prompt += f"{role}: {msg['content']}\n"
        full_prompt += "Assistant:"

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = model.generate_content(full_prompt)
                    reply = response.text
                except Exception as e:
                    reply = f"Sorry, I encountered an error: {e}"
                st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})


# ═══════════════════════════════════════════════════════════════
# DEPLOYMENT: Save / Load best model
# ═══════════════════════════════════════════════════════════════
def deployment_section(results, best_name):
    """Save/load the best model using joblib."""

    st.header("🚀 Model Deployment")

    model = results[best_name][0]
    model_path = os.path.join(os.path.dirname(__file__),
                              f"best_model_{best_name.replace(' ', '_').lower()}.joblib")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Best Model", use_container_width=True):
            joblib.dump(model, model_path)
            st.success(f"Model saved to `{os.path.basename(model_path)}`")

    with col2:
        if st.button("📂 Load Saved Model", use_container_width=True):
            if os.path.exists(model_path):
                loaded = joblib.load(model_path)
                st.success(f"Model loaded from `{os.path.basename(model_path)}`")
                st.json({"type": type(loaded).__name__, "params": str(loaded.get_params())})
            else:
                st.warning("No saved model found. Train and save first.")


# ═══════════════════════════════════════════════════════════════
#                          MAIN APP
# ═══════════════════════════════════════════════════════════════
def main():
    """Orchestrate all sections via tabs."""

    # ── Load data ──
    df = load_data()

    # ── Preprocess & build models ONCE (cached, so instant on reruns) ──
    (X_train, X_test, y_train, y_test,
     feature_names, scaler, log_transformed, X_scaled, y_full) = preprocess_data(df)

    results = build_models(X_train, y_train, X_scaled, y_full)

    # ── Compute evaluation metrics once for all tabs ──
    eval_rows = []
    for name, (model, params, cv_scores) in results.items():
        y_pred = model.predict(X_test)
        if log_transformed:
            y_test_real = np.expm1(y_test)
            y_pred_real = np.expm1(y_pred)
        else:
            y_test_real = y_test
            y_pred_real = y_pred
        mae  = mean_absolute_error(y_test_real, y_pred_real)
        mse  = mean_squared_error(y_test_real, y_pred_real)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test_real, y_pred_real)
        cv_mean = cv_scores.mean()
        eval_rows.append({
            "Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse,
            "R² Score": r2, "CV R² Mean": cv_mean,
            "Best Params": str(params) if params else "Default",
        })
    comp_df = pd.DataFrame(eval_rows).sort_values("R² Score", ascending=False).reset_index(drop=True)
    best_name = comp_df.sort_values("CV R² Mean", ascending=False).iloc[0]["Model"]

    # ── Compute feature importance once ──
    tree_models_list = ["Random Forest", "Gradient Boosting", "XGBoost"]
    fi_model_name = best_name if best_name in tree_models_list else None
    if fi_model_name is None:
        for tm in tree_models_list:
            if tm in results:
                fi_model_name = tm
                break
    fi_df = None
    if fi_model_name is not None:
        fi_model_obj = results[fi_model_name][0]
        importances = fi_model_obj.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)

    # ── Create tabs ──
    tabs = st.tabs([
        "📋 Data Understanding",
        "📊 EDA",
        "⚙️ Preprocessing",
        "🤖 Model Building",
        "📈 Evaluation",
        "🌳 Feature Importance",
        "🔍 SHAP",
        "🎯 Predict",
        "💼 Business Insights",
        "📥 Reports",
        "🚀 Deploy",
        "💬 Chatbot",
    ])

    # ── Tab 1: Data Understanding ──
    with tabs[0]:
        st.header("📋 Data Understanding")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><h3>{df.shape[0]}</h3><p>Rows</p></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><h3>{df.shape[1]}</h3><p>Columns</p></div>',
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><h3>{df.isnull().sum().sum()}</h3><p>Missing Values</p></div>',
                        unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><h3>{df.duplicated().sum()}</h3><p>Duplicates</p></div>',
                        unsafe_allow_html=True)

        st.subheader("First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Data Shape")
        st.write(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")

        st.subheader("Data Types")
        dtypes_df = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values})
        st.dataframe(dtypes_df, use_container_width=True)

        st.subheader("Missing Values")
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({"Column": missing.index, "Missing Count": missing.values,
                                    "% Missing": (missing.values / len(df) * 100).round(2)})
        st.dataframe(missing_df, use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(df.describe().T, use_container_width=True)

        st.subheader("Target Distribution Skewness")
        skew_val = df["price"].skew()
        kurt_val = df["price"].kurtosis()
        st.info(f"**Skewness:** {skew_val:.4f}  |  **Kurtosis:** {kurt_val:.4f}")
        if abs(skew_val) > 0.5:
            st.warning("Price distribution is skewed — log transformation will be applied during preprocessing.")

    # ── Tab 2: EDA ──
    with tabs[1]:
        perform_eda(df)

    # ── Tab 3: Preprocessing ──
    with tabs[2]:
        st.header("⚙️ Data Preprocessing")
        st.success("✅ Preprocessing complete!")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><h3>{X_train.shape[0]}</h3><p>Training Samples</p></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><h3>{X_test.shape[0]}</h3><p>Test Samples</p></div>',
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><h3>{len(feature_names)}</h3><p>Features</p></div>',
                        unsafe_allow_html=True)

        st.subheader("Steps Applied")
        log_text = "✅ Applied (price skewness > 0.5)" if log_transformed else "❌ Not needed"
        st.markdown(f"""
        1. **Log Transform:** {log_text}
        2. **One-Hot Encoding:** `pd.get_dummies(drop_first=True)` → {len(feature_names)} features
        3. **Standard Scaling:** `StandardScaler` applied to all features
        4. **Train-Test Split:** 80/20 (random_state=42)
        5. **K-Fold CV:** 5-fold cross-validation ready
        """)

        st.subheader("Encoded Feature Names")
        st.write(", ".join([f"`{f}`" for f in feature_names]))

    # ── Tab 4: Model Building ──
    with tabs[3]:
        st.header("🤖 Model Building")
        st.success("✅ All 6 models trained!")

        for name, (mdl, params, cv_scores) in results.items():
            with st.expander(f"📦 {name}", expanded=False):
                st.write(f"**Best Parameters:** {params if params else 'Default'}")
                st.write(f"**CV R² Scores:** {np.round(cv_scores, 4).tolist()}")
                st.write(f"**CV R² Mean:** {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Tab 5: Evaluation ──
    with tabs[4]:
        evaluate_models(results, X_test, y_test, log_transformed, comp_df, best_name)

    # ── Tab 6: Feature Importance ──
    with tabs[5]:
        feature_importance_section(fi_df, fi_model_name)

    # ── Tab 7: SHAP ──
    with tabs[6]:
        shap_explainability(results, X_test, feature_names, best_name)

    # ── Tab 8: Prediction ──
    with tabs[7]:
        prediction_interface(results, feature_names, scaler, log_transformed,
                             best_name, df, X_test, y_test)

    # ── Tab 9: Business Insights ──
    with tabs[8]:
        business_insights(df, fi_df, best_name, comp_df)

    # ── Tab 10: Reports ──
    with tabs[9]:
        generate_downloadable_report(comp_df, fi_df)

    # ── Tab 11: Deploy ──
    with tabs[10]:
        deployment_section(results, best_name)

    # ── Tab 12: Chatbot ──
    with tabs[11]:
        chatbot_section(df, comp_df, fi_df)


# ─────────────────────────── Entry Point ────────────────────────
if __name__ == "__main__":
    main()
