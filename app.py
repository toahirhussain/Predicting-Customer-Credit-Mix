# app.py
# Streamlit app for Credit Mix prediction (Top-15 pipeline) + Rules + SHAP

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Optional: SHAP (only used in Explain tab)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ----------------------------
# Page + styling
# ----------------------------
st.set_page_config(
    page_title="Credit Mix Predictor",
    page_icon="💳",
    layout="wide",
)

st.markdown(
    """
<style>
.stApp {
  background: linear-gradient(180deg, rgba(16, 24, 40, 1) 0%, rgba(2, 6, 23, 1) 100%);
}
.block-container {
  padding-top: 2.2rem;
  padding-bottom: 2.2rem;
}
.hero {
  padding: 22px 22px 14px 22px;
  border-radius: 16px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.hero h1 {
  margin: 0 0 6px 0;
  font-size: 34px;
  line-height: 1.15;
  color: rgba(255,255,255,0.96);
}
.hero p {
  margin: 0;
  color: rgba(255,255,255,0.72);
  font-size: 15px;
}
.card {
  padding: 18px;
  border-radius: 16px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}
.muted {
  color: rgba(255,255,255,0.70);
  font-size: 13px;
}
.pill {
  display: inline-block;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.08);
  color: rgba(255,255,255,0.92);
  font-weight: 600;
}
.footer {
  margin-top: 16px;
  color: rgba(255,255,255,0.55);
  font-size: 12px;
}
hr {
  border: none;
  border-top: 1px solid rgba(255,255,255,0.10);
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Configuration
# ----------------------------
ART_PIPELINE = "credit_mix_pipeline_top15.joblib"
ART_SCHEMA = "top15_schema.joblib"
ART_LABEL_MAP = "label_map.joblib"

# SHAP helper artifacts (recommended)
ART_SHAP_BG = "shap_background_top15.joblib"           # dataframe in ORIGINAL top15 feature space
ART_FEATURE_NAMES = "model_feature_names_top15.joblib" # list of transformed feature names (after preprocess)


# ----------------------------
# Display names (UI labels)
# Keep keys EXACTLY the same as your dataset columns
# ----------------------------
DISPLAY_NAMES = {
    "Interest_Rate": "Interest Rate (%)",
    "Outstanding_Debt": "Outstanding Debt",
    "Num_of_Delayed_Payment": "Number of Delayed Payments",
    "Delay_from_due_date": "Delay from Due Date (days)",
    "Credit_History_Age": "Credit History Age (years)",
    "Num_of_Loan": "Number of Loans",
    "Num_Credit_Card": "Number of Credit Cards",
    "Num_Bank_Accounts": "Number of Bank Accounts",
    "Changed_Credit_Limit": "Change in Credit Limit",
    "Total_EMI_per_month": "Total EMI per Month",
    "Annual_Income": "Annual Income",
    "Age": "Age",
    "Num_Credit_Inquiries": "Number of Credit Inquiries",
    "Payment_of_Min_Amount": "Minimum Amount Paid (Yes/No)",
    "Occupation": "Occupation",
}

# Which numeric fields should be integers (no decimals in UI)
INT_COLS = {
    "Num_of_Delayed_Payment",
    "Num_Bank_Accounts",
    "Delay_from_due_date",
    "Credit_History_Age",
    "Num_of_Loan",
    "Num_Credit_Card",
    "Age",
    "Num_Credit_Inquiries",
}

# Which categorical fields should be dropdowns (not free-text)
CATEGORICAL_DROPDOWNS = {
    "Payment_of_Min_Amount": ["Yes", "No"],
}

# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource
def load_artifacts():
    """
    Loads model + schema + label map. Expects files next to app.py.
    """
    pipe = joblib.load(ART_PIPELINE)
    schema = joblib.load(ART_SCHEMA)
    label_map = joblib.load(ART_LABEL_MAP)
    return pipe, schema, label_map


@st.cache_resource
def load_shap_artifacts():
    """
    Optional artifacts for SHAP explainability.
    If missing, we'll gracefully disable SHAP page.
    """
    bg = None
    feat_names = None

    if os.path.exists(ART_SHAP_BG):
        bg = joblib.load(ART_SHAP_BG)

    if os.path.exists(ART_FEATURE_NAMES):
        feat_names = joblib.load(ART_FEATURE_NAMES)

    return bg, feat_names


def build_default_values(num_cols, cat_cols):
    defaults = {}
    for c in num_cols:
        defaults[c] = 0
    for c in cat_cols:
        defaults[c] = ""
    return defaults


def cast_inputs(df: pd.DataFrame, num_cols, cat_cols) -> pd.DataFrame:
    out = df.copy()
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in cat_cols:
        out[c] = out[c].astype(str)
    return out


def pretty_label(lbl: str) -> str:
    return str(lbl).strip().title()


def friendly_takeaway(lbl: str) -> str:
    lbl_l = str(lbl).lower()
    if "good" in lbl_l:
        return "Your profile aligns with a stronger credit mix. Keep payments consistent and avoid unnecessary new credit."
    if "standard" in lbl_l:
        return "You’re in the middle range. Improving payment consistency and reducing utilization usually helps."
    if "bad" in lbl_l:
        return "This indicates risk signals. Focus on on-time payments, lowering outstanding balances, and limiting new credit."
    return "Prediction generated."


def apply_business_rules(X_row: pd.DataFrame):
    """
    Returns (decision_label_or_None, reason_or_None, action)
    action in {"override", "manual_review", "none"}
    """
    x = X_row.iloc[0]

    # Convert safely
    def num(v):
        try:
            return float(v)
        except Exception:
            return np.nan

    outstanding_debt = num(x.get("Outstanding_Debt", np.nan))
    delayed = num(x.get("Num_of_Delayed_Payment", np.nan))
    min_paid = str(x.get("Payment_of_Min_Amount", "")).strip().lower()

    # Example enterprise-style rules (adjust thresholds to your logic)
    # Rule 1: Very high debt + high delays => override to Bad
    if np.isfinite(outstanding_debt) and np.isfinite(delayed):
        if outstanding_debt >= 1_000_000 and delayed >= 8:
            return ("Bad", "Rule triggered: extremely high outstanding debt + many delayed payments.", "override")

    # Rule 2: If minimum payment is unknown/empty => manual review
    if min_paid not in {"yes", "no"}:
        return (None, "Rule triggered: Minimum payment field is missing or invalid. Send to manual review.", "manual_review")

    # Rule 3: If minimum payment is "no" AND delayed payments high => manual review
    if min_paid == "no" and np.isfinite(delayed) and delayed >= 12:
        return (None, "Rule triggered: Minimum payment not paid + many delays. Manual review required.", "manual_review")

    return (None, None, "none")


def safe_label_from_model_output(pred_enc, label_map):
    try:
        return label_map.get(int(pred_enc), str(pred_enc))
    except Exception:
        return str(pred_enc)


def build_probability_table(model, X_new, label_map):
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(X_new)[0]

    rows = []
    for idx, p in enumerate(proba):
        rows.append({"Class": pretty_label(label_map.get(idx, idx)), "Probability": float(p)})
    return pd.DataFrame(rows).sort_values("Probability", ascending=False)


# ----------------------------
# Load artifacts
# ----------------------------
try:
    model, schema, label_map = load_artifacts()
except Exception as e:
    st.error(
        "Could not load model artifacts. Make sure these files exist next to app.py:\n"
        f"- {ART_PIPELINE}\n"
        f"- {ART_SCHEMA}\n"
        f"- {ART_LABEL_MAP}\n\n"
        f"Error: {e}"
    )
    st.stop()

top15 = schema.get("top15", [])
top_num_cols = schema.get("top_num_cols", [])
top_cat_cols = schema.get("top_cat_cols", [])

# Optional SHAP artifacts
shap_bg, model_feature_names = load_shap_artifacts()

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
<div class="hero">
  <h1>💳 Credit Mix Predictor</h1>
  <p>Instant prediction (Bad / Standard / Good) using a trained Top-15 pipeline, with business rules + explainability.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

tabs = st.tabs(["🔮 Predict", "📊 Explain (SHAP)", "📄 About / How to Use"])

# =========================================================
# TAB 1: Predict
# =========================================================
with tabs[0]:
    left, right = st.columns([1.2, 0.8], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 Inputs")
        st.markdown(
            '<div class="muted">Tip: Fill what you know. Unknown categories are handled safely. If required fields are missing, we may flag manual review.</div>',
            unsafe_allow_html=True,
        )
        st.write("")

        if not top15:
            st.error("Schema missing 'top15'. Re-save `top15_schema.joblib` from your notebook.")
            st.stop()

        defaults = build_default_values(top_num_cols, top_cat_cols)

        with st.form("credit_mix_form", clear_on_submit=False):
            st.markdown("### Numeric fields")
            num_cols_1, num_cols_2 = st.columns(2, gap="medium")
            numeric_inputs = {}

            for i, col in enumerate(top_num_cols):
                target_col = num_cols_1 if i % 2 == 0 else num_cols_2
                ui_label = DISPLAY_NAMES.get(col, col)

                with target_col:
                    if col in INT_COLS:
                        numeric_inputs[col] = st.number_input(
                            label=ui_label,
                            min_value=0,
                            value=int(defaults[col]),
                            step=1,
                            format="%d",
                        )
                    else:
                        numeric_inputs[col] = st.number_input(
                            label=ui_label,
                            min_value=0.0,
                            value=float(defaults[col]),
                            step=0.1,
                            format="%.2f",
                        )

            st.write("")
            st.markdown("### Categorical fields")
            cat_cols_1, cat_cols_2 = st.columns(2, gap="medium")
            categorical_inputs = {}

            for i, col in enumerate(top_cat_cols):
                target_col = cat_cols_1 if i % 2 == 0 else cat_cols_2
                ui_label = DISPLAY_NAMES.get(col, col)

                with target_col:
                    if col in CATEGORICAL_DROPDOWNS:
                        options = CATEGORICAL_DROPDOWNS[col]
                        categorical_inputs[col] = st.selectbox(
                            label=ui_label,
                            options=options,
                            index=0,
                        )
                    else:
                        categorical_inputs[col] = st.text_input(
                            label=ui_label,
                            value=str(defaults[col]),
                            placeholder="Type a value…",
                        )

            st.write("")
            submitted = st.form_submit_button("🔮 Predict Credit Mix", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Prediction")

        if submitted:
            row = {}
            row.update(numeric_inputs)
            row.update(categorical_inputs)

            # Single-row DF in exact top15 order
            X_new = pd.DataFrame([row], columns=top15)
            X_new = cast_inputs(X_new, top_num_cols, top_cat_cols)

            # Business rules first
            rule_label, rule_reason, rule_action = apply_business_rules(X_new)

            try:
                pred_enc = model.predict(X_new)[0]
                pred_label = safe_label_from_model_output(pred_enc, label_map)

                final_label = pred_label
                banner = None

                if rule_action == "override":
                    final_label = rule_label
                    banner = f"✅ **Business rule override applied**: {rule_reason}"
                elif rule_action == "manual_review":
                    banner = f"⚠️ **Manual review recommended**: {rule_reason}"

                st.markdown(f'<div class="pill">Result: {pretty_label(final_label)}</div>', unsafe_allow_html=True)
                st.write("")

                if banner:
                    st.info(banner)

                st.write(friendly_takeaway(final_label))

                # Probabilities
                prob_df = build_probability_table(model, X_new, label_map)
                if prob_df is not None:
                    st.write("")
                    st.caption("Confidence (model probabilities)")
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)

                # Save last prediction + row for SHAP tab
                st.session_state["last_X_new"] = X_new
                st.session_state["last_pred_label"] = final_label

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        else:
            st.markdown('<div class="muted">Fill the inputs and click <b>Predict</b>.</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown("---")
        st.caption("Model: Top-15 feature pipeline (joblib). Rules layer included.")
        st.caption("Developed by Md Toahir Hussain")
        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 2: Explain (SHAP)
# =========================================================
with tabs[1]:
    st.info("""
    ### 🧠 How to read this explanation
    
    This section explains *why* the model made this prediction.
    
    Each feature gets a **SHAP contribution score**:
    
    • Positive value → pushes the decision **toward** this result  
    • Negative value → pushes the decision **away** from this result  
    • Larger absolute value → stronger influence  
    
    This is a **local explanation** for this specific customer, not a global feature importance ranking.
    """)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Explain prediction (SHAP)")

    if not SHAP_AVAILABLE:
        st.warning("SHAP is not installed in this environment. Add `shap` to requirements.txt and redeploy.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    if "last_X_new" not in st.session_state:
        st.info("Run a prediction first (Predict tab). Then come back here to see the explanation.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    if shap_bg is None or model_feature_names is None:
        st.warning(
            "SHAP artifacts not found.\n\n"
            "Please export these from the notebook and place them next to app.py:\n"
            f"- {ART_SHAP_BG}\n"
            f"- {ART_FEATURE_NAMES}\n"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    X_new = st.session_state["last_X_new"]

    try:
        pre = model.named_steps["preprocess"]
        est = model.named_steps["model"]

        # Transform to model feature space
        X_new_trans = pre.transform(X_new)

        # Background also must be in original space -> transform
        X_bg_trans = pre.transform(shap_bg)

        explainer = shap.TreeExplainer(est, X_bg_trans)

        shap_values = explainer.shap_values(X_new_trans)

        # Predicted encoded class (0/1/2)
        pred_enc = int(model.predict(X_new)[0])

        # Map predicted class to SHAP class index safely
        if hasattr(est, "classes_") and est.classes_ is not None:
            classes = list(est.classes_)
            class_idx = classes.index(pred_enc) if pred_enc in classes else 0
        else:
            class_idx = pred_enc  # fallback

        st.caption("Top contributing features for the predicted class")

        # Extract SHAP vector for the single row
        if isinstance(shap_values, list):
            sv_1d = shap_values[class_idx][0]      # (n_features,)
        else:
            sv_1d = shap_values[0, :, class_idx]   # (n_features,)

        contrib = pd.DataFrame({
            "Feature": model_feature_names,
            "SHAP_Contribution": sv_1d
        })
        contrib["Abs"] = contrib["SHAP_Contribution"].abs()
        contrib = (
            contrib.sort_values("Abs", ascending=False)
                  .drop(columns=["Abs"])
                  .head(15)
        )

        st.dataframe(contrib, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")

    def _pretty_feature_name(raw_name: str) -> str:
        # If you already have "cat__" / "num__" prefixes, clean them
        name = str(raw_name)
        name = name.replace("num__", "").replace("cat__", "")
        return DISPLAY_NAMES.get(name, name.replace("_", " "))

    def build_shap_summary(contrib_df: pd.DataFrame, pred_label: str, top_k_pos=3, top_k_neg=2) -> str:
        """
        contrib_df columns: Feature, SHAP_Contribution
        Positive => pushes toward predicted class; Negative => pushes away
        """
        df = contrib_df.copy()
        df["Feature"] = df["Feature"].apply(_pretty_feature_name)
    
        pos = df[df["SHAP_Contribution"] > 0].sort_values("SHAP_Contribution", ascending=False).head(top_k_pos)
        neg = df[df["SHAP_Contribution"] < 0].sort_values("SHAP_Contribution", ascending=True).head(top_k_neg)
    
        pos_feats = pos["Feature"].tolist()
        neg_feats = neg["Feature"].tolist()
    
        pred_label_clean = str(pred_label).strip().title()
        if pred_label_clean not in {"Good", "Standard", "Bad"}:
            pred_label_clean = "this result"
    
        # Short “what to do” suggestion per class (optional)
        advice = {
            "Good": "Keep payments consistent and avoid unnecessary new credit.",
            "Standard": "Pay on time, reduce utilization, and limit new credit inquiries.",
            "Bad": "Focus on on-time payments, lowering outstanding balances, and avoiding new credit until stable.",
        }.get(pred_label_clean, "")
    
    def bullets(items):
        return "\n".join([f"- {x}" for x in items]) if items else "- (No strong drivers found)"
    
        text = f"""### 📝 Plain English Summary
        
        The model predicted **{pred_label_clean}** mainly because:
        {bullets(pos_feats)}
        
        However, the prediction was weakened by:
        {bullets(neg_feats)}
        """
        if advice:
            text += f"\n**Suggested action:** {advice}\n"
    
        return text

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# TAB 3: About / How to Use
# =========================================================
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📄 About this app")

    st.write(
        "This app predicts a customer's **Credit Mix category** (Bad / Standard / Good) "
        "using a trained XGBoost model wrapped in a scikit-learn pipeline."
    )

    st.write("### How this would be used in real life")
    st.write(
        "- **Pre-screening**: An analyst enters known customer attributes and gets a quick risk signal.\n"
        "- **Scenario testing**: Adjust inputs (debt, delays, inquiries) to see how risk changes.\n"
        "- **Decision support**: Combines model output with a small set of **business rules** and flags cases for manual review."
    )

    st.write("### Important note")
    st.warning(
        "This is a demo/portfolio app by Md Toahir Hussain. In production, you would add monitoring, drift checks, "
        "logging, access control, and stronger validation for input ranges."
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    """
<div class="footer">
  Built with Streamlit • Educational/demo use only (not financial advice).
</div>
""",
    unsafe_allow_html=True,
)
