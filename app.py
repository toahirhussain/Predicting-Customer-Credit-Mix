# app.py
# Streamlit app for Credit Mix prediction (Top-15 pipeline)

import os
import joblib
import pandas as pd
import streamlit as st

DISPLAY_NAMES = {
    "Interest_Rate": "Interest Rate (%)",
    "Outstanding_Debt": "Outstanding Debt",
    "Num_of_Delayed_Payment": "Number of Delayed Payments",
    "Num_Bank_Accounts": "Number of Bank Accounts",
    "Delay_from_due_date": "Delay from Due Date (days)",
    "Changed_Credit_Limit": "Change in Credit Limit",
    "Credit_History_Age": "Credit History Age (years)",
    "Num_of_Loan": "Number of Loans",
    "Num_Credit_Card": "Number of Credit Cards",
    "Age": "Age",
    "Total_EMI_per_month": "Total EMI per Month",
    "Num_Credit_Inquiries": "Number of Credit Inquiries",
    "Annual_Income": "Annual Income"
}

DISPLAY_NAMES.update({
    "Outstanding_Debt": "Outstanding Debt",
    "Changed_Credit_Limit": "Change in Credit Limit",
    "Total_EMI_per_month": "Total EMI per Month",
    "Annual_Income": "Annual Income",
    "Payment_of_Min_Amount": "Payment of Minimum Amount",
    "Occupation": "Occupation",
    "Interest_Rate": "Interest Rate (%)",  # optional: nicer display
})


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
/* App background */
.stApp {
  background: linear-gradient(180deg, rgba(16, 24, 40, 1) 0%, rgba(2, 6, 23, 1) 100%);
}

/* Main container spacing */
.block-container {
  padding-top: 2.2rem;
  padding-bottom: 2.2rem;
}

/* Headline */
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

/* Cards */
.card {
  padding: 18px;
  border-radius: 16px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}

/* Secondary text */
.muted {
  color: rgba(255,255,255,0.70);
  font-size: 13px;
}

/* Prediction pill */
.pill {
  display: inline-block;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.08);
  color: rgba(255,255,255,0.92);
  font-weight: 600;
}

/* Footer */
.footer {
  margin-top: 16px;
  color: rgba(255,255,255,0.55);
  font-size: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource
def load_artifacts():
    """
    Loads model + schema + label map. Expect files to be in the same folder as app.py.
    """
    pipe = joblib.load("credit_mix_pipeline_top15.joblib")
    schema = joblib.load("top15_schema.joblib")
    label_map = joblib.load("label_map.joblib")
    return pipe, schema, label_map


def build_default_values(num_cols, cat_cols):
    defaults = {}
    for c in num_cols:
        defaults[c] = 0.0
    for c in cat_cols:
        defaults[c] = ""
    return defaults


def cast_inputs(df: pd.DataFrame, num_cols, cat_cols) -> pd.DataFrame:
    """
    Make sure model receives the right dtypes.
    """
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


# ----------------------------
# Load artifacts (with friendly error)
# ----------------------------
try:
    model, schema, label_map = load_artifacts()
except Exception as e:
    st.error(
        "Could not load model artifacts. Make sure these files exist next to app.py:\n"
        "- credit_mix_pipeline_top15.joblib\n"
        "- top15_schema.joblib\n"
        "- label_map.joblib\n\n"
        f"Error: {e}"
    )
    st.stop()

top15 = schema.get("top15", [])
top_num_cols = schema.get("top_num_cols", [])
top_cat_cols = schema.get("top_cat_cols", [])


# Reverse map (optional; not required)
inv_label_map = {v: k for k, v in label_map.items()}

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
<div class="hero">
  <h1>💳 Credit Mix Predictor</h1>
  <p>Enter the key inputs below and get an instant prediction (Bad / Standard / Good) using the trained Top-15 pipeline.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")
left, right = st.columns([1.2, 0.8], gap="large")

# ----------------------------
# Left: Inputs
# ----------------------------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Inputs")

    st.markdown(
        '<div class="muted">Tip: Fill what you know. Unknown categories are handled safely, and missing numerics become blank.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    # Build form
    defaults = build_default_values(top_num_cols, top_cat_cols)

    with st.form("credit_mix_form", clear_on_submit=False):
        st.markdown("### Numeric fields")
        num_cols_1, num_cols_2 = st.columns(2, gap="medium")

        numeric_inputs = {}

        # Spread numeric inputs into two columns
        # Decide which numeric columns should be integers vs floats
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
        
        for i, col in enumerate(top_num_cols):
            target_col = num_cols_1 if i % 2 == 0 else num_cols_2
            with target_col:
                if col in INT_COLS:
                    label = DISPLAY_NAMES.get(col, col)
                    numeric_inputs[col] = st.number_input(
                        label=label,
                        min_value=0,
                        value=int(defaults[col]),
                        step=1,
                        format="%d",
                    )
                else:
                    numeric_inputs[col] = st.number_input(
                        label=col,
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
            with target_col:
                label = DISPLAY_NAMES.get(col, col)
                categorical_inputs[col] = st.text_input(
                    label=label,
                    value=str(defaults[col]),
                    placeholder="Type a value…",
                )

        st.write("")
        submitted = st.form_submit_button("🔮 Predict Credit Mix", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Right: Results + info
# ----------------------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Prediction")

    if not top15:
        st.warning("Schema is missing 'top15'. Re-save `top15_schema.joblib` from your notebook.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    if submitted:
        # Build a single-row dataframe in exact top15 order
        row = {}
        row.update(numeric_inputs)
        row.update(categorical_inputs)

        X_new = pd.DataFrame([row], columns=top15)
        X_new = cast_inputs(X_new, top_num_cols, top_cat_cols)

        # Basic missing check for numerics
        num_missing = X_new[top_num_cols].isna().sum().sum() if top_num_cols else 0
        if num_missing:
            st.warning(
                "Some numeric inputs could not be parsed and became blank (NaN). "
                "This may reduce prediction quality."
            )

        try:
            pred_enc = model.predict(X_new)[0]
            pred_label = label_map.get(int(pred_enc), str(pred_enc))

            st.markdown(
                f'<div class="pill">Result: {pretty_label(pred_label)}</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.write(friendly_takeaway(pred_label))

            # Optional: show probabilities if model supports it
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_new)[0]

                # If label_map uses 0/1/2 keys, order is by encoded class indices
                # We'll build a small dataframe for display
                prob_rows = []
                for idx, p in enumerate(proba):
                    prob_rows.append(
                        {
                            "Class": pretty_label(label_map.get(idx, idx)),
                            "Probability": float(p),
                        }
                    )
                prob_df = pd.DataFrame(prob_rows).sort_values("Probability", ascending=False)

                st.write("")
                st.caption("Confidence (model probabilities)")
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    else:
        st.markdown('<div class="muted">Fill the inputs and click <b>Predict</b>.</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown("---")
    st.caption("Model: Top-15 feature pipeline (saved via joblib).")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    """
<div class="footer">
  Built with Streamlit • This tool is for educational/demo use and should not be treated as financial advice.
</div>
""",
    unsafe_allow_html=True,
)
