import streamlit as st
import joblib
import pandas as pd

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(page_title="Campaign Predictor", page_icon="\U0001F4CA", layout="centered")

st.title("\U0001F4CA Campaign Performance Predictor")

# ---------------------------------------------------------
# Load all models (cached so they only load once)
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    rf_model = joblib.load("RF_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    lr_model = joblib.load("LR_model.pkl")
    return rf_model, label_encoder, lr_model

try:
    RF, le, LR = load_models()
    models_loaded = True
except FileNotFoundError as e:
    models_loaded = False
    missing_file_error = str(e)

if not models_loaded:
    st.error(
        "One or more model files are missing. Make sure "
        "'RF_model.pkl', 'label_encoder.pkl', and 'LR_model.pkl' "
        "are all in the same folder as this app.py file."
    )
    st.caption(f"Details: {missing_file_error}")
    st.stop()

# ---------------------------------------------------------
# Tabs for the two prediction modes
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["\U0001F4B0 Profit / Loss Prediction", "\U0001F4C8 Revenue Prediction"])

# ===========================================================
# TAB 1: Profit / Loss Prediction (Random Forest)
# Features (exact order from RF_model.pkl):
# ['Impressions','Clicks','Leads','Conversions','Revenue','Acquisition_Cost',
#  'Email','Facebook','Google','Instagram','WhatsApp','YouTube']
# ===========================================================
with tab1:
    st.subheader("Predict Profit or Loss")
    st.caption("Model: Random Forest Classifier (100 trees)")

    col1, col2, col3 = st.columns(3)
    with col1:
        pl_impressions = st.number_input("Impressions", min_value=0, value=5000, step=100, key="pl_imp")
        pl_clicks = st.number_input("Clicks", min_value=0, value=300, step=10, key="pl_clicks")
    with col2:
        pl_leads = st.number_input("Leads", min_value=0, value=150, step=10, key="pl_leads")
        pl_conversions = st.number_input("Conversions", min_value=0, value=100, step=1, key="pl_conv")
    with col3:
        pl_revenue = st.number_input("Revenue (\u20b9)", min_value=0.0, value=50000.0, step=1000.0, key="pl_rev")
        pl_acquisition_cost = st.number_input(
            "Acquisition Cost per Conversion (\u20b9)", min_value=0.0, value=200.0, step=10.0, key="pl_cost"
        )

    st.markdown("**Channels Used**")
    channel_names = ["Email", "Facebook", "Google", "Instagram", "WhatsApp", "YouTube"]
    channel_cols = st.columns(3)
    channels_selected = {}
    for i, channel in enumerate(channel_names):
        with channel_cols[i % 3]:
            channels_selected[channel] = st.checkbox(channel, value=False, key=f"chk_{channel}")

    if st.button("Predict Profit / Loss", type="primary", use_container_width=True, key="btn_pl"):
        input_data = pd.DataFrame([{
            "Impressions": pl_impressions,
            "Clicks": pl_clicks,
            "Leads": pl_leads,
            "Conversions": pl_conversions,
            "Revenue": pl_revenue,
            "Acquisition_Cost": pl_acquisition_cost,
            "Email": int(channels_selected["Email"]),
            "Facebook": int(channels_selected["Facebook"]),
            "Google": int(channels_selected["Google"]),
            "Instagram": int(channels_selected["Instagram"]),
            "WhatsApp": int(channels_selected["WhatsApp"]),
            "YouTube": int(channels_selected["YouTube"]),
        }])

        prediction = RF.predict(input_data)[0]
        prediction_label = le.inverse_transform([prediction])[0]

        proba = RF.predict_proba(input_data)[0]
        confidence = max(proba) * 100

        st.divider()
        if prediction_label.lower() == "profit":
            st.success(f"### \u2705 Predicted Result: {prediction_label}")
        else:
            st.error(f"### \u26a0\ufe0f Predicted Result: {prediction_label}")

        st.metric("Model Confidence", f"{confidence:.1f}%")
        st.caption(
            "Random Forest confidence is averaged across 100 trees, so it's "
            "generally more realistic than a single Decision Tree's score."
        )

        with st.expander("See input summary"):
            st.dataframe(input_data)

# ===========================================================
# TAB 2: Revenue Prediction (Linear Regression)
# Features (exact order from LR_model.pkl):
# ['Impressions','Clicks','Leads','Conversions','Acquisition_Cost','ROI','Profit_Loss']
# ===========================================================
with tab2:
    st.subheader("Predict Revenue")
    st.caption("Model: Linear Regression")

    col1, col2 = st.columns(2)
    with col1:
        rv_impressions = st.number_input("Impressions", min_value=0, value=5000, step=100, key="rv_imp")
        rv_clicks = st.number_input("Clicks", min_value=0, value=300, step=10, key="rv_clicks")
        rv_leads = st.number_input("Leads", min_value=0, value=150, step=10, key="rv_leads")
        rv_conversions = st.number_input("Conversions", min_value=0, value=100, step=1, key="rv_conv")
    with col2:
        rv_acquisition_cost = st.number_input(
            "Acquisition Cost per Conversion (\u20b9)", min_value=0.0, value=200.0, step=10.0, key="rv_cost"
        )
        rv_roi = st.number_input("ROI", value=1.5, step=0.1, format="%.2f", key="rv_roi")
        rv_profit_loss = st.number_input(
            "Profit / Loss Amount (\u20b9)", value=10000.0, step=500.0,
            help="Positive = profit, negative = loss", key="rv_pl"
        )

    if st.button("Predict Revenue", type="primary", use_container_width=True, key="btn_rev"):
        input_data = pd.DataFrame([{
            "Impressions": rv_impressions,
            "Clicks": rv_clicks,
            "Leads": rv_leads,
            "Conversions": rv_conversions,
            "Acquisition_Cost": rv_acquisition_cost,
            "ROI": rv_roi,
            "Profit_Loss": rv_profit_loss,
        }])

        predicted_revenue = LR.predict(input_data)[0]

        st.divider()
        st.success(f"### \U0001F4B5 Predicted Revenue: \u20b9{predicted_revenue:,.2f}")

        with st.expander("See input summary"):
            st.dataframe(input_data)

st.divider()
st.caption(
    "Profit/Loss model: Random Forest Classifier (100 trees) | "
    "Revenue model: Linear Regression"
)