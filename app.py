import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="InsureAI | Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "insurance_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    features = joblib.load(BASE_DIR / "features.pkl")
    return model, scaler, list(features)

try:
    model, scaler, features = load_artifacts()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #263247;
}
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #172338, #111b2d);
    border: 1px solid #293950;
    padding: 20px;
    border-radius: 14px;
}
[data-testid="stMetricLabel"] {
    color: #9caec5;
}
[data-testid="stMetricValue"] {
    color: #60a5fa;
}
div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border: 0;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 600;
    width: 100%;
}
div.stButton > button:hover {
    color: white;
    border: 1px solid #60a5fa;
}
h1, h2, h3 {
    color: #f1f5f9;
}
.card {
    background: #111c2e;
    border: 1px solid #293950;
    padding: 22px;
    border-radius: 14px;
    margin-bottom: 12px;
}
.small-muted {
    color: #94a3b8;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🏥 InsureAI")
    st.caption("MACHINE LEARNING ANALYTICS")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Premium Predictor", "Model Insights"],
        label_visibility="visible"
    )

    st.divider()
    st.markdown("### Model Status")
    st.success("Model files loaded")
    st.caption(f"Features detected: {len(features)}")
    st.caption("Predictions are model estimates.")

# ---------------- HELPERS ----------------
def make_input():
    values = {}

    for f in features:
        name = f.replace("_", " ").title()

        if f == "sex":
            choice = st.selectbox(
                "Sex",
                ["Female", "Male"],
                key=f
            )
            values[f] = 1 if choice == "Male" else 0

        elif f == "smoker":
            choice = st.selectbox(
                "Smoker",
                ["No", "Yes"],
                key=f
            )
            values[f] = 1 if choice == "Yes" else 0

        elif f == "region":
            values[f] = st.selectbox(
                name,
                ["northeast", "northwest", "southeast", "southwest"],
                key=f
            )

        elif f == "age":
            values[f] = st.number_input(
                name, 18, 100, 25, key=f
            )

        elif f == "bmi":
            values[f] = st.number_input(
                name, 10.0, 80.0, 25.0,
                step=0.1, key=f
            )

        elif f == "children":
            values[f] = st.number_input(
                name, 0, 20, 0, key=f
            )

        else:
            values[f] = st.number_input(
                name, value=0.0, key=f
            )

    return values


def prepare_input(values):
    df = pd.DataFrame([values])[features]

    # Preserve the feature names expected by sklearn
    # and apply the saved scaler.
    if hasattr(scaler, "feature_names_in_"):
        df = df[scaler.feature_names_in_]

    X = scaler.transform(df)

    # Models trained on scaled data generally accept arrays.
    # Models trained on named DataFrames may need column names.
    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
        if len(model_features) == df.shape[1]:
            X = pd.DataFrame(X, columns=model_features)

    return X


def predict_premium(values):
    X = prepare_input(values)
    return float(np.asarray(model.predict(X)).reshape(-1)[0])


def plot_layout(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="#111c2e",
        plot_bgcolor="#111c2e",
        font=dict(color="#dbeafe"),
        margin=dict(l=25, r=20, t=55, b=25),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1")
        ),
        xaxis=dict(
            gridcolor="#263449",
            zerolinecolor="#263449"
        ),
        yaxis=dict(
            gridcolor="#263449",
            zerolinecolor="#263449"
        )
    )
    return fig


# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.markdown("# Insurance Analytics Dashboard")
    st.markdown(
        '<p class="small-muted">Machine learning powered '
        'health insurance premium estimation</p>',
        unsafe_allow_html=True
    )

    # Example values are illustrative, not actual dataset statistics.
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("ML Model", "Loaded", "Ready")
    c2.metric("Input Features", len(features))
    c3.metric("Prediction Engine", "Active")
    c4.metric("Visualization", "Plotly")

    st.divider()

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown("### 🎯 Quick Premium Estimator")
        st.caption("Enter details to estimate a premium.")

        with st.form("dashboard_prediction"):
            values = make_input()
            submitted = st.form_submit_button(
                "Calculate Estimated Premium"
            )

        if submitted:
            try:
                result = predict_premium(values)
                st.session_state["last_prediction"] = result
                st.session_state["last_inputs"] = values
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        if "last_prediction" in st.session_state:
            st.markdown(
                '<div class="card">'
                '<span class="small-muted">ESTIMATED PREMIUM</span>'
                f'<h1 style="color:#60a5fa;">'
                f'{st.session_state["last_prediction"]:,.2f}</h1>'
                '<span class="small-muted">'
                'Predicted by your trained ML model</span>'
                '</div>',
                unsafe_allow_html=True
            )

    with right:
        st.markdown("### 📊 Input Profile")
        st.caption("Profile visualization for the latest prediction.")

        if "last_inputs" in st.session_state:
            profile = st.session_state["last_inputs"]
        else:
            profile = {
                f: 0 for f in features
            }

        profile_df = pd.DataFrame({
            "Feature": list(profile.keys()),
            "Value": [str(v) for v in profile.values()]
        })

        fig = go.Figure(
            data=[go.Table(
                header=dict(
                    values=["Feature", "Input Value"],
                    fill_color="#1d4ed8",
                    font=dict(color="white", size=13),
                    align="left"
                ),
                cells=dict(
                    values=[
                        profile_df["Feature"],
                        profile_df["Value"]
                    ],
                    fill_color="#172338",
                    font=dict(color="#e2e8f0", size=12),
                    align="left",
                    height=30
                )
            )]
        )
        fig.update_layout(
            height=350,
            paper_bgcolor="#111c2e",
            margin=dict(l=5, r=5, t=5, b=5)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### 📈 Prediction Overview")

    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]

        fig = go.Figure(go.Indicator(
            mode="number+gauge",
            value=result,
            number={"valueformat": ",.2f"},
            title={"text": "Estimated Premium"},
            gauge={
                "shape": "angular",
                "axis": {"visible": True},
                "bar": {"color": "#60a5fa"},
                "bgcolor": "#172338",
                "bordercolor": "#334155"
            }
        ))
        plot_layout(fig, 300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Submit a prediction to display the premium gauge. "
            "No sample prediction is fabricated."
        )

elif page == "Premium Predictor":
    st.title("🎯 Premium Predictor")
    st.write(
        "Enter the customer's information to generate a prediction."
    )

    with st.form("premium_form"):
        values = make_input()
        submitted = st.form_submit_button("Predict Premium")

    if submitted:
        try:
            result = predict_premium(values)
            st.session_state["last_prediction"] = result
            st.session_state["last_inputs"] = values

            st.success("Prediction completed")
            st.metric("Estimated Premium", f"{result:,.2f}")

            st.caption(
                "This is a model estimate, not an official insurance quote."
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif page == "Model Insights":
    st.title("🧠 Model Insights")
    st.write("Inspect supported model properties and feature importance.")

    # Feature importance for supported estimators only.
    importance = None
    importance_type = None

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
        importance_type = "Tree-based feature importance"

    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 1:
            importance = np.abs(coef)
        else:
            importance = np.mean(np.abs(coef), axis=0)
        importance_type = "Absolute model coefficients"

    if importance is not None and len(importance) == len(features):
        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values("Importance", ascending=True)

        st.markdown(f"### {importance_type}")

        fig = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            title="Feature Importance"
        )
        fig.update_layout(coloraxis_showscale=False)
        plot_layout(fig, max(350, 40 * len(features)))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Feature importance and coefficient magnitude are "
            "different measures. Neither alone proves causation."
        )
    else:
        st.info(
            "This model does not expose compatible built-in "
            "feature importance for the saved feature list. "
            "No importance values have been invented."
        )

    st.divider()
    st.markdown("### Model Information")
    st.write("Model class:", type(model).__name__)
    st.write("Scaler class:", type(scaler).__name__)
    st.write("Number of features:", len(features))

    if hasattr(model, "n_features_in_"):
        st.write("Expected model inputs:", model.n_features_in_)

    if hasattr(scaler, "n_features_in_"):
        st.write("Scaler input count:", scaler.n_features_in_)

st.divider()
st.caption(
    "InsureAI • ML-powered estimate • Not an official insurance quote"
)