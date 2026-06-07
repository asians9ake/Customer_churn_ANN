import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import plotly.graph_objects as go
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

[data-testid="stMetricValue"] {
    font-size: 28px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg,#2563eb,#7c3aed);
    color: white;
    border-radius: 10px;
    height: 3rem;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

# --------------------------------------------------
# LOAD ENCODERS
# --------------------------------------------------
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div style="
padding:25px;
border-radius:15px;
background: linear-gradient(90deg,#2563eb,#7c3aed);
text-align:center;
color:white;">
<h1>🏦 Customer Churn Prediction Dashboard</h1>
<p>AI Powered Banking Analytics & Customer Retention System</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("📋 Customer Information")

credit_score = st.sidebar.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

geography = st.sidebar.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.sidebar.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.sidebar.slider(
    "Age",
    18,
    92,
    35
)

tenure = st.sidebar.slider(
    "Tenure",
    0,
    10,
    5
)

balance = st.sidebar.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

num_of_products = st.sidebar.slider(
    "Number of Products",
    1,
    4,
    1
)

has_cr_card = st.sidebar.selectbox(
    "Has Credit Card",
    ["Yes", "No"]
)

is_active_member = st.sidebar.selectbox(
    "Active Member",
    ["Yes", "No"]
)

estimated_salary = st.sidebar.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=100000.0
)

# --------------------------------------------------
# CONVERT VALUES
# --------------------------------------------------
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------
st.subheader("📊 Customer Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Credit Score", credit_score)

with c2:
    st.metric("Age", age)

with c3:
    st.metric("Balance", f"${balance:,.0f}")

with c4:
    st.metric("Salary", f"${estimated_salary:,.0f}")

st.divider()

# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform(
    [[geography]]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(
        ['Geography']
    )
)

input_data = pd.concat(
    [input_data.reset_index(drop=True),
     geo_encoded_df.reset_index(drop=True)],
    axis=1
)

input_data_scaled = scaler.transform(input_data)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("🚀 Predict Churn"):

    prediction = model.predict(
        input_data_scaled,
        verbose=0
    )

    prediction_proba = float(prediction[0][0])

    # -----------------------------------------
    # CHARTS
    # -----------------------------------------
    col1, col2 = st.columns(2)

    with col1:

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba * 100,
            title={'text': "Churn Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 50], 'color': 'lightgreen'},
                    {'range': [50, 80], 'color': 'orange'},
                    {'range': [80, 100], 'color': 'red'}
                ]
            }
        ))

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

    with col2:

        pie = px.pie(
            values=[
                prediction_proba,
                1 - prediction_proba
            ],
            names=[
                "Churn",
                "Stay"
            ],
            title="Customer Outcome Probability"
        )

        st.plotly_chart(
            pie,
            use_container_width=True
        )

    st.divider()

    # -----------------------------------------
    # PREDICTION RESULT
    # -----------------------------------------
    st.subheader("🎯 Prediction Confidence")

    st.progress(prediction_proba)

    st.metric(
        "Churn Probability",
        f"{prediction_proba:.2%}"
    )

    if prediction_proba > 0.5:
        st.error(
            f"⚠️ Customer is likely to churn ({prediction_proba:.2%})"
        )
    else:
        st.success(
            f"✅ Customer is likely to stay ({1-prediction_proba:.2%})"
        )

    st.divider()

    # -----------------------------------------
    # AI INSIGHTS
    # -----------------------------------------
    st.subheader("🤖 AI Insights")

    insights = []

    if age > 50:
        insights.append("👴 Customer belongs to higher age group.")

    if balance > 100000:
        insights.append("💰 High account balance detected.")

    if is_active_member == 0:
        insights.append("📉 Customer is not an active member.")

    if num_of_products == 1:
        insights.append("🛒 Customer uses only one banking product.")

    if credit_score < 500:
        insights.append("⚠️ Low credit score detected.")

    if len(insights) == 0:
        st.success("No major churn indicators detected.")
    else:
        for item in insights:
            st.warning(item)

    st.divider()

    # -----------------------------------------
    # CUSTOMER PROFILE
    # -----------------------------------------
    st.subheader("📋 Customer Profile")

    profile = pd.DataFrame({
        "Feature": [
            "Credit Score",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "Products",
            "Credit Card",
            "Active Member",
            "Salary"
        ],
        "Value": [
            credit_score,
            geography,
            gender,
            age,
            tenure,
            balance,
            num_of_products,
            "Yes" if has_cr_card else "No",
            "Yes" if is_active_member else "No",
            estimated_salary
        ]
    })

    st.dataframe(
        profile,
        use_container_width=True
    )

    st.divider()

    st.caption(
        "Customer Churn Prediction Dashboard | Built with Streamlit, TensorFlow & Plotly"
    )