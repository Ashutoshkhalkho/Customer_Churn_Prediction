import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Customer Churn Prediction")
st.markdown("Predict whether a customer will churn using Machine Learning (Random Forest)")
st.markdown("---")

# ── Sidebar: File Upload ─────────────────────────────────────────────────────
st.sidebar.header("📂 Upload Dataset")
st.sidebar.markdown("Upload your `churn_data_csv.csv` file to get started.")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# ── Stop app if no file uploaded ─────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Please upload your **churn_data_csv.csv** file from the sidebar to get started!")
    st.stop()

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_raw_data(file):
    return pd.read_csv(file)

df_raw = load_raw_data(uploaded_file)

# ── Preprocessing ─────────────────────────────────────────────────────────────
CAT_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

def preprocess(df):
    df = df.copy()
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    le = LabelEncoder()
    for col in CAT_COLS:
        df[col] = le.fit_transform(df[col])
    return df

def train_model(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])
    cm = confusion_matrix(y_test, y_pred)
    return model, acc, report, cm, X.columns.tolist()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview",
    "📈 Exploratory Analysis",
    "🤖 Train Model",
    "🔮 Predict Customer"
])

# TAB 1 — Data Overview
with tab1:
    st.subheader("📋 Raw Dataset Preview")
    st.write(f"**Rows:** {df_raw.shape[0]}  |  **Columns:** {df_raw.shape[1]}")
    st.dataframe(df_raw.head(20), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        churn_rate = (df_raw['Churn'].value_counts(normalize=True)['Yes'] * 100)
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with col2:
        st.metric("Total Customers", df_raw.shape[0])
    with col3:
        st.metric("Avg Tenure (months)", f"{df_raw['tenure'].mean():.1f}")

    st.subheader("🔍 Missing Values")
    missing = df_raw.isnull().sum().reset_index()
    missing.columns = ['Column', 'Missing Count']
    st.dataframe(missing, use_container_width=True)

    st.subheader("📐 Statistical Summary")
    st.dataframe(df_raw.describe(), use_container_width=True)

# TAB 2 — EDA
with tab2:
    st.subheader("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        churn_counts = df_raw['Churn'].value_counts()
        fig1 = px.pie(values=churn_counts.values, names=churn_counts.index,
                      title="Churn Distribution",
                      color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.4)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.histogram(df_raw, x='tenure', color='Churn', barmode='overlay',
                            title="Tenure vs Churn",
                            color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.box(df_raw, x='Churn', y='MonthlyCharges', color='Churn',
                      title="Monthly Charges vs Churn",
                      color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        df_raw['TotalCharges_num'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
        fig4 = px.box(df_raw, x='Churn', y='TotalCharges_num', color='Churn',
                      title="Total Charges vs Churn",
                      color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig4, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        cc = df_raw.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
        fig5 = px.bar(cc, x='Contract', y='Count', color='Churn', barmode='group',
                      title="Contract Type vs Churn",
                      color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig5, use_container_width=True)
    with col6:
        pc = df_raw.groupby(['PaymentMethod', 'Churn']).size().reset_index(name='Count')
        fig6 = px.bar(pc, x='PaymentMethod', y='Count', color='Churn', barmode='group',
                      title="Payment Method vs Churn",
                      color_discrete_sequence=['#2ecc71', '#e74c3c'])
        fig6.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig6, use_container_width=True)

    col7, col8 = st.columns(2)
    with col7:
        fig7 = px.histogram(df_raw, x='InternetService', color='Churn', barmode='group',
                            title="Internet Service vs Churn",
                            color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig7, use_container_width=True)
    with col8:
        sc = df_raw.groupby(['SeniorCitizen', 'Churn']).size().reset_index(name='Count')
        sc['SeniorCitizen'] = sc['SeniorCitizen'].map({0: 'Non-Senior', 1: 'Senior'})
        fig8 = px.bar(sc, x='SeniorCitizen', y='Count', color='Churn', barmode='group',
                      title="Senior Citizen vs Churn",
                      color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig8, use_container_width=True)

# TAB 3 — Train Model
with tab3:
    st.subheader("🤖 Train Random Forest Model")
    st.info("Click the button below to train the ML model on your uploaded data.")

    if st.button("🚀 Train Model"):
        with st.spinner("Training model..."):
            df_processed = preprocess(df_raw)
            model, acc, report, cm, feature_names = train_model(df_processed)
            st.session_state['model'] = model
            st.session_state['feature_names'] = feature_names
            st.session_state['model_trained'] = True

        st.success(f"✅ Model Trained! Accuracy: **{acc * 100:.2f}%**")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📋 Classification Report")
            st.code(report, language='text')
        with col2:
            st.subheader("🔲 Confusion Matrix")
            fig_cm = ff.create_annotated_heatmap(
                z=cm,
                x=['Predicted: No Churn', 'Predicted: Churn'],
                y=['Actual: No Churn', 'Actual: Churn'],
                colorscale='Reds', showscale=True
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("📊 Top 10 Feature Importances")
        feat_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        fig_fi = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                        title="Top 10 Features", color='Importance',
                        color_continuous_scale='Blues')
        fig_fi.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_fi, use_container_width=True)

# TAB 4 — Predict
with tab4:
    st.subheader("🔮 Predict Churn for a Single Customer")

    if 'model_trained' not in st.session_state:
        st.warning("⚠️ Please go to **'Train Model'** tab and train the model first!")
        st.stop()

    model = st.session_state['model']
    feature_names = st.session_state['feature_names']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**👤 Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    with col2:
        st.markdown("**📞 Services**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col3:
        st.markdown("**💳 Account Info**")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))

    st.markdown("**📺 Add-on Services**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    with c2:
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    with c3:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    with c4:
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    if st.button("🎯 Predict Now"):
        input_dict = {
            'gender': gender, 'SeniorCitizen': senior_citizen,
            'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
            'PhoneService': phone_service, 'MultipleLines': multiple_lines,
            'InternetService': internet_service, 'OnlineSecurity': online_security,
            'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
            'TechSupport': tech_support, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies, 'Contract': contract,
            'PaperlessBilling': paperless_billing, 'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges, 'TotalCharges': str(total_charges)
        }

        input_df = pd.DataFrame([input_dict])
        df_full = df_raw.copy()
        df_full.drop('customerID', axis=1, inplace=True)
        df_full['TotalCharges'] = pd.to_numeric(df_full['TotalCharges'], errors='coerce')
        df_full.dropna(subset=['TotalCharges'], inplace=True)
        input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')

        le = LabelEncoder()
        for col in CAT_COLS:
            le.fit(df_full[col])
            input_df[col] = le.transform(input_df[col])

        input_df = input_df[feature_names]
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        if prediction == 1:
            st.error("⚠️ **This customer is LIKELY TO CHURN**")
            st.metric("Churn Probability", f"{prob * 100:.1f}%")
            st.markdown("""
            **💡 Suggested Retention Actions:**
            - Offer a loyalty discount or free upgrade
            - Reach out with a personal retention call
            - Offer a longer contract with better pricing
            """)
        else:
            st.success("✅ **This customer is NOT likely to churn**")
            st.metric("Churn Probability", f"{prob * 100:.1f}%")
            st.markdown("**Customer appears stable. Continue monitoring engagement.**")
