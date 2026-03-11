import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# ── All categorical columns in your dataset ──────────────────────────────────
CAT_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    # Drop customerID (not useful for prediction)
    df.drop('customerID', axis=1, inplace=True)

    # Fix TotalCharges — it's stored as string in your dataset
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Encode target column
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Encode all categorical columns
    le = LabelEncoder()
    label_mappings = {}
    for col in CAT_COLS:
        df[col] = le.fit_transform(df[col])
        label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    return df, label_mappings


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

    # Save model and feature names
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)

    return model, acc, report, cm, X.columns.tolist()