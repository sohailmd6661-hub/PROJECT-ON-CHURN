"""
ChurnIQ — Flask API for Telco Customer Churn Prediction
Pipeline: raw numeric → sqrt(tenure) → StandardScaler → LogisticRegression
"""
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import os
import sys

app = Flask(__name__)

# ── Load artifacts ──────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, "MODEL.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE, "standscaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

FEATURE_ORDER = list(scaler.feature_names_in_)

# ── Helper mappings ─────────────────────────────────────────────────────────
def add_sim(payment_method: str) -> str:
    return {
        "Electronic check":          "Reliance Jio",
        "Mailed check":              "Airtel",
        "Bank transfer (automatic)": "Vi-idea",
    }.get(payment_method, "BSNL")

def contract_ordinal(contract: str) -> float:
    return {"Month-to-month": 0.0, "One year": 1.0, "Two year": 2.0}.get(contract, 0.0)

# ── Preprocessing ───────────────────────────────────────────────────────────
def preprocess(data: dict) -> np.ndarray:
    """
    Convert raw form input → scaled feature vector (1, 32).
    Numeric transformations matching var_tran.py pipeline:
      - MonthlyCharges  → pass raw (yeo-johnson ≈ identity for this range after IQR trim)
      - TotalCharges    → pass raw (boxcox ≈ identity after IQR trim)
      - tenure          → sqrt(tenure)
    """
    sim             = add_sim(data["PaymentMethod"])
    total_charges   = float(data.get("TotalCharges") or 0)
    monthly_charges = float(data.get("MonthlyCharges") or 0)
    tenure          = float(data.get("tenure") or 0)

    def flag(col, val):
        return 1 if data.get(col) == val else 0

    features = {
        "SeniorCitizen":                        int(data.get("SeniorCitizen", 0)),
        "TotalCharges_replaced":                total_charges,
        "MonthlyCharges_yeo_trim":              monthly_charges,
        "tenure_sqrt_trim":                     np.sqrt(tenure),
        "gender_Male":                          flag("gender", "Male"),
        "Partner_Yes":                          flag("Partner", "Yes"),
        "Dependents_Yes":                       flag("Dependents", "Yes"),
        "PhoneService_Yes":                     flag("PhoneService", "Yes"),
        "MultipleLines_No phone service":       flag("MultipleLines", "No phone service"),
        "MultipleLines_Yes":                    flag("MultipleLines", "Yes"),
        "InternetService_Fiber optic":          flag("InternetService", "Fiber optic"),
        "InternetService_No":                   flag("InternetService", "No"),
        "OnlineSecurity_No internet service":   flag("OnlineSecurity", "No internet service"),
        "OnlineSecurity_Yes":                   flag("OnlineSecurity", "Yes"),
        "OnlineBackup_No internet service":     flag("OnlineBackup", "No internet service"),
        "OnlineBackup_Yes":                     flag("OnlineBackup", "Yes"),
        "DeviceProtection_No internet service": flag("DeviceProtection", "No internet service"),
        "DeviceProtection_Yes":                 flag("DeviceProtection", "Yes"),
        "TechSupport_No internet service":      flag("TechSupport", "No internet service"),
        "TechSupport_Yes":                      flag("TechSupport", "Yes"),
        "StreamingTV_No internet service":      flag("StreamingTV", "No internet service"),
        "StreamingTV_Yes":                      flag("StreamingTV", "Yes"),
        "StreamingMovies_No internet service":  flag("StreamingMovies", "No internet service"),
        "StreamingMovies_Yes":                  flag("StreamingMovies", "Yes"),
        "PaperlessBilling_Yes":                 flag("PaperlessBilling", "Yes"),
        "PaymentMethod_Credit card (automatic)": flag("PaymentMethod", "Credit card (automatic)"),
        "PaymentMethod_Electronic check":        flag("PaymentMethod", "Electronic check"),
        "PaymentMethod_Mailed check":            flag("PaymentMethod", "Mailed check"),
        "Sim_BSNL":         1 if sim == "BSNL"         else 0,
        "Sim_Reliance Jio": 1 if sim == "Reliance Jio" else 0,
        "Sim_Vi-idea":      1 if sim == "Vi-idea"       else 0,
        "Contract_re":      contract_ordinal(data.get("Contract", "Month-to-month")),
    }

    df     = pd.DataFrame([{f: features[f] for f in FEATURE_ORDER}])
    scaled = scaler.transform(df)
    return scaled


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        vec  = preprocess(data)

        pred  = int(model.predict(vec)[0])
        prob  = float(model.predict_proba(vec)[0][1])

        risk_level = (
            "Low"    if prob < 0.35 else
            "Medium" if prob < 0.65 else
            "High"
        )

        return jsonify({
            "prediction": pred,
            "churn":       bool(pred),
            "probability": round(prob * 100, 1),
            "risk_level":  risk_level,
        })

    except Exception as e:
        _, _, tb = sys.exc_info()
        return jsonify({"error": str(e), "line": tb.tb_lineno}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)