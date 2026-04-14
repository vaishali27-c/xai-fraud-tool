"""
XAI Plug-in Tool: Model-Agnostic Explanations for Fraud Detection
FastAPI Backend v2.0 — Advanced Features Edition
New: Counterfactual, Batch CSV, enhanced SHAP/LIME
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import io, os

app = FastAPI(title="XAI Fraud Detection API v2", version="2.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "final_fraud_detection_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"[WARN] Could not load model: {e}")

FEATURE_NAMES = [
    "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "isFlaggedFraud",
    "balanceDiffOrig", "balanceDiffDest"
]
TYPE_MAP = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}

class TransactionInput(BaseModel):
    step: int = 1
    type: str = "TRANSFER"
    amount: float = 10000.0
    oldbalanceOrg: float = 50000.0
    newbalanceOrig: float = 40000.0
    oldbalanceDest: float = 5000.0
    newbalanceDest: float = 15000.0
    isFlaggedFraud: int = 0

class BatchInput(BaseModel):
    transactions: list[TransactionInput]

def preprocess(t: TransactionInput) -> np.ndarray:
    type_encoded = TYPE_MAP.get(t.type.upper(), 3)
    return np.array([[
        t.step, type_encoded, t.amount,
        t.oldbalanceOrg, t.newbalanceOrig,
        t.oldbalanceDest, t.newbalanceDest,
        t.isFlaggedFraud,
        t.oldbalanceOrg - t.newbalanceOrig,
        t.newbalanceDest - t.oldbalanceDest
    ]])

def get_risk_label(prob: float) -> str:
    if prob >= 0.75: return "HIGH"
    if prob >= 0.40: return "MEDIUM"
    return "LOW"

@app.get("/")
def root(): return {"status": "ok", "version": "2.0.0"}

@app.get("/health")
def health(): return {"model_loaded": model is not None, "features": FEATURE_NAMES}

@app.post("/predict")
def predict(transaction: TransactionInput):
    if model is None: raise HTTPException(503, "Model not loaded")
    X = preprocess(transaction)
    prob = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])
    return {"prediction": pred, "fraud_probability": round(prob, 4),
            "risk_level": get_risk_label(prob), "is_fraud": bool(pred == 1)}

@app.post("/explain/shap")
def explain_shap(transaction: TransactionInput):
    if model is None: raise HTTPException(503, "Model not loaded")
    X = preprocess(transaction)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(df)
        if isinstance(shap_vals, list):
            vals, base = shap_vals[1][0], float(explainer.expected_value[1])
        else:
            vals, base = shap_vals[0], float(explainer.expected_value)
        contributions = [
            {"feature": FEATURE_NAMES[i], "shap_value": round(float(vals[i]), 5),
             "feature_value": round(float(X[0][i]), 4)}
            for i in range(len(FEATURE_NAMES))
        ]
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        prob = float(model.predict_proba(X)[0][1])
        return {"method": "SHAP", "base_value": round(base, 5),
                "fraud_probability": round(prob, 4), "risk_level": get_risk_label(prob),
                "contributions": contributions}
    except Exception as e:
        raise HTTPException(500, f"SHAP error: {str(e)}")

@app.post("/explain/lime")
def explain_lime(transaction: TransactionInput):
    if model is None: raise HTTPException(503, "Model not loaded")
    X = preprocess(transaction)
    np.random.seed(42)
    n = 500
    bg = np.column_stack([
        np.random.randint(1,744,n), np.random.randint(0,5,n),
        np.random.exponential(50000,n), np.random.uniform(0,1e6,n),
        np.random.uniform(0,1e6,n), np.random.uniform(0,1e6,n),
        np.random.uniform(0,1e6,n), np.zeros(n),
        np.random.uniform(-1e5,1e5,n), np.random.uniform(-1e5,1e5,n),
    ])
    try:
        lime_exp = LimeTabularExplainer(bg, feature_names=FEATURE_NAMES,
            class_names=["Not Fraud","Fraud"], mode="classification", random_state=42)
        exp = lime_exp.explain_instance(X[0], model.predict_proba, num_features=10, num_samples=300)
        contributions = [{"feature": f, "lime_weight": round(float(w), 5)} for f,w in exp.as_list()]
        prob = float(model.predict_proba(X)[0][1])
        return {"method": "LIME", "fraud_probability": round(prob,4),
                "risk_level": get_risk_label(prob), "contributions": contributions,
                "intercept": round(float(exp.intercept[1]),5)}
    except Exception as e:
        raise HTTPException(500, f"LIME error: {str(e)}")

@app.post("/explain/both")
def explain_both(transaction: TransactionInput):
    return {"prediction": predict(transaction), "shap": explain_shap(transaction),
            "lime": explain_lime(transaction)}

@app.get("/global/feature-importance")
def global_feature_importance():
    if model is None: raise HTTPException(503, "Model not loaded")
    try:
        importances = model.feature_importances_
        result = [{"feature": FEATURE_NAMES[i], "importance": round(float(importances[i]),5)}
                  for i in range(len(FEATURE_NAMES))]
        result.sort(key=lambda x: x["importance"], reverse=True)
        return {"feature_importances": result}
    except Exception as e:
        raise HTTPException(500, str(e))

# ─── COUNTERFACTUAL EXPLANATION ───────────────────────────────────────────────
@app.post("/explain/counterfactual")
def counterfactual(transaction: TransactionInput):
    if model is None: raise HTTPException(503, "Model not loaded")
    X = preprocess(transaction)
    orig_prob = float(model.predict_proba(X)[0][1])

    if orig_prob < 0.4:
        return {"original_probability": round(orig_prob,4), "original_risk": get_risk_label(orig_prob),
                "already_safe": True, "counterfactuals": [],
                "message": "Transaction is already LOW RISK. No changes needed."}

    suggestions = []

    # 1. Reduce amount
    for factor in [0.5, 0.3, 0.1]:
        t2 = TransactionInput(
            step=transaction.step, type=transaction.type,
            amount=transaction.amount * factor,
            oldbalanceOrg=transaction.oldbalanceOrg,
            newbalanceOrig=transaction.oldbalanceOrg - (transaction.amount * factor),
            oldbalanceDest=transaction.oldbalanceDest,
            newbalanceDest=transaction.newbalanceDest,
            isFlaggedFraud=transaction.isFlaggedFraud
        )
        X2 = preprocess(t2)
        p2 = float(model.predict_proba(X2)[0][1])
        if p2 < orig_prob:
            suggestions.append({
                "change": f"Reduce amount to ₹{t2.amount:,.0f} ({int((1-factor)*100)}% reduction)",
                "field": "amount", "original_value": transaction.amount,
                "new_value": round(t2.amount, 2), "new_probability": round(p2,4),
                "new_risk": get_risk_label(p2), "reduction": round((orig_prob-p2)*100,1)
            })
            if p2 < 0.4: break

    # 2. Change transaction type to safer types
    for stype in ["PAYMENT", "CASH_IN"]:
        if stype == transaction.type.upper(): continue
        t2 = TransactionInput(
            step=transaction.step, type=stype, amount=transaction.amount,
            oldbalanceOrg=transaction.oldbalanceOrg, newbalanceOrig=transaction.newbalanceOrig,
            oldbalanceDest=transaction.oldbalanceDest, newbalanceDest=transaction.newbalanceDest,
            isFlaggedFraud=transaction.isFlaggedFraud
        )
        X2 = preprocess(t2)
        p2 = float(model.predict_proba(X2)[0][1])
        if p2 < orig_prob:
            suggestions.append({
                "change": f"Change type from {transaction.type} → {stype}",
                "field": "type", "original_value": transaction.type,
                "new_value": stype, "new_probability": round(p2,4),
                "new_risk": get_risk_label(p2), "reduction": round((orig_prob-p2)*100,1)
            })

    # 3. Keep partial balance (don't drain account)
    if transaction.newbalanceOrig == 0 and transaction.oldbalanceOrg > 0:
        keep = transaction.oldbalanceOrg * 0.3
        t2 = TransactionInput(
            step=transaction.step, type=transaction.type,
            amount=transaction.oldbalanceOrg * 0.7,
            oldbalanceOrg=transaction.oldbalanceOrg, newbalanceOrig=keep,
            oldbalanceDest=transaction.oldbalanceDest, newbalanceDest=transaction.newbalanceDest,
            isFlaggedFraud=transaction.isFlaggedFraud
        )
        X2 = preprocess(t2)
        p2 = float(model.predict_proba(X2)[0][1])
        if p2 < orig_prob:
            suggestions.append({
                "change": f"Keep ₹{keep:,.0f} remaining in sender account (avoid full drain)",
                "field": "newbalanceOrig", "original_value": 0,
                "new_value": round(keep,2), "new_probability": round(p2,4),
                "new_risk": get_risk_label(p2), "reduction": round((orig_prob-p2)*100,1)
            })

    # 4. Split transaction
    split = transaction.amount / 3
    t2 = TransactionInput(
        step=transaction.step, type=transaction.type, amount=split,
        oldbalanceOrg=transaction.oldbalanceOrg,
        newbalanceOrig=transaction.oldbalanceOrg - split,
        oldbalanceDest=transaction.oldbalanceDest, newbalanceDest=transaction.newbalanceDest,
        isFlaggedFraud=transaction.isFlaggedFraud
    )
    X2 = preprocess(t2)
    p2 = float(model.predict_proba(X2)[0][1])
    if p2 < orig_prob:
        suggestions.append({
            "change": f"Split into 3 transactions of ₹{split:,.0f} each",
            "field": "amount", "original_value": transaction.amount,
            "new_value": round(split,2), "new_probability": round(p2,4),
            "new_risk": get_risk_label(p2), "reduction": round((orig_prob-p2)*100,1)
        })

    suggestions.sort(key=lambda x: x["new_probability"])
    return {
        "original_probability": round(orig_prob,4),
        "original_risk": get_risk_label(orig_prob),
        "already_safe": False,
        "counterfactuals": suggestions[:4],
        "message": f"Found {len(suggestions)} ways to reduce fraud risk"
    }

# ─── BATCH CSV UPLOAD ─────────────────────────────────────────────────────────
@app.post("/batch/upload-csv")
async def batch_upload_csv(file: UploadFile = File(...)):
    if model is None: raise HTTPException(503, "Model not loaded")
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        df.columns = [c.strip() for c in df.columns]
        required = ["step","type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]
        missing = [c for c in required if c not in df.columns]
        if missing: raise HTTPException(400, f"Missing columns: {missing}")

        df = df.head(500)
        if "isFlaggedFraud" not in df.columns: df["isFlaggedFraud"] = 0
        df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
        df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
        df["type_enc"] = df["type"].map(TYPE_MAP).fillna(3)

        X = df[["step","type_enc","amount","oldbalanceOrg","newbalanceOrig",
                "oldbalanceDest","newbalanceDest","isFlaggedFraud",
                "balanceDiffOrig","balanceDiffDest"]].values

        probs = model.predict_proba(X)[:,1]
        preds = model.predict(X)

        results = []
        for i in range(len(df)):
            prob = float(probs[i])
            results.append({
                "row": i+1, "type": str(df["type"].iloc[i]),
                "amount": float(df["amount"].iloc[i]),
                "fraud_probability": round(prob,4),
                "risk_level": get_risk_label(prob),
                "is_fraud": bool(preds[i]==1)
            })

        fraud_count = sum(1 for r in results if r["is_fraud"])
        type_fraud = {}
        for r in results:
            t = r["type"]
            if t not in type_fraud: type_fraud[t] = {"total":0,"fraud":0}
            type_fraud[t]["total"] += 1
            if r["is_fraud"]: type_fraud[t]["fraud"] += 1

        return {
            "summary": {
                "total_transactions": len(results),
                "fraud_detected": fraud_count,
                "fraud_percentage": round(fraud_count/len(results)*100,2),
                "high_risk_count": sum(1 for r in results if r["risk_level"]=="HIGH"),
                "average_fraud_probability": round(float(np.mean(probs)),4),
            },
            "type_breakdown": type_fraud,
            "results": results
        }
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")

@app.post("/batch/predict")
def batch_predict(batch: BatchInput):
    if model is None: raise HTTPException(503, "Model not loaded")
    results = []
    for t in batch.transactions:
        X = preprocess(t)
        prob = float(model.predict_proba(X)[0][1])
        pred = int(model.predict(X)[0])
        results.append({"fraud_probability": round(prob,4),
                        "risk_level": get_risk_label(prob), "is_fraud": bool(pred==1)})
    return {"results": results, "total": len(results),
            "fraud_count": sum(1 for r in results if r["is_fraud"])}
