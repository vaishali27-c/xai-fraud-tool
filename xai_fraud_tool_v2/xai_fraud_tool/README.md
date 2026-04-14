# XAI Plug-in Tool: Model-Agnostic Explanations for Fraud Detection

> Transparent, trustworthy fraud detection powered by SHAP + LIME.

---

## Project Structure

```
xai_fraud_tool/
├── backend/
│   ├── app.py               ← FastAPI server (all API routes)
│   ├── requirements.txt     ← Python dependencies
│   └── final_fraud_detection_model.pkl  ← your trained model (copy here)
└── frontend/
    └── index.html           ← Full React dashboard (single file, open in browser)
```

---

## Backend Setup

### 1. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Place your model
Copy `final_fraud_detection_model.pkl` into the `backend/` folder:
```bash
cp /path/to/final_fraud_detection_model.pkl backend/
```

### 3. Run the server
```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be live at: **http://localhost:8000**

Interactive docs: **http://localhost:8000/docs**

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Check model status |
| POST | `/predict` | Predict fraud for one transaction |
| POST | `/explain/shap` | SHAP local explanation |
| POST | `/explain/lime` | LIME local explanation |
| POST | `/explain/both` | SHAP + LIME combined |
| GET | `/global/feature-importance` | Global model feature importances |
| POST | `/batch/predict` | Batch prediction for multiple transactions |

---

## Frontend Setup

No build step needed — it's a single HTML file with React loaded from CDN.

Just open `frontend/index.html` in your browser. Make sure the backend is running on port 8000.

---

## Transaction Input Fields

| Field | Type | Description |
|-------|------|-------------|
| step | int | Time step of transaction |
| type | str | TRANSFER, CASH_OUT, PAYMENT, CASH_IN, DEBIT |
| amount | float | Transaction amount |
| oldbalanceOrg | float | Sender's balance before |
| newbalanceOrig | float | Sender's balance after |
| oldbalanceDest | float | Receiver's balance before |
| newbalanceDest | float | Receiver's balance after |
| isFlaggedFraud | int | System flagged (0 or 1) |

---

## XAI Methods

### SHAP (SHapley Additive exPlanations)
- Uses TreeExplainer for fast, exact explanations on tree-based models
- Shows each feature's contribution to the fraud score
- Positive SHAP = pushes toward fraud, Negative = pushes toward not fraud

### LIME (Local Interpretable Model-agnostic Explanations)
- Fits a simple linear model around the prediction point
- Model-agnostic: works with any classifier
- Shows which feature conditions influenced the decision

---

## Problem Statement

Current fraud detection systems detect fraud accurately but do not explain how decisions are made. This XAI plug-in addresses that gap by providing clear **local explanations** (why was THIS transaction flagged?) and **global explanations** (which features matter most overall?) — improving transparency, trust, and accountability.
