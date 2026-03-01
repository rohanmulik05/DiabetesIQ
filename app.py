"""
DiabetesIQ - Dual Model Prediction Server
Works locally (python app.py) and on Vercel (via api/index.py)
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import os, logging, joblib

# ── Path resolution (works both locally and on Vercel) ──────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Fallback constants (PIMA dataset approximations)
# ─────────────────────────────────────────────────────────────
FALLBACK_MEDIANS = {
    'Glucose':       {0: 107.0, 1: 140.0},
    'BloodPressure': {0: 70.0,  1: 74.0},
    'SkinThickness': {0: 27.0,  1: 32.0},
    'Insulin':       {0: 102.5, 1: 169.5},
    'BMI':           {0: 30.1,  1: 34.3},
}
FALLBACK_INSULIN_CAP = 196.0
FALLBACK_ROBUST = {
    'Pregnancies':              (3.0,   4.0),
    'Glucose':                  (117.0, 37.0),
    'BloodPressure':            (72.0,  18.0),
    'SkinThickness':            (29.0,  14.0),
    'Insulin':                  (125.0, 93.75),
    'BMI':                      (32.0,  9.9),
    'DiabetesPedigreeFunction': (0.37,  0.38),
    'Age':                      (29.0,  14.0),
}

# ─────────────────────────────────────────────────────────────
# Lazy model cache — loaded once on first request, not at import
# This prevents Vercel cold-start crashes
# ─────────────────────────────────────────────────────────────
_cache = {}

def safe_load(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        log.warning(f"Not found: {filename}")
        return None
    try:
        obj = joblib.load(path)
        log.info(f"✓ Loaded: {filename}")
        return obj
    except Exception as e:
        log.error(f"✗ Failed to load {filename}: {e}")
        return None

def get_models():
    """Load all models lazily on first call, cache after."""
    if _cache:
        return _cache

    _cache['clinical_model']       = safe_load("clinical_model.pkl")
    _cache['clinical_scaler']      = safe_load("clinical_preprocessor.pkl")
    _cache['clinical_medians']     = safe_load("clinical_medians.pkl")
    _cache['clinical_insulin_cap'] = safe_load("clinical_insulin_cap.pkl")
    _cache['lifestyle_model']      = safe_load("lifestyle_model.pkl")
    _cache['lifestyle_pipeline']   = safe_load("lifestyle_pipeline.pkl")

    # Rebuild lifestyle pipeline if saved one is broken (sklearn version mismatch)
    if _cache['lifestyle_pipeline'] is None:
        _cache['lifestyle_pipeline'] = _build_lifestyle_pipeline()

    return _cache

'''
# ─────────────────────────────────────────────────────────────
# Lifestyle pipeline rebuild
# ─────────────────────────────────────────────────────────────
LIFESTYLE_COLS = [
    'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
    'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching',
    'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness',
    'Alopecia', 'Obesity', 'Age'
]
BINARY_COLS = [c for c in LIFESTYLE_COLS if c != 'Age']

def _build_lifestyle_pipeline():
    log.info("Rebuilding lifestyle pipeline...")
    rows = []
    for gender in ['Male', 'Female']:
        for yn in ['Yes', 'No']:
            rows.append({'Gender': gender, **{c: yn for c in BINARY_COLS if c != 'Gender'}, 'Age': 30.0})
    for age in [1.0, 100.0]:
        rows.append({'Gender': 'Male', **{c: 'Yes' for c in BINARY_COLS if c != 'Gender'}, 'Age': age})
    df_fit = pd.DataFrame(rows, columns=LIFESTYLE_COLS)
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(drop='first'), BINARY_COLS)],
        remainder='passthrough'
    )
    pipe = Pipeline([('preprocessor', preprocessor), ('scaler', MinMaxScaler())])
    pipe.fit(df_fit)
    log.info("✓ Lifestyle pipeline rebuilt.")
    return pipe
'''

# ─────────────────────────────────────────────────────────────
# Clinical preprocessing
# ─────────────────────────────────────────────────────────────
NUMERIC_COLS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

def preprocess_clinical(raw, m):
    row = {
        'Pregnancies':              float(raw['pregnancies']),
        'Glucose':                  float(raw['glucose']),
        'BloodPressure':            float(raw['blood_pressure']),
        'SkinThickness':            float(raw['skin_thickness']),
        'Insulin':                  float(raw['insulin']),
        'BMI':                      float(raw['bmi']),
        'DiabetesPedigreeFunction': float(raw['dpf']),
        'Age':                      float(raw['age']),
    }
    # 0 → NaN
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        if row[col] == 0:
            row[col] = np.nan
    # Fill NaN
    medians = m['clinical_medians'] or FALLBACK_MEDIANS
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        if np.isnan(row[col]) and col in medians:
            row[col] = (medians[col][0] + medians[col][1]) / 2.0
    # Cap Insulin
    cap = float(m['clinical_insulin_cap']) if m['clinical_insulin_cap'] is not None else FALLBACK_INSULIN_CAP
    if row['Insulin'] > cap:
        row['Insulin'] = cap
    # Feature engineering
    bmi = row['BMI']
    if   bmi < 18.5:  new_bmi = 'Underweight'
    elif bmi <= 24.9: new_bmi = 'Normal'
    elif bmi <= 29.9: new_bmi = 'Overweight'
    elif bmi <= 34.9: new_bmi = 'Obesity 1'
    elif bmi <= 39.9: new_bmi = 'Obesity 2'
    else:             new_bmi = 'Obesity 3'
    new_insulin = 'Normal' if 16 <= row['Insulin'] <= 166 else 'Abnormal'
    g = row['Glucose']
    if   g <= 70:   new_glucose = 'Low'
    elif g <= 99:   new_glucose = 'Normal'
    elif g <= 126:  new_glucose = 'Overweight'
    else:           new_glucose = 'Secret'
    # OHE
    ohe = {
        'NewBMI_Obesity 1':       int(new_bmi == 'Obesity 1'),
        'NewBMI_Obesity 2':       int(new_bmi == 'Obesity 2'),
        'NewBMI_Obesity 3':       int(new_bmi == 'Obesity 3'),
        'NewBMI_Overweight':      int(new_bmi == 'Overweight'),
        'NewBMI_Underweight':     int(new_bmi == 'Underweight'),
        'NewInsulinScore_Normal': int(new_insulin == 'Normal'),
        'NewGlucose_Low':         int(new_glucose == 'Low'),
        'NewGlucose_Normal':      int(new_glucose == 'Normal'),
        'NewGlucose_Overweight':  int(new_glucose == 'Overweight'),
        'NewGlucose_Secret':      int(new_glucose == 'Secret'),
    }
    # RobustScale
    if m['clinical_scaler']:
        scaled = m['clinical_scaler'].transform([[row[c] for c in NUMERIC_COLS]])[0]
    else:
        scaled = np.array([(row[col] - FALLBACK_ROBUST[col][0]) / FALLBACK_ROBUST[col][1] for col in NUMERIC_COLS])
    return np.concatenate([scaled, list(ohe.values())]).reshape(1, -1)


# ─────────────────────────────────────────────────────────────
# Lifestyle preprocessing
# ─────────────────────────────────────────────────────────────
BIN = {0: 'No', 1: 'Yes'}
GEN = {0: 'Female', 1: 'Male'}

def preprocess_lifestyle(raw, m):
    row = {
        'Gender':             GEN[int(raw['gender'])],
        'Polyuria':           BIN[int(raw['polyuria'])],
        'Polydipsia':         BIN[int(raw['polydipsia'])],
        'sudden weight loss': BIN[int(raw['weight_loss'])],
        'weakness':           BIN[int(raw['weakness'])],
        'Polyphagia':         BIN[int(raw['polyphagia'])],
        'Genital thrush':     BIN[int(raw['genital_thrush'])],
        'visual blurring':    BIN[int(raw['visual_blurring'])],
        'Itching':            BIN[int(raw['itching'])],
        'Irritability':       BIN[int(raw['irritability'])],
        'delayed healing':    BIN[int(raw['delayed_healing'])],
        'partial paresis':    BIN[int(raw['partial_paresis'])],
        'muscle stiffness':   BIN[int(raw['muscle_stiffness'])],
        'Alopecia':           BIN[int(raw['alopecia'])],
        'Obesity':            BIN[int(raw['obesity'])],
        'Age':                float(raw['age']),
    }
    df_row = pd.DataFrame([row], columns=LIFESTYLE_COLS)
    return m['lifestyle_pipeline'].transform(df_row)


def run_predict(model, X):
    pred = int(model.predict(X)[0])
    if hasattr(model, 'predict_proba'):
        prob = float(model.predict_proba(X)[0][1])
    elif hasattr(model, 'decision_function'):
        prob = float(1 / (1 + np.exp(-model.decision_function(X)[0])))
    else:
        prob = float(pred)
    return prob, pred


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    m = get_models()
    return jsonify({
        "clinical_model_loaded":     m['clinical_model'] is not None,
        "lifestyle_model_loaded":    m['lifestyle_model'] is not None,
        "clinical_scaler_loaded":    m['clinical_scaler'] is not None,
        "lifestyle_pipeline_loaded": m['lifestyle_pipeline'] is not None,
        "clinical_medians_loaded":   m['clinical_medians'] is not None,
    })


@app.route("/predict/clinical", methods=["POST"])
def predict_clinical_route():
    m = get_models()
    if m['clinical_model'] is None:
        return jsonify({"error": "clinical_model.pkl not found in /models"}), 503
    try:
        X = preprocess_clinical(request.get_json(), m)
        prob, pred = run_predict(m['clinical_model'], X)
        return jsonify({"probability": prob, "prediction": pred})
    except Exception as e:
        log.exception("Clinical prediction error")
        return jsonify({"error": str(e)}), 400


@app.route("/predict/lifestyle", methods=["POST"])
def predict_lifestyle_route():
    m = get_models()
    if m['lifestyle_model'] is None:
        return jsonify({"error": "lifestyle_model.pkl not found in /models"}), 503
    try:
        X = preprocess_lifestyle(request.get_json(), m)
        prob, pred = run_predict(m['lifestyle_model'], X)
        return jsonify({"probability": prob, "prediction": pred})
    except Exception as e:
        log.exception("Lifestyle prediction error")
        return jsonify({"error": str(e)}), 400


@app.route("/predict/combined", methods=["POST"])
def predict_combined_route():
    m = get_models()
    if m['clinical_model'] is None or m['lifestyle_model'] is None:
        return jsonify({"error": "Both models must be loaded for combined prediction"}), 503
    try:
        data = request.get_json()
        Xc = preprocess_clinical(data, m)
        Xl = preprocess_lifestyle(data, m)
        cp, cpred = run_predict(m['clinical_model'],  Xc)
        lp, lpred = run_predict(m['lifestyle_model'], Xl)
        ep = (cp + lp) / 2.0
        return jsonify({
            "clinical":  {"probability": cp,  "prediction": cpred},
            "lifestyle": {"probability": lp,  "prediction": lpred},
            "combined":  {"probability": ep,  "prediction": int(ep >= 0.5)},
        })
    except Exception as e:
        log.exception("Combined prediction error")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
