import os
import pickle

import numpy as np
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_model_path():
    """Find Linear.pkl even if the case differs (Linux is case-sensitive),
    or fall back to the first .pkl file found in the project folder."""
    pkls = []
    for root, _dirs, files in os.walk(BASE_DIR):
        if ".venv" in root or ".git" in root:
            continue
        for name in files:
            if name.lower().endswith(".pkl"):
                pkls.append(os.path.join(root, name))
    for p in pkls:
        if os.path.basename(p).lower() == "linear.pkl":
            return p
    if pkls:
        return pkls[0]
    raise FileNotFoundError(
        "No .pkl model file found in the repo. Commit Linear.pkl to GitHub "
        "next to app.py and redeploy."
    )


MODEL_PATH = find_model_path()


def load_model(path):
    """Load the model with joblib first, then fall back to plain pickle."""
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


model = load_model(MODEL_PATH)

# ---- Detect input features automatically from the trained model ----
if hasattr(model, "feature_names_in_"):
    FEATURES = [str(c) for c in model.feature_names_in_]
elif hasattr(model, "n_features_in_"):
    FEATURES = [f"Feature {i + 1}" for i in range(int(model.n_features_in_))]
else:
    FEATURES = ["Feature 1"]  # edit this list manually if needed

TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Linear Regression Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Poppins', sans-serif;
    min-height: 100vh;
    display: flex; align-items: center; justify-content: center;
    padding: 24px;
    background: linear-gradient(135deg, #0f2027, #203a43 30%, #2c5364 55%, #7b2ff7 100%);
    background-size: 300% 300%;
    animation: bgShift 14s ease infinite;
    color: #fff;
  }
  @keyframes bgShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
  }
  .card {
    width: 100%; max-width: 560px;
    background: rgba(255,255,255,0.10);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 24px;
    padding: 36px 32px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35);
  }
  h1 {
    font-size: 1.9rem; font-weight: 700; text-align: center;
    background: linear-gradient(90deg, #00f5a0, #00d9f5, #c471f5);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .sub { text-align: center; font-weight: 300; opacity: .8; margin: 6px 0 26px; font-size: .95rem; }
  label { display: block; font-size: .82rem; font-weight: 600; margin: 14px 0 6px; letter-spacing: .4px; opacity: .9; }
  input {
    width: 100%; padding: 13px 16px;
    border-radius: 12px; border: 1px solid rgba(255,255,255,0.25);
    background: rgba(255,255,255,0.12); color: #fff;
    font-size: 1rem; font-family: inherit; outline: none; transition: .25s;
  }
  input::placeholder { color: rgba(255,255,255,0.5); }
  input:focus { border-color: #00f5a0; box-shadow: 0 0 0 3px rgba(0,245,160,0.25); background: rgba(255,255,255,0.18); }
  button {
    width: 100%; margin-top: 26px; padding: 14px;
    border: none; border-radius: 14px; cursor: pointer;
    font-size: 1.05rem; font-weight: 600; font-family: inherit; color: #0f2027;
    background: linear-gradient(90deg, #00f5a0, #00d9f5);
    transition: transform .2s, box-shadow .2s;
  }
  button:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(0,245,160,0.35); }
  .result {
    margin-top: 26px; padding: 22px; text-align: center; border-radius: 16px;
    background: linear-gradient(135deg, rgba(0,245,160,0.25), rgba(196,113,245,0.30));
    border: 1px solid rgba(255,255,255,0.25);
    animation: pop .5s ease;
  }
  .result .lbl { font-size: .85rem; opacity: .85; letter-spacing: 1px; text-transform: uppercase; }
  .result .val { font-size: 2.4rem; font-weight: 700; margin-top: 4px; }
  .error { margin-top: 20px; padding: 14px; border-radius: 12px; background: rgba(255,70,90,0.28); text-align: center; font-size: .9rem; }
  @keyframes pop { from { transform: scale(.92); opacity: 0; } to { transform: scale(1); opacity: 1; } }
  footer { text-align: center; margin-top: 22px; font-size: .75rem; opacity: .55; }
</style>
</head>
<body>
  <div class="card">
    <h1>&#128200; Linear Regression Predictor</h1>
    <p class="sub">Enter the values below to get an instant prediction</p>

    <form method="POST" action="/">
      {% for f in features %}
        <label for="f{{ loop.index0 }}">{{ f }}</label>
        <input type="number" step="any" id="f{{ loop.index0 }}" name="f{{ loop.index0 }}"
               placeholder="Enter {{ f }}" value="{{ values[loop.index0] if values else '' }}" required>
      {% endfor %}
      <button type="submit">Predict &#10140;</button>
    </form>

    {% if prediction is not none %}
      <div class="result">
        <div class="lbl">Predicted Value</div>
        <div class="val">{{ prediction }}</div>
      </div>
    {% endif %}
    {% if error %}
      <div class="error">{{ error }}</div>
    {% endif %}

    <footer>Built with Flask &middot; Deployed on Render</footer>
  </div>
</body>
</html>
"""


def run_prediction(raw_values):
    arr = np.array([float(v) for v in raw_values]).reshape(1, -1)
    try:
        import pandas as pd
        if hasattr(model, "feature_names_in_"):
            arr = pd.DataFrame(arr, columns=FEATURES)
    except Exception:
        pass
    result = model.predict(arr)
    return round(float(np.ravel(result)[0]), 4)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction, error, values = None, None, None
    if request.method == "POST":
        values = [request.form.get(f"f{i}", "") for i in range(len(FEATURES))]
        try:
            prediction = run_prediction(values)
        except Exception as e:
            error = f"Could not predict: {e}"
    return render_template_string(
        TEMPLATE, features=FEATURES, prediction=prediction, error=error, values=values
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API: {"features": [1.2, 3.4, ...]}"""
    try:
        data = request.get_json(force=True)
        return jsonify({"prediction": run_prediction(data["features"])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "features": FEATURES})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
