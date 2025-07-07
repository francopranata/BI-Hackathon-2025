import pandas as pd
import joblib
from flask import Flask, request, jsonify
import numpy as np
import xgboost as xgb
import logging
import os
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# === Fungsi unduh file jika tidak tersedia ===
def download_if_missing(url, dest_path):
    if not os.path.exists(dest_path):
        logging.info(f"Downloading {dest_path} from {url} ...")
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise Exception(f"Failed to download {url}: {r.status_code}")
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logging.info(f"Downloaded {dest_path}")

# === Setup direktori ===
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("assets", exist_ok=True)

# Unduh model dari Hugging Face
download_if_missing(
    "https://huggingface.co/Franco197/xgb_best_fold_5/resolve/main/xgb_best_fold_5.json",
    "models/xgb_best_fold_5.json"
)

# Unduh final1.csv dari Google Drive menggunakan gdown
if not os.path.exists("data/final1.csv"):
    logging.info("Downloading final1.csv from Google Drive...")
    os.system("gdown --id 11CSR7aw0WQEx2WOasgqBNfZwLsmUfrS9 -O data/final1.csv")

# Unduh features dari GitHub
download_if_missing(
    "https://raw.githubusercontent.com/francopranata/BI-Hackathon-2025/main/assets/features2.pkl",
    "assets/features2.pkl"
)

# === Load assets ===
MODEL_PATH = "models/xgb_best_fold_5.json"
FEATURE_PATH = "assets/features2.pkl"
DATA_PATH = "data/final1.csv"

model = xgb.Booster()
model.load_model(MODEL_PATH)
expected_features = joblib.load(FEATURE_PATH)

# Fungsi efisien ambil baris dari CSV besar berdasarkan account_number
def get_row_by_account_number(account_number):
    chunksize = 10000
    for chunk in pd.read_csv(DATA_PATH, chunksize=chunksize):
        row = chunk[chunk["account_number"] == int(account_number)]
        if not row.empty:
            return row
    return pd.DataFrame()

# === Flask API ===
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        account_number = request.json.get("account_number")
        if account_number is None:
            return jsonify({"error": "Missing account_number"}), 400

        row = get_row_by_account_number(account_number)
        if row.empty:
            return jsonify({"error": f"Account number {account_number} not found"}), 404

        x = row[expected_features].copy()

        object_cols = x.select_dtypes(include='object').columns.tolist()
        if object_cols:
            return jsonify({
                "error": f"Kolom berikut masih berupa string dan harus diubah ke numerik: {object_cols}"
            }), 400

        if set(x.columns) != set(expected_features):
            return jsonify({
                "error": "Mismatch kolom fitur",
                "expected_not_found": list(set(expected_features) - set(x.columns)),
                "unexpected_present": list(set(x.columns) - set(expected_features))
            }), 400

        x = x.astype(np.float32)
        dtest = xgb.DMatrix(x.values, feature_names=expected_features)
        proba = float(model.predict(dtest)[0])

        if proba >= 0.05:
            status = "BERBAHAYA"
        elif proba >= 0.03:
            status = "WARNING"
        else:
            status = "AMAN"

        logging.info(f"[{account_number}] status: {status}, prob: {proba:.4f}")
        return jsonify({
            "account_number": account_number,
            "status": status,
            "probability": round(proba, 4)
        })

    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route("/verify", methods=["POST"])
def verify():
    try:
        account_number = request.json.get("account_number")
        if not account_number:
            return jsonify({"error": "Missing account_number"}), 400

        # Gunakan fungsi yang sudah ada untuk mencari rekening
        row = get_row_by_account_number(account_number)

        if not row.empty:
            # Jika rekening ditemukan, kembalikan sukses dengan nama samaran
            logging.info(f"[/verify] Rekening ditemukan: {account_number}")
            return jsonify({"account_name": "A*** N***"}), 200
        else:
            # Jika tidak ditemukan, kembalikan error 404
            logging.info(f"[/verify] Rekening TIDAK ditemukan: {account_number}")
            return jsonify({"error": f"Account number {account_number} not found"}), 404

    except Exception as e:
        logging.error(f"[/verify] Exception occurred: {e}", exc_info=True)
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    # Ganti port ke 5001 atau port lain yang bebas
    port = int(os.environ.get("PORT", 5001)) 
    app.run(host='0.0.0.0', debug=True, port=port)