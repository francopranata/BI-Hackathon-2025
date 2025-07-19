import pandas as pd
import joblib
from flask import Flask, request, jsonify
import numpy as np
import xgboost as xgb
import logging
import os
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

app = Flask(__name__)

def download_if_missing(url, dest_path):
    """Mengunduh file dari URL jika belum ada secara lokal."""
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

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("assets", exist_ok=True)

download_if_missing(
    "https://huggingface.co/Franco197/xgb_best_fold_5/resolve/main/xgb_best_fold_5.json",
    "models/xgb_best_fold_5.json"
)
if not os.path.exists("data/final3.csv"):
    logging.info("Downloading final3.csv from Google Drive...")
    os.system("gdown --id 1AnfPWss8Sv5zE_RwyLjuGq01cN__URlq -O data/final3.csv")
download_if_missing(
    "https://raw.githubusercontent.com/francopranata/BI-Hackathon-2025/main/assets/features2.pkl",
    "assets/features2.pkl"
)

MODEL_PATH = "models/xgb_best_fold_5.json"
FEATURE_PATH = "assets/features2.pkl"
DATA_PATH = "data/final3.csv" 

model = xgb.Booster()
model.load_model(MODEL_PATH)
expected_features = joblib.load(FEATURE_PATH)

def get_row_by_account_number(account_number, bank_name=None):
    """Fungsi efisien untuk mengambil baris dari CSV besar berdasarkan nomor rekening dan nama bank."""
    chunksize = 10000
    for chunk in pd.read_csv(DATA_PATH, chunksize=chunksize):
        row = chunk[chunk["account_number"] == int(account_number)]
        if bank_name and not row.empty:
            # Menggunakan .str.upper() untuk pencarian case-insensitive
            row = row[row["bank_name"].str.strip().str.upper() == bank_name.strip().upper()]
        if not row.empty:
            return row
    return pd.DataFrame()

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint untuk prediksi fraud rekening bank."""
    try:
        data = request.json
        account_number = data.get("account_number")
        bank_name_req = data.get("bank_name")

        if not account_number:
            return jsonify({"error": "Missing account_number"}), 400

        row = get_row_by_account_number(account_number, bank_name_req)
        if row.empty:
            return jsonify({"error": f"Account number {account_number} for bank {bank_name_req} not found"}), 404
        
        account_name_raw = row["account_name"].iloc[0]
        bank_name_raw = row["bank_name"].iloc[0]

        account_name_safe = account_name_raw if pd.notna(account_name_raw) else "Nama Tidak Tersedia"
        bank_name_safe = bank_name_raw if pd.notna(bank_name_raw) else "Bank Tidak Diketahui"
        
        x = row[expected_features].copy()
        x = x.astype(np.float32)
        dtest = xgb.DMatrix(x.values, feature_names=expected_features)
        proba = float(model.predict(dtest)[0])

        if proba >= 0.07:
            status = "BERBAHAYA"
        elif proba >= 0.037:
            status = "WARNING"
        else:
            status = "AMAN"

        logging.info(f"[{account_number}] status: {status}, prob: {proba:.4f}")
        return jsonify({
            "account_number": account_number,
            "status": status,
            "probability": round(proba, 4),
            "bank_name": bank_name_safe,
            "account_name": account_name_safe
        })

    except Exception as e:
        logging.error(f"[/predict] Exception occurred: {e}", exc_info=True)
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route("/verify", methods=["POST"])
def verify():
    """Endpoint untuk verifikasi keberadaan rekening di form laporan."""
    try:
        data = request.json
        account_number = data.get("account_number")
        # PERBAIKAN: Ambil bank_name dari request yang dikirim oleh aplikasi
        bank_name_req = data.get("bank_name")

        if not account_number or not bank_name_req:
            return jsonify({"error": "Missing account_number or bank_name"}), 400

        # PERBAIKAN: Teruskan bank_name ke fungsi pencarian untuk validasi yang ketat
        row = get_row_by_account_number(account_number, bank_name_req)

        if not row.empty:
            account_name_raw = row["account_name"].iloc[0]
            bank_name_raw = row["bank_name"].iloc[0]

            account_name_safe = account_name_raw if pd.notna(account_name_raw) else "Nama Tidak Tersedia"
            bank_name_safe = bank_name_raw if pd.notna(bank_name_raw) else "Bank Tidak Diketahui"
            
            logging.info(f"[/verify] Rekening ditemukan: {account_number} di bank {bank_name_req}")
            return jsonify({
                "account_name": account_name_safe,
                "bank_name": bank_name_safe,
            }), 200
        else:
            # Jika tidak ditemukan dengan kombinasi no. rek & bank, kembalikan 404
            logging.info(f"[/verify] Rekening TIDAK ditemukan: {account_number} di bank {bank_name_req}")
            return jsonify({"error": f"Account number {account_number} not found for bank {bank_name_req}"}), 404

    except Exception as e:
        logging.error(f"[/verify] Exception occurred: {e}", exc_info=True)
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001)) 
    app.run(host='0.0.0.0', debug=True, port=port)
