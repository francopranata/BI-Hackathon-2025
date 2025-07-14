# import pandas as pd
# import joblib
# from flask import Flask, request, jsonify
# import numpy as np
# import xgboost as xgb
# import logging
# import os
# import requests

# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# # === Fungsi unduh file jika tidak tersedia ===
# def download_if_missing(url, dest_path):
#     if not os.path.exists(dest_path):
#         logging.info(f"Downloading {dest_path} from {url} ...")
#         r = requests.get(url, stream=True)
#         if r.status_code != 200:
#             raise Exception(f"Failed to download {url}: {r.status_code}")
#         with open(dest_path, "wb") as f:
#             for chunk in r.iter_content(chunk_size=8192):
#                 if chunk:
#                     f.write(chunk)
#         logging.info(f"Downloaded {dest_path}")


# # === Setup direktori ===
# os.makedirs("models", exist_ok=True)
# os.makedirs("data", exist_ok=True)
# os.makedirs("assets", exist_ok=True)

# # Unduh model dari Hugging Face
# download_if_missing(
#     "https://huggingface.co/Franco197/xgb_best_fold_5/resolve/main/xgb_best_fold_5.json",
#     "models/xgb_best_fold_5.json"
# )

# if not os.path.exists("data/final3.csv"):
#     logging.info("Downloading final3.csv from Google Drive...")
#     os.system("gdown --id 1AnfPWss8Sv5zE_RwyLjuGq01cN__URlq -O data/final3.csv")

# # Unduh features dari GitHub
# download_if_missing(
#     "https://raw.githubusercontent.com/francopranata/BI-Hackathon-2025/main/assets/features2.pkl",
#     "assets/features2.pkl"
# )

# # === Load assets ===
# MODEL_PATH = "models/xgb_best_fold_5.json"
# FEATURE_PATH = "assets/features2.pkl"
# DATA_PATH = "data/final3.csv"

# model = xgb.Booster()
# model.load_model(MODEL_PATH)
# expected_features = joblib.load(FEATURE_PATH)

# # Fungsi efisien ambil baris dari CSV besar berdasarkan account_number
# def get_row_by_account_number(account_number):
#     chunksize = 10000
#     for chunk in pd.read_csv(DATA_PATH, chunksize=chunksize):
#         row = chunk[chunk["account_number"] == int(account_number)]
#         if not row.empty:
#             return row
#     return pd.DataFrame()

# # === Flask API ===
# app = Flask(__name__)

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         account_number = request.json.get("account_number")
#         if account_number is None:
#             return jsonify({"error": "Missing account_number"}), 400

#         row = get_row_by_account_number(account_number)
#         if row.empty:
#             return jsonify({"error": f"Account number {account_number} not found"}), 404
        
#         bank_name = row["bank_name"].iloc[0]
#         real_name = row["account_name"].iloc[0]

#         x = row[expected_features].copy()

#         object_cols = x.select_dtypes(include='object').columns.tolist()
#         if object_cols:
#             return jsonify({
#                 "error": f"Kolom berikut masih berupa string dan harus diubah ke numerik: {object_cols}"
#             }), 400

#         if set(x.columns) != set(expected_features):
#             return jsonify({
#                 "error": "Mismatch kolom fitur",
#                 "expected_not_found": list(set(expected_features) - set(x.columns)),
#                 "unexpected_present": list(set(x.columns) - set(expected_features))
#             }), 400

#         x = x.astype(np.float32)
#         dtest = xgb.DMatrix(x.values, feature_names=expected_features)
#         proba = float(model.predict(dtest)[0])

#         if proba >= 0.05:
#             status = "BERBAHAYA"
#         elif proba >= 0.03:
#             status = "WARNING"
#         else:
#             status = "AMAN"

#         logging.info(f"[{account_number}] status: {status}, prob: {proba:.4f}")
#         return jsonify({
#             "account_number": account_number,
#             "status": status,
#             "probability": round(proba, 4),
#             "bank_name": bank_name,
#             "account_name": real_name 
#         })

#     except Exception as e:
#         logging.error("Exception occurred", exc_info=True)
#         return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# @app.route("/verify", methods=["POST"])
# def verify():
#     try:
#         account_number = request.json.get("account_number")
#         if not account_number:
#             return jsonify({"error": "Missing account_number"}), 400

#         # Gunakan fungsi yang sudah ada untuk mencari rekening
#         row = get_row_by_account_number(account_number)

#         if not row.empty:
#             real_name = row["account_name"].iloc[0]
#             bank_name = row["bank_name"].iloc[0]

#             logging.info(f"[/verify] Rekening ditemukan: {account_number}")
#             return jsonify({
#                 "account_name": real_name,
#                 "bank_name": bank_name,
#             }), 200
#         else:
#             # Jika tidak ditemukan, kembalikan error 404
#             logging.info(f"[/verify] Rekening TIDAK ditemukan: {account_number}")
#             return jsonify({"error": f"Account number {account_number} not found"}), 404

#     except Exception as e:
#         logging.error(f"[/verify] Exception occurred: {e}", exc_info=True)
#         return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

# if __name__ == "__main__":
#     # Ganti port ke 5001 atau port lain yang bebas
#     port = int(os.environ.get("PORT", 5001)) 
#     app.run(host='0.0.0.0', debug=True, port=port)


import pandas as pd
import joblib
from flask import Flask, request, jsonify
import numpy as np
import xgboost as xgb
import logging
import os
import requests
# from dotenv import load_dotenv

# 1. SETUP & KONFIGURASI
# =================================

# Muat variabel dari file .env (jika ada)
# load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

app = Flask(__name__)

# --- Fungsi Helper ---

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

def mask_name(full_name):
    """Menyamarkan nama, misal 'Franco Antonio' -> 'F***** A******'."""
    if not isinstance(full_name, str):
        return ""
    parts = full_name.split()
    masked_parts = []
    for part in parts:
        if len(part) > 1:
            masked_parts.append(part[0] + '*' * (len(part) - 1))
        else:
            masked_parts.append(part)
    return ' '.join(masked_parts)

# --- Setup Direktori & Unduh Aset ---

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

# 2. MEMUAT MODEL & DATA
# ========================

MODEL_PATH = "models/xgb_best_fold_5.json"
FEATURE_PATH = "assets/features2.pkl"
# PERBAIKAN: Pastikan path data menunjuk ke file yang benar
DATA_PATH = "data/final3.csv" 

model = xgb.Booster()
model.load_model(MODEL_PATH)
expected_features = joblib.load(FEATURE_PATH)

def get_row_by_account_number(account_number, bank_name=None):
    """Fungsi efisien untuk mengambil baris dari CSV besar berdasarkan nomor rekening dan nama bank."""
    chunksize = 10000
    for chunk in pd.read_csv(DATA_PATH, chunksize=chunksize):
        # Filter awal berdasarkan nomor rekening
        row = chunk[chunk["account_number"] == int(account_number)]
        
        # PERBAIKAN: Jika bank_name diberikan, filter lebih lanjut
        if bank_name and not row.empty:
            # Menggunakan .str.upper() untuk pencarian case-insensitive
            row = row[row["bank_name"].str.upper() == bank_name.upper()]

        if not row.empty:
            return row
    return pd.DataFrame()


# 3. ENDPOINT FLASK API
# ======================

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint untuk prediksi fraud rekening bank."""
    try:
        data = request.json
        account_number = data.get("account_number")
        bank_name = data.get("bank_name") # Ambil bank_name dari request

        if not account_number:
            return jsonify({"error": "Missing account_number"}), 400

        # PERBAIKAN: Teruskan bank_name ke fungsi pencarian
        row = get_row_by_account_number(account_number, bank_name)
        if row.empty:
            # Jika tidak ditemukan dengan kombinasi no. rek & bank, kembalikan 404
            return jsonify({"error": f"Account number {account_number} for bank {bank_name} not found"}), 404
        
        # Ambil nama asli dan samarkan
        real_name = row["account_name"].iloc[0]
        masked_name = mask_name(real_name)
        
        # Ekstrak fitur untuk model
        x = row[expected_features].copy()
        
        # ... (Validasi fitur tidak berubah) ...

        x = x.astype(np.float32)
        dtest = xgb.DMatrix(x.values, feature_names=expected_features)
        proba = float(model.predict(dtest)[0])

        if proba >= 0.07: # Menggunakan ambang batas dari analisis fraud rate
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
            "bank_name": bank_name,
            "account_name": masked_name # Kirim nama yang sudah disamarkan
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
        bank_name = data.get("bank_name") # Ambil bank_name dari request

        if not account_number:
            return jsonify({"error": "Missing account_number"}), 400

        # PERBAIKAN: Teruskan bank_name ke fungsi pencarian
        row = get_row_by_account_number(account_number, bank_name)

        if not row.empty:
            real_name = row["account_name"].iloc[0]
            masked_name = mask_name(real_name)
            
            logging.info(f"[/verify] Rekening ditemukan: {account_number}")
            return jsonify({
                "account_name": masked_name,
                "bank_name": bank_name,
            }), 200
        else:
            logging.info(f"[/verify] Rekening TIDAK ditemukan: {account_number} di bank {bank_name}")
            return jsonify({"error": f"Account number {account_number} not found"}), 404

    except Exception as e:
        logging.error(f"[/verify] Exception occurred: {e}", exc_info=True)
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


# 4. MENJALANKAN SERVER
# ======================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001)) 
    app.run(host='0.0.0.0', debug=True, port=port)
