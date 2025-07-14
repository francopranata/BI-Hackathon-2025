import os
import pandas as pd
import torch
import torch.nn.functional as F
from flask import Flask, jsonify
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import Data
from torch_geometric.nn import GINConv

# --- 1. KONFIGURASI & SETUP AWAL ---
app = Flask(__name__)

# Pastikan nama file ini sesuai dengan nama file Anda
FEATURES_PATH = 'data/dummy_elliptic_txs_features1.csv'
EDGELIST_PATH = 'data/elliptic_txs_edgelist.csv'
MODEL_PATH = 'models/gin_model.pth' 

# Pastikan mapping kelas ini sesuai dengan hasil training Anda
# Umumnya 0: licit, 1: illicit
CLASS_MAP = {0: "AMAN", 1: "BERBAHAYA", 2: "TIDAK DIKETAHUI"} 

# --- 2. DEFINISI ARSITEKTUR GIN ---
class GIN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GIN, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(num_node_features, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64))
        self.conv1 = GINConv(nn1)
        nn2 = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64))
        self.conv2 = GINConv(nn2)
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# --- 3. FUNGSI UNTUK MEMUAT DAN MEMPROSES DATA (Hanya dijalankan sekali) ---
def load_resources():
    """Memuat dataset dari CSV dan membangun objek graf PyTorch."""
    print("🚀 Memuat dataset dan membangun graph...")
    try:
        df_features = pd.read_csv(FEATURES_PATH)
        df_edgelist = pd.read_csv(EDGELIST_PATH)

        # Membuat mapping dari ID transaksi ke indeks integer
        tx_id_mapping = {tx_id: idx for idx, tx_id in enumerate(df_features['txID'])}
        
        # Memfilter dan memetakan edge list
        edges_filtered = df_edgelist[df_edgelist['txId1'].isin(tx_id_mapping.keys()) & df_edgelist['txId2'].isin(tx_id_mapping.keys())].copy()
        edges_filtered['Id1'] = edges_filtered['txId1'].map(tx_id_mapping)
        edges_filtered['Id2'] = edges_filtered['txId2'].map(tx_id_mapping)
        
        # Membuat tensor edge_index
        edge_index = torch.tensor(edges_filtered[['Id1', 'Id2']].values.T, dtype=torch.long)
        
        # Membuat tensor node_features
        # 'label' ditambahkan agar tidak dianggap sebagai fitur
        feature_cols = [col for col in df_features.columns if col not in ['txID', 'wallet_address', 'label']]
        node_features = torch.tensor(df_features[feature_cols].values, dtype=torch.float)
        
        # Membuat objek Data yang lengkap untuk model
        graph = Data(x=node_features, edge_index=edge_index)
        print("✅ Graph berhasil dibuat.")
        
        # Membuat mapping dari wallet_address ke node indices (PENTING untuk API)
        if 'wallet_address' not in df_features.columns:
            raise KeyError("Kolom 'wallet_address' tidak ditemukan di file fitur. API tidak bisa berfungsi.")
            
        wallet_map = df_features.groupby('wallet_address')['txID'].apply(list).to_dict()
        for wallet, tx_ids in wallet_map.items():
            wallet_map[wallet] = [tx_id_mapping.get(tx_id) for tx_id in tx_ids if tx_id_mapping.get(tx_id) is not None]
        print("✅ Mapping wallet berhasil dibuat.")

        return graph, wallet_map
    
    except FileNotFoundError as e:
        print(f"❌ Error: File tidak ditemukan! Pastikan '{FEATURES_PATH}' dan '{EDGELIST_PATH}' ada. {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error saat memproses data: {e}")
        return None, None

# --- Inisialisasi global saat server dimulai ---
graph_data, wallet_to_nodes_map = load_resources()

# Inisialisasi model
model = GIN(num_node_features=166, num_classes=3) 
try:
    if graph_data:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print("✅ Model GIN berhasil dimuat.")
except Exception as e:
    print(f"❌ Error saat memuat model: {e}")
    model = None

# --- 4. ENDPOINT API ---
@app.route('/predict/<string:wallet_address>', methods=['GET'])
def predict_wallet_status(wallet_address):
    """Endpoint untuk memprediksi status sebuah dompet."""
    if model is None or graph_data is None:
        return jsonify({"error": "Server tidak siap, model atau data gagal dimuat."}), 500

    if wallet_address not in wallet_to_nodes_map or not wallet_to_nodes_map[wallet_address]:
        return jsonify({"error": f"Alamat dompet '{wallet_address}' tidak ditemukan dalam dataset."}), 404

    node_indices = wallet_to_nodes_map[wallet_address]
    
    with torch.no_grad():
        all_preds_logits = model(graph_data)
        all_preds_classes = all_preds_logits.argmax(dim=1)

    wallet_predictions = all_preds_classes[node_indices]
    
    # Strategi: Jika ada 1 transaksi saja terdeteksi sebagai kelas 1 (BERBAHAYA)
    if 1 in wallet_predictions:
        status = CLASS_MAP[1]
    else:
        status = CLASS_MAP[0]

    return jsonify({
        "wallet_address": wallet_address,
        "predicted_status": status,
        "total_transactions_in_graph": len(node_indices)
    })

# --- 5. JALANKAN APLIKASI ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=False) # Set debug=False untuk produksi