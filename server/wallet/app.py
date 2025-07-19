import os
import pandas as pd
import torch
import torch.nn.functional as F
from flask import Flask, jsonify
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
import gdown

app = Flask(__name__)

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

FEATURES_DRIVE_ID = "1YIwRjMfmwFc-xtmeyb5Nxq2k3bLVUFV2"
EDGELIST_DRIVE_ID = "1iKsZStdmIG7BQWMGXpLvYSpL42GAR8zL"

FEATURES_PATH = "data/dummy_elliptic_txs_features1.csv"
EDGELIST_PATH = "data/elliptic_txs_edgelist.csv"
MODEL_PATH = "models/gin_model.pth"

def download_from_drive(drive_id, output_path):
    if not os.path.exists(output_path):
        print(f"📥 Mengunduh {output_path} dari Google Drive...")
        gdown.download(id=drive_id, output=output_path, quiet=False)
        print(f"✅ File disimpan di {output_path}")

download_from_drive(FEATURES_DRIVE_ID, FEATURES_PATH)
download_from_drive(EDGELIST_DRIVE_ID, EDGELIST_PATH)

class GIN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GIN, self).__init__()
        nn1 = Sequential(Linear(num_node_features, 64), ReLU(), Linear(64, 64))
        self.conv1 = GINConv(nn1)
        nn2 = Sequential(Linear(64, 64), ReLU(), Linear(64, 64))
        self.conv2 = GINConv(nn2)
        self.fc = Linear(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def load_resources():
    try:
        df_features = pd.read_csv(FEATURES_PATH)
        df_edgelist = pd.read_csv(EDGELIST_PATH)

        tx_id_mapping = {tx_id: idx for idx, tx_id in enumerate(df_features['txID'])}
        edges_filtered = df_edgelist[
            df_edgelist['txId1'].isin(tx_id_mapping) & df_edgelist['txId2'].isin(tx_id_mapping)
        ].copy()
        edges_filtered['Id1'] = edges_filtered['txId1'].map(tx_id_mapping)
        edges_filtered['Id2'] = edges_filtered['txId2'].map(tx_id_mapping)

        edge_index = torch.tensor(edges_filtered[['Id1', 'Id2']].values.T, dtype=torch.long)
        feature_cols = [col for col in df_features.columns if col not in ['txID', 'wallet_address', 'label']]
        node_features = torch.tensor(df_features[feature_cols].values, dtype=torch.float)

        graph = Data(x=node_features, edge_index=edge_index)

        wallet_map = df_features.groupby('wallet_address')['txID'].apply(list).to_dict()
        for wallet, tx_ids in wallet_map.items():
            wallet_map[wallet] = [tx_id_mapping.get(tx_id) for tx_id in tx_ids if tx_id_mapping.get(tx_id) is not None]

        return graph, wallet_map
    except Exception as e:
        print(f"Error saat memproses data: {e}")
        return None, None

graph_data, wallet_to_nodes_map = load_resources()

model = GIN(num_node_features=166, num_classes=3)
try:
    if graph_data:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print("Model berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None

CLASS_MAP = {0: "AMAN", 1: "BERBAHAYA", 2: "TIDAK DIKETAHUI"}

@app.route('/predict/<string:wallet_address>', methods=['POST'])
def predict_wallet_status(wallet_address):
    if model is None or graph_data is None:
        return jsonify({"error": "Server tidak siap, model atau data gagal dimuat."}), 500

    if wallet_address not in wallet_to_nodes_map or not wallet_to_nodes_map[wallet_address]:
        return jsonify({"error": f"Alamat dompet '{wallet_address}' tidak ditemukan."}), 404

    node_indices = wallet_to_nodes_map[wallet_address]

    with torch.no_grad():
        logits = model(graph_data)
        predictions = logits.argmax(dim=1)

    wallet_predictions = predictions[node_indices]

    if 1 in wallet_predictions:
        status = CLASS_MAP[1]
    else:
        status = CLASS_MAP[0]

    return jsonify({
        "wallet_address": wallet_address,
        "predicted_status": status,
        "total_transactions": len(node_indices)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=False)
