import requests

# Ganti URL ini dengan alamat Railway (atau localhost jika testing lokal)
API_URL = "http://0.0.0.0:5001/verify"

# Ganti dengan account_number yang ingin diuji
payload = {
    "account_number": 6889238680  # <- ganti sesuai keperluan
}

try:
    response = requests.post(API_URL, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:", response.json())

except Exception as e:
    print("Request failed:", e)
