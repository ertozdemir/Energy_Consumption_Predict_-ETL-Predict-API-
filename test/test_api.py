import requests
import json

# API'nin çalıştığı adres (Varsayılan olarak uvicorn 8000 portunu kullanır)
URL = "http://127.0.0.1:8000/predict"

# 1. Test verisi oluşturuyoruz
# Not: building_type ve day_of_week modelin bildiği kategorilerden olmalıdır (veya OneHotEncoder handle_unknown='ignore' ile çalışır)
data = {
    "building_type": "Commercial",
    "square_meters": 250.5,
    "number_occupants": 12,
    "appliances_used": 8,
    "day_of_week": "Monday"
}

print(f"API'ye gönderilen veri: {data}")

# 2. POST isteği gönderiyoruz
try:
    response = requests.post(URL, json=data)
    
    # Başarı durumunu kontrol ediyoruz
    if response.status_code == 200:
        result = response.json()
        print("\n--- Başarılı Tahmin ---")
        print(f"Tahmin Edilen Enerji Tüketimi: {result['predicted_energy_consumption']:.2f}")
    else:
        print(f"\nHata Oluştu! Durum Kodu: {response.status_code}")
        print(f"Hata Mesajı: {response.text}")

except requests.exceptions.ConnectionError:
    print("\nHATA: API sunucusuna bağlanılamadı. Lütfen 'uvicorn main:app --reload' komutuyla API'nin çalıştığından emin olun.")
