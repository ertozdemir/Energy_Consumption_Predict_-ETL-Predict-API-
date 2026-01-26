from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import sqlite3
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. FastAPI uygulamasını başlatıyoruz
app = FastAPI(title="Enerji Tüketim Tahmin API")

# Anasayfa için statik dosyayı sunmak üzere (index.html)
# Bu satır, request geldiğinde "/" yolunda özelleştirilmiş bir işlem yapmamızı kolaylaştırır,
# ancak basitlik adına FileResponse kullanacağız.

# 2. Model dosyasının yolunu belirliyoruz ve yüklüyoruz
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost.pkl")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Model {MODEL_PATH} başarıyla yüklendi.")
else:
    print(f"HATA: {MODEL_PATH} bulunamadı!")
    model = None

# 3. Veritabanı Fonksiyonları
# Kullanıcıdan gelen her isteği ve sonucu kaydetmek için
def init_db():
    db_folder = os.path.join(BASE_DIR, 'database')
    os.makedirs(db_folder, exist_ok=True)
    conn = sqlite3.connect(os.path.join(db_folder, 'energy_consumption.db'))
    c = conn.cursor()
    # 'user' tablosu yoksa oluşturuyoruz
    c.execute('''CREATE TABLE IF NOT EXISTS user 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp DATETIME,
                  building_type TEXT, 
                  square_meters REAL, 
                  number_occupants INTEGER, 
                  appliances_used INTEGER, 
                  day_of_week TEXT, 
                  predicted_consumption REAL)''')
    conn.commit()
    conn.close()

# Uygulama başladığında DB'yi hazırla
init_db()

def log_to_db(data, prediction):
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'database', 'energy_consumption.db'))
    c = conn.cursor()
    c.execute("INSERT INTO user (timestamp, building_type, square_meters, number_occupants, appliances_used, day_of_week, predicted_consumption) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (datetime.now(), data.building_type, data.square_meters, data.number_occupants, data.appliances_used, data.day_of_week, prediction))
    conn.commit()
    conn.close()

# 4. Giriş verisi modeli (Validation)
class EnergyInput(BaseModel):
    building_type: str
    square_meters: float
    number_occupants: int
    appliances_used: int
    day_of_week: str

# 5. Endpointler

# Anasayfada HTML dosyasını göster
@app.get("/")
def read_root():
    return FileResponse(os.path.join(BASE_DIR, 'index.html'))

@app.post("/predict")
def predict(data: EnergyInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model yüklü değil.")

    try:
        # Veriyi DataFrame'e çevirip tahmin al
        input_df = pd.DataFrame([data.model_dump()])
        prediction = float(model.predict(input_df)[0])
        
        # Sonucu veritabanına kaydet
        log_to_db(data, prediction)
        
        return {
            "input": data,
            "predicted_energy_consumption": prediction
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
