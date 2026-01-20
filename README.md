# Energy Consumption Prediction Pipeline

Farklı bina türleri ve çevresel faktörlere dayalı olarak enerji tüketimini tahmin eden uçtan uca bir makine öğrenmesi projesi.

Bu proje, veri işleme (ETL), model eğitimi ve canlı tahmin sunan bir REST API (FastAPI) içerir. Geliştirme sürecinde gerçek hayat senaryosuna uygun olarak veriler SQLite veritabanı üzerinden akar ve tahminler loglanır.

## Özellikler

- **Veri Pipeline'ı:** Ham CSV verilerinin işlenip veritabanına aktarılması.
- **Model Eğitimi:** Linear Regression, Random Forest ve XGBoost modellerinin karşılaştırmalı eğitimi. `RandomizedSearchCV` ile hiperparametre optimizasyonu.
- **API Servisi:** FastAPI ile oluşturulmuş, eğitilen modeli servis eden ve basit bir ön yüz (HTML/JS) sunan web uygulaması.
- **Loglama:** Yapılan her tahmin isteğinin veritabanına kaydedilmesi.

## Kurulum

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

### 1. Gereksinimler

Python 3.8+ önerilir. Gerekli paketleri yüklemek için:

```bash
pip install -r requirements.txt
```

### 2. Veri Hazırlığı (ETL)

Veri setlerini veritabanına (`database/energy_consumption.db`) aktarmak için pipeline scriptini çalıştırın:

```bash
python src/pipeline.py
```

### 3. Model Eğitimi

Modelleri eğitmek ve `models/` klasörüne kaydetmek için:

```bash
python src/model.py
```
Bu işlem sonunda en iyi performansı gösteren modeller `.pkl` formatında kaydedilecektir.

### 4. Uygulamayı Başlatma

API sunucusunu başlatmak için:

```bash
uvicorn src.main:app --reload
```

Tarayıcınızda [http://127.0.0.1:8000](http://127.0.0.1:8000) adresine giderek arayüzü görebilir ve test edebilirsiniz.

## Proje Yapısı

```
mlpipeline/
├── data/          # Ham veri setleri (CSV)
├── database/      # SQLite veritabanı
├── models/        # Eğitilmiş model dosyaları (.pkl)
├── src/           # Kaynak kodlar
│   ├── main.py    # FastAPI uygulaması
│   ├── model.py   # Model eğitimi scripti
│   └── pipeline.py # Veri işleme scripti
├── test/          # Test scriptleri
└── index.html     # Web arayüzü
```

## Kullanılan Teknolojiler

- **Python**: Ana programlama dili.
- **FastAPI**: Web API framework'ü.
- **Pandas & Scikit-learn**: Veri manipülasyonu ve modelleme.
- **XGBoost**: Gradient boosting kütüphanesi.
- **SQLite**: Veri saklama.

---
Herhangi bir hata veya öneri için issue açabilir veya pull request gönderebilirsiniz.
