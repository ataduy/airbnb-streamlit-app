# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os

# 📁 CSV dosyasını oku
#file_path = "/Users/atakandogulu/Desktop/airbnb_streamlit_app/data/airbnb_temizlenmis.csv"
file_path = "data/airbnb_temizlenmis.csv"
df = pd.read_csv(file_path)

# 🎯 Hedef değişken
TARGET = "price_clean"

# ✅ Yeni location bilgisi (neighbourhood üzerinden)
df["location_name"] = df["neighbourhood"].fillna("Other")

# ✅ Kategorik kolonları one-hot encode et
df = pd.get_dummies(df, columns=["room_type", "location_name"], drop_first=False)

# ✅ Özellik mühendisliği
df["price_per_accommodate"] = df["price_clean"] / df["accommodates"].replace(0, np.nan)
df["price_per_bed"] = df["price_clean"] / df["beds"].replace(0, np.nan)

# ✅ segmented_amenities encode et
if "segmented_amenities" in df.columns:
    df = pd.get_dummies(df, columns=["segmented_amenities"], drop_first=False)

# 🔧 Özellik listesi
feature_columns = [
    "accommodates", "beds", "bedrooms", "bathrooms_numeric",
    "availability_365", "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
    "price_per_accommodate", "price_per_bed"
] + [col for col in df.columns if col.startswith("room_type_") or col.startswith("location_name_") or col.startswith("segmented_amenities_")]

# ❌ Eksik verileri çıkar
df = df.dropna(subset=feature_columns + [TARGET])

# 🎯 Model verilerini ayır
X = df[feature_columns]
y = df[TARGET]

# 🧪 Eğitim ve test kümeleri
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Modeli eğit
model = XGBRegressor(random_state=42, n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# 🔮 Tahmin yap
y_pred = model.predict(X_test)

# 📉 Değerlendirme
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📉 Mean Squared Error (MSE): {mse:.2f}")
print(f"📈 R2 Score: {r2:.2f}")

# 💾 Modeli kaydet
model_path = "model/price_model.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print(f"✅ Model başarıyla kaydedildi: {model_path}")

# 📊 Özellik önemlerini CSV olarak kaydet
importance_df = pd.DataFrame({
    "feature": model.feature_names_in_,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

feature_csv_path = "model/feature_importance.csv"
importance_df.to_csv(feature_csv_path, index=False)
print(f"📁 Özellik önemleri CSV olarak kaydedildi: {feature_csv_path}")
