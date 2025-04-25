import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Sayfa başlığı

def fiyattahmini():

    # 🎯 Model yükle
    model_path = "model/price_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model bulunamadı. Lütfen önce train_model.py dosyasını çalıştırın.")
        st.stop()

    model = joblib.load(model_path)
    feature_names = model.feature_names_in_

    # Kullanıcıdan giriş al
    st.subheader("📥 Bilgilerinizi Girin")

    col1, col2 = st.columns(2)

    with col1:
        accommodates = st.slider("👥 Kaç kişilik konaklama?", min_value=1, max_value=16, value=2)
        beds = st.slider("🛏️ Yatak Sayısı", min_value=0, max_value=10, value=1)
        bedrooms = st.slider("🚪 Yatak Odası Sayısı", min_value=0, max_value=10, value=1)
        bathrooms = st.slider("🛁 Banyo Sayısı", min_value=0.0, max_value=5.0, step=0.5, value=1.0)

    with col2:
        availability_365 = st.slider("📅 Yıllık Müsaitlik (gün)", 0, 365, 180)
        min_nights_avg = st.slider("📉 Ortalama Minimum Gece Sayısı", 1, 30, 2)
        max_nights_avg = st.slider("📈 Ortalama Maksimum Gece Sayısı", 1, 365, 90)

    # Bölge ve oda tipi seçimleri
    room_type = st.selectbox("🏠 Oda Tipi", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
    location_name = st.selectbox("📍 Lokasyon", [
        "West Asheville, North Carolina, United States",
        "Arden, North Carolina, United States",
        "Asheville, North Carolina, United States",
        "Candler, North Carolina, United States",
        "Fletcher, North Carolina, United States",
        "Asheville Biltmore, North Carolina, United States",
        "North Asheville, North Carolina, United States",
        "East Asheville, North Carolina, United States"
    ])

    # Özellik mühendisliği
    price_per_accommodate = 100  # Tahmin yapıldığı için bilinmiyor
    price_per_bed = 80  # Aynı şekilde bilinmiyor

    # Özellik sözlüğü oluştur
    input_data = {
        "accommodates": accommodates,
        "beds": beds,
        "bedrooms": bedrooms,
        "bathrooms_numeric": bathrooms,
        "availability_365": availability_365,
        "minimum_nights_avg_ntm": min_nights_avg,
        "maximum_nights_avg_ntm": max_nights_avg,
        "price_per_accommodate": price_per_accommodate,
        "price_per_bed": price_per_bed
    }

    # One-hot encoded alanlar
    for col in feature_names:
        if col.startswith("room_type_"):
            input_data[col] = 1 if col == f"room_type_{room_type}" else 0
        elif col.startswith("location_name_"):
            input_data[col] = 1 if col == f"location_name_{location_name}" else 0

    # Eksik kalan tüm feature'ları sıfırla
    for col in feature_names:
        if col not in input_data:
            input_data[col] = 0

    # Tahmin
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]  # Özellik sırasını modelle aynı yap
    predicted_price = model.predict(input_df)[0]

    # Sonuç
    st.subheader("📢 Tahmin Edilen Fiyat")
    st.success(f"💰 ${predicted_price:.2f} USD")

    st.caption("Bu tahmin, sağladığınız bilgilere göre XGBoost regresyon modeli tarafından hesaplanmıştır.")
