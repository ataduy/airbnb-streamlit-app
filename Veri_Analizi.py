import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Sayfa ayarları

def veri_analizi():

    # 📁 Veri yolu
    data_path = "data/airbnb_temizlenmis.csv"

    # ✅ Dosya kontrolü
    if not os.path.exists(data_path):
        st.error("❌ Temizlenmiş veri bulunamadı.")
        st.stop()

    # 📥 Veri oku
    df = pd.read_csv(data_path)

    # 📌 Zipcode → Lokasyon ismi eşleştirmesi
    zip_to_location = {
        28806: "West Asheville, North Carolina, United States",
        28704: "Arden, North Carolina, United States",
        28801: "Asheville, North Carolina, United States",
        28715: "Candler, North Carolina, United States",
        28732: "Fletcher, North Carolina, United States",
        28803: "Asheville Biltmore, North Carolina, United States",
        28804: "North Asheville, North Carolina, United States",
        28805: "East Asheville, North Carolina, United States"
    }
    df["neighbourhood_cleansed"] = pd.to_numeric(df["neighbourhood_cleansed"], errors="coerce")
    df["location_name"] = df["neighbourhood_cleansed"].map(zip_to_location)

    # 🧹 Fiyat temizleme
    df["price_clean"] = (
        df["price"]
        .astype(str)
        .str.extract(r"(\$?\d+[\.,]?\d*)")[0]
        .str.replace(",", "")
        .str.replace("$", "")
        .astype(float)
    )

    # 🧼 accommodates segmentleme
    def categorize_accommodates(x):
        if x <= 2:
            return "1-2 Kişi"
        elif x <= 4:
            return "3-4 Kişi"
        elif x <= 6:
            return "5-6 Kişi"
        elif x <= 8:
            return "7-8 Kişi"
        else:
            return "9+ Kişi"
    df["accommodates_segment"] = df["accommodates"].apply(categorize_accommodates)

    # 📍 Lokasyon seçimi
    st.subheader("📍 Lokasyona Göre Filtreleme")
    selected_location = st.selectbox("Bir lokasyon seçin:", sorted(df["location_name"].dropna().unique()))
    filtered_df = df[df["location_name"] == selected_location]
    st.write(f"📌 {selected_location} için {len(filtered_df)} ilan bulundu.")

    # 📊 Temel metrikler
    avg_price = filtered_df["price_clean"].mean()
    avg_beds = filtered_df["beds"].mean()
    avg_reviews = filtered_df["number_of_reviews"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Ortalama Fiyat", f"${avg_price:.2f}")
    col2.metric("🛏️ Ortalama Yatak Sayısı", f"{avg_beds:.1f}")
    col3.metric("🗣️ Ortalama Yorum Sayısı", f"{avg_reviews:.0f}")

    # 🎨 Grafikler - 2'li düzende 6 grafik

    # 🎯 Grafik 1: Fiyat Dağılımı
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💵 Fiyat Dağılımı")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(filtered_df["price_clean"], bins=30, kde=True, ax=ax1)
        st.pyplot(fig1)

    # 🎯 Grafik 2: Oda Tipine Göre Ortalama Fiyat
    with col2:
        st.subheader("🏷️ Oda Tipine Göre Ortalama Fiyatlar")
        room_avg = filtered_df.groupby("room_type")["price_clean"].mean().sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=room_avg.values, y=room_avg.index, palette="viridis", ax=ax2)
        ax2.set_xlabel("Ortalama Fiyat ($)")
        ax2.set_ylabel("Oda Tipi")
        st.pyplot(fig2)

    # 🎯 Grafik 3: Kişi Sayısı vs Fiyat
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("👥 Kişi Sayısı vs Fiyat")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=filtered_df, x="accommodates", y="price_clean", hue="room_type", alpha=0.6, ax=ax3)
        st.pyplot(fig3)

    # 🎯 Grafik 4: Korelasyon Isı Haritası
    with col4:
        st.subheader("📈 En Yüksek Korelasyonlar")
        corr = df.select_dtypes(include="number").corr()["price_clean"].abs().sort_values(ascending=False)[1:21]
        top_corr_features = corr.index.tolist()
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[top_corr_features + ["price_clean"]].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax4)
        st.pyplot(fig4)

    # 🎯 Grafik 5: Özelliklerin Önemi
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("🧠 Modeldeki Özelliklerin Önemi")
        model_path = "model/price_model.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            feature_names = model.feature_names_in_
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            fig5, ax5 = plt.subplots(figsize=(6, 4))
            sns.barplot(x="Importance", y="Feature", data=importance_df.head(20), palette="crest", ax=ax5)
            st.pyplot(fig5)
        else:
            st.warning("🎯 Model dosyası bulunamadı. Önce modeli eğitmelisiniz.")

    # 🎯 Grafik 6: Segmentlere Göre Ortalama Fiyat (ekstra)
    with col6:
        st.subheader("👪 Kişi Segmentine Göre Ortalama Fiyat")
        segment_avg = filtered_df.groupby("accommodates_segment")["price_clean"].mean().sort_values()
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=segment_avg.index, y=segment_avg.values, palette="flare", ax=ax6)
        ax6.set_xlabel("Kişi Segmenti")
        ax6.set_ylabel("Ortalama Fiyat ($)")
        st.pyplot(fig6)
