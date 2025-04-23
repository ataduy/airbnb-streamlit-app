import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Sayfa başlığı
st.set_page_config(page_title="Airbnb Insight Dashboard", layout="wide")
st.title("🧠 Airbnb Veri Analizi & İçgörü Paneli")

# 📁 CSV veri yolu
data_path = "/Users/atakandogulu/Desktop/airbnb_streamlit_app/data/airbnb_temizlenmis.csv"

# Dosya kontrolü
if not os.path.exists(data_path):
    st.error("❌ CSV dosyası bulunamadı.")
    st.stop()

# Veri yükleme
df = pd.read_csv(data_path)

# Fiyat temizliği (eğer yoksa)
if "price_clean" not in df.columns:
    df["price_clean"] = (
        df["price"]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .astype(float)
    )

# Sidebar filtreler
st.sidebar.header("🔍 Filtreler")
max_price = st.sidebar.slider("💰 Maksimum Fiyat ($)", min_value=10, max_value=1000, value=500)
selected_room_types = st.sidebar.multiselect(
    "🏷️ Oda Tipi Seçimi",
    options=sorted(df["room_type"].dropna().unique()),
    default=list(df["room_type"].dropna().unique())
)

# Filtre uygulanmış veri
filtered_df = df[
    (df["price_clean"] <= max_price) &
    (df["room_type"].isin(selected_room_types))
]

# 🎨 Grid yapısı: 2 sütunlu layout
st.markdown("## 📊 Grafikler")

# --- Grafik 1 ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🏠 Oda Tipine Göre Ortalama Fiyat")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=filtered_df, x="room_type", y="price_clean", ax=ax1, palette="Set3")
    ax1.set_ylabel("Fiyat ($)")
    ax1.set_xlabel("Oda Tipi")
    ax1.set_title("Fiyat Dağılımı")
    st.pyplot(fig1)

# --- Grafik 2 ---
with col2:
    st.markdown("#### 🗺️ Bölge ve Oda Tipi Kombinasyonu")
    heatmap_data = filtered_df.pivot_table(
        index='neighbourhood_cleansed',
        columns='room_type',
        values='price_clean',
        aggfunc='mean'
    )
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="magma", linewidths=0.5, cbar=False, ax=ax2)
    ax2.set_title("Ortalama Fiyat (Bölge & Oda Tipi)")
    st.pyplot(fig2)

# --- Grafik 3 ---
col3, col4 = st.columns(2)

with col3:
    st.markdown("#### 📈 Mahalle Bazlı Değer Skoru")
    score_df = df.groupby("neighbourhood")[["price_clean", "number_of_reviews"]].mean().dropna()
    score_df["value_score"] = score_df["number_of_reviews"] / score_df["price_clean"]
    top_scores = score_df.sort_values("value_score", ascending=False).head(10)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=top_scores.index, y=top_scores["value_score"], ax=ax3, palette="cool")
    ax3.set_ylabel("Yorum / Fiyat Skoru")
    ax3.set_xlabel("Mahalle")
    ax3.set_title("En Değerli 10 Mahalle")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

# --- Grafik 4: Oda Tipine Göre Ortalama Yorum Sayısı ---
with col4:
    st.markdown("#### 💬 Oda Tipine Göre Ortalama Yorum Sayısı")
    review_avg = filtered_df.groupby("room_type")["number_of_reviews"].mean().sort_values(ascending=False)

    fig4, ax4 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=review_avg.index, y=review_avg.values, ax=ax4, palette="Blues_r")
    ax4.set_ylabel("Ortalama Yorum Sayısı")
    ax4.set_xlabel("Oda Tipi")
    ax4.set_title("Oda Tipine Göre Yorum Ortalaması")
    st.pyplot(fig4)

# ✅ Tamamlandı
st.success("Tüm grafikler düzenli şekilde yüklendi. Sidebar'dan filtrelemeye devam edebilirsin! ✨")
