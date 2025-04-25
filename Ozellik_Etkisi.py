import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Sayfa başlığı

def ozelliketkisi():

    # CSV yolu
    csv_path = "model/feature_importance.csv"


    # Dosya kontrolü
    if not os.path.exists(csv_path):
        st.error("❌ Özellik skorları bulunamadı. Lütfen önce train_model.py dosyasını çalıştırın.")
        st.stop()

    # CSV oku
    df = pd.read_csv(csv_path)

    # Sidebar ayarları
    #st.sidebar.header("🎛️ Görselleştirme Ayarları")
    top_n = st.sidebar.slider("Kaç özellik gösterilsin?", min_value=5, max_value=30, value=20)

    # İlk tablo
    st.subheader("📄 İlk Özellikler")
    st.dataframe(df.head(top_n))

    # ➤ 1. Satır - Interaktif Bar & Kümülatif Etki
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Interaktif Bar (Plotly)")
        fig1 = px.bar(
            df.head(top_n).sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
            labels={"importance": "Önem Skoru", "feature": "Özellik"},
            title=f"En Önemli {top_n} Özellik"
        )
        fig1.update_layout(yaxis=dict(categoryorder='total ascending'))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("### 📊 Kümülatif Önem (Line Plot)")
        df_sorted = df.sort_values("importance", ascending=False).reset_index(drop=True)
        df_sorted["cumulative_importance"] = df_sorted["importance"].cumsum()
        fig2, ax2 = plt.subplots()
        sns.lineplot(data=df_sorted.head(top_n), x=df_sorted.index[:top_n], y="cumulative_importance", marker="o", ax=ax2)
        ax2.set_xlabel("Özellik Sırası")
        ax2.set_ylabel("Kümülatif Önem")
        ax2.set_title("Kümülatif Özellik Katkısı")
        st.pyplot(fig2)

    # ➤ 2. Satır - Kategorilere göre dağılım & Önem yoğunluğu
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 🔍 Özellik Türüne Göre Ortalama Önem")
        df["group"] = df["feature"].apply(lambda x: x.split("_")[0])
        group_avg = df.groupby("group")["importance"].mean().sort_values(ascending=False)
        fig3, ax3 = plt.subplots()
        sns.barplot(x=group_avg.values, y=group_avg.index, palette="viridis", ax=ax3)
        ax3.set_title("Özellik Gruplarının Ortalama Önemi")
        ax3.set_xlabel("Ortalama Önem")
        ax3.set_ylabel("Özellik Grubu")
        st.pyplot(fig3)

    with col4:
        st.markdown("### 🌡️ Önem Yoğunluk Grafiği")
        fig4, ax4 = plt.subplots()
        sns.kdeplot(df["importance"], fill=True, color="skyblue", ax=ax4)
        ax4.set_title("Özellik Önem Dağılımı")
        ax4.set_xlabel("Önem Skoru Yoğunluğu")
        st.pyplot(fig4)

    # Açıklama
    st.info("""
    Bu panelde modelin fiyat tahmininde hangi özelliklere ne kadar önem verdiğini inceleyebilirsin. 
    Interaktif grafikler, kümülatif analizler ve dağılım görselleri ile detaylı içgörüler sunulmuştur.
    """)
