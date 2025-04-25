import streamlit as st
from Ozellik_Etkisi import ozelliketkisi
from Veri_Analizi import veri_analizi
from Veri_Insightları import veri_Insightlar
from Fiyat_Tahmini import fiyattahmini


st.set_page_config(page_title="Airbnb Dashboard", layout="wide")


st.sidebar.title("📊 Airbnb Dash")
page = st.sidebar.radio("Menü", [
    "Veri Analizi",
    "Veri Insights",
    "Fiyat Tahmini",
    "Özellik Etkisi"
])

if page == "Veri Analizi":
    st.title("📊 Airbnb Verisi Keşfi ve İstatistiksel Analiz")
    veri_analizi()



elif page == "Veri Insights":
    st.title("🧠 Airbnb Veri Analizi & İçgörü Paneli")
    veri_Insightlar()


elif page == "Fiyat Tahmini":
    st.title("💸 Airbnb Fiyat Tahmini Aracı")
    fiyattahmini()


elif page == "Özellik Etkisi":
    st.title("📊 Özelliklerin Fiyata Etkisi (Detaylı Görsel Analiz)")
    ozelliketkisi()