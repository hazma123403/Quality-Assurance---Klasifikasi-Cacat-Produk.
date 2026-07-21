import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Quality Assurance - Klasifikasi Cacat",
    page_icon="🔍",
    layout="wide"
)

# Load Dataset QC
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 400
    ketebalan = np.random.normal(5.0, 0.5, n)
    presisi = np.random.normal(0.02, 0.01, n)
    suhu = np.random.normal(200, 15, n)
    
    # Aturan Pelabelan Cacat Produk
    kategori = []
    for i in range(n):
        if suhu[i] > 220:
            kategori.append("Crack (Retak)")
        elif ketebalan[i] < 4.2:
            kategori.append("Scratch (Gores)")
        elif presisi[i] > 0.035:
            kategori.append("Dent (Penyok)")
        else:
            kategori.append("Lolos QC (Pass)")
            
    df = pd.DataFrame({
        'Ketebalan_mm': np.round(ketebalan, 2),
        'Deviasi_Presisi_mm': np.round(presisi, 3),
        'Suhu_Proses_C': np.round(suhu, 1),
        'Kategori_Cacat': kategori
    })
    return df

df = load_data()

# Latih Model Random Forest Classifier
X = df[['Ketebalan_mm', 'Deviasi_Presisi_mm', 'Suhu_Proses_C']]
y = df['Kategori_Cacat']
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Judul Utama
st.title("🔍 Quality Assurance: Klasifikasi Jenis Cacat Produk")
st.markdown("Aplikasi web ini menggunakan algoritma **Random Forest Classifier** untuk memprediksi tipe cacat fisik produk manufaktur berdasarkan parameter proses produksi.")

# Navigation Sidebar
st.sidebar.header("⚙️ Navigasi Aplikasi")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["Dashboard & Distribusi QC", "Simulasi Tes QC Real-time", "Interpretasi & Strategi Mutu"]
)

# Halaman 1: Dashboard Utama
if menu == "Dashboard & Distribusi QC":
    st.header("📊 Ringkasan Hasil Inspeksi QC")
    
    col1, col2, col3, col4 = st.columns(4)
    total_batch = len(df)
    pass_count = len(df[df['Kategori_Cacat'] == 'Lolos QC (Pass)'])
    defect_count = total_batch - pass_count
    pass_rate = round((pass_count / total_batch) * 100, 1)
    
    col1.metric("Total Sampel Inspeksi", total_batch)
    col2.metric("Lolos QC (Pass)", pass_count)
    col3.metric("Produk Cacat", defect_count, delta="-Reject", delta_color="inverse")
    col4.metric("Pass Rate", f"{pass_rate}%")
    
    st.markdown("---")
    
    st.subheader("Distribusi Kategori Cacat Produk")
    fig = px.histogram(
        df, 
        x="Kategori_Cacat", 
        color="Kategori_Cacat",
        title="Jumlah Hasil Inspeksi Menurut Kategori",
        text_auto=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Pratinjau Data QC")
    st.dataframe(df.head(10), use_container_width=True)

# Halaman 2: Form Simulasi Prediksi QC
elif menu == "Simulasi Tes QC Real-time":
    st.header("🧪 Form Pengujian Kualitas Produk")
    st.write("Masukkan nilai parameter uji laboratorium untuk memprediksi status kelayakan produk:")
    
    with st.form("qc_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            val_k = st.number_input("Ketebalan Material (mm)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
        with col2:
            val_p = st.number_input("Deviasi Presisi (mm)", min_value=0.000, max_value=0.100, value=0.020, step=0.005, format="%.3f")
        with col3:
            val_s = st.number_input("Suhu Proses Pembentukan (°C)", min_value=100.0, max_value=300.0, value=200.0, step=1.0)
            
        submit_btn = st.form_submit_button("Jalankan Uji QC")
        
    if submit_btn:
        input_data = np.array([[val_k, val_p, val_s]])
        prediksi = rf_model.predict(input_data)[0]
        
        st.markdown("---")
        st.subheader("Hasil Analisis Mutu:")
        
        if prediksi == "Lolos QC (Pass)":
            st.success(f"✅ **HASIL: {prediksi.upper()}** — Produk memenuhi standar kualitas dan siap dikemas.")
        elif "Crack" in prediksi:
            st.error(f"❌ **HASIL: {prediksi.upper()}** — Ditemukan retakan material!")
        elif "Scratch" in prediksi:
            st.warning(f"⚠️ **HASIL: {prediksi.upper()}** — Ditemukan goresan permukaan material!")
        else:
            st.warning(f"⚠️ **HASIL: {prediksi.upper()}** — Ditemukan penyok pada struktur material!")

# Halaman 3: Insights Bisnis
elif menu == "Interpretasi & Strategi Mutu":
    st.header("💡 Strategi Pengendalian Mutu (Quality Control)")
    
    st.markdown("""
    ### 📌 Analisis Akar Masalah (Root Cause Analysis):
    
    * **1. Cacat Retak (Crack):**
      * **Penyebab:** Suhu pencetakan/proses terlalu tinggi (>220°C) yang menyebabkan struktur material menjadi rapuh saat pendinginan.
      * **Tindakan:** Turunkan suhu *heater* mesin dan kalibrasi termokopel secara berkala.
      
    * **2. Cacat Gores (Scratch):**
      * **Penyebab:** Bahan mentah yang masuk terlalu tipis (<4.2 mm) sehingga mudah tergores oleh roller konveyor.
      * **Tindakan:** Lakukan evaluasi pada pemasok (*supplier*) bahan baku mentah.
      
    * **3. Cacat Penyok (Dent):**
      * **Penyebab:** Deviasi presisi mesin pencetak terlalu longgar (>0.035 mm).
      * **Tindakan:** Kencangkan setelan presisi dies/cetakan pada mesin.
    """)
