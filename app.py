import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import os

# 1. Load Model, Scaler, dan Feature Columns yang sudah disimpan
model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("feature_columns.joblib")

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="💳", layout="wide")

st.title("💳 Sistem Pendeteksi Penipuan Kartu Kredit (Credit Card Fraud Detection)")
st.write("Aplikasi ini menggunakan algoritma **Random Forest** untuk mendeteksi apakah suatu transaksi kartu kredit berisiko penipuan atau tidak.")

# Sidebar untuk navigasi menu (Deployment vs Monitoring)
menu = st.sidebar.selectbox("Pilih Menu", ["Prediksi Transaksi (Deployment)", "Dashboard Monitoring"])

if menu == "Prediksi Transaksi (Deployment)":
    st.subheader("Form Input Data Transaksi")
    st.write("Masukkan detail transaksi di bawah ini untuk melakukan prediksi secara *real-time*:")

    # Membuat form input dinamis berdasarkan kolom fitur yang ada saat training
    # Anda dapat menyesuaikan input ini dengan fitur asli dataset Anda
    with st.form("fraud_form"):
        col1, col2 = st.columns(2)
        
        input_data = {}
        # Contoh input umum untuk transaksi (sesuaikan dengan fitur dataset Anda)
        with col1:
            input_data['Time'] = st.number_input("Time (Detik dari transaksi awal)", value=0.0)
            input_data['Amount'] = st.number_input("Transaction Amount ($)", min_value=0.0, value=50.0)
            
        with col2:
            st.write("Fitur V (V1 - Vn dari hasil PCA / Fitur Pendukung):")
            # Jika dataset Anda menggunakan V1 sampai V28 (seperti dataset standar Fraud Detection), 
            # Anda bisa membuat input sampel atau input utama. Di sini kita buat contoh interaktif untuk V1 dan V2:
            input_data['V1'] = st.number_input("V1", value=0.0)
            input_data['V2'] = st.number_input("V2", value=0.0)
            # Jika fitur kolom Anda banyak, pastikan semua masuk ke dictionary input_data.
            # Untuk amannya, kita isi sisa kolom fitur dengan nilai 0 jika belum ada di form:

        submitted = st.form_submit_button("Prediksi Sekarang")

    if submitted:
        # Menyiapkan DataFrame sesuai urutan kolom fitur saat training
        # Kita lengkapi semua kolom fitur dengan 0 terlebih dahulu, lalu timpa dengan input user
        full_input = {col: 0.0 for col in feature_columns}
        for key in input_data:
            if key in full_input:
                full_input[key] = input_data[key]
                
        df_input = pd.DataFrame([full_input])
        
        # Lakukan scaling data menggunakan scaler yang sudah disimpan
        df_scaled = pd.DataFrame(scaler.transform(df_input), columns=feature_columns)
        
        # Prediksi menggunakan model Random Forest
        prediction = model.predict(df_scaled)[0]
        prediction_proba = model.predict_proba(df_scaled)[0]
        
        fraud_probability = prediction_proba[1] * 100

        # Tampilkan Hasil Prediksi
        st.markdown("---")
        st.subheader("Hasil Analisis:")
        if prediction == 1:
            st.error(f"⚠️ **Peringatan! Transaksi Terindikasi PENIPUAN (Fraud)**\n\nTingkat Risiko: **{fraud_probability:.2f}%**")
        else:
            st.success(f"✅ **Transaksi Normal (Sah)**\n\nTingkat Risiko Fraud: **{fraud_probability:.2f}%**")

        # --- FITUR MONITORING: Simpan log transaksi ke file CSV setiap kali prediksi dilakukan ---
        log_entry = {
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Amount": input_data['Amount'],
            "Prediction": "Fraud" if prediction == 1 else "Normal",
            "Probability_Fraud": f"{fraud_probability:.2f}%"
        }
        
        log_file = "prediction_monitor_log.csv"
        df_log = pd.DataFrame([log_entry])
        
        if os.path.exists(log_file):
            df_existing = pd.read_csv(log_file)
            df_updated = pd.concat([df_existing, df_log], ignore_index=True)
            df_updated.to_csv(log_file, index=False)
        else:
            df_log.to_csv(log_file, index=False)

elif menu == "Dashboard Monitoring":
    st.subheader("📊 Dashboard Monitoring Sistem Prediksi")
    st.write("Bagian ini memantau riwayat transaksi yang telah diuji pada sistem (Logging & Monitoring).")
    
    log_file = "prediction_monitor_log.csv"
    if os.path.exists(log_file):
        df_logs = pd.read_csv(log_file)
        
        # Metrik Ringkasan (Monitoring Metrics)
        total_predictions = len(df_logs)
        total_fraud_detected = len(df_logs[df_logs["Prediction"] == "Fraud"])
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Total Transaksi Dipantau", total_predictions)
        col_m2.metric("Total Terdeteksi Fraud", total_fraud_detected, delta_color="inverse")
        
        st.markdown("---")
        st.write("### Riwayat Log Prediksi Terakhir:")
        st.dataframe(df_logs.tail(10)) # Menampilkan 10 riwayat terakhir
        
        # Tombol untuk mengunduh file log monitoring
        with open(log_file, "rb") as file:
            st.download_button(
                label="Unduh Log Monitoring (CSV)",
                data=file,
                file_name="prediction_monitor_log.csv",
                mime="text/csv"
            )
    else:
        st.info("Belum ada data monitoring. Silakan lakukan beberapa prediksi terlebih dahulu pada menu **Prediksi Transaksi**.")