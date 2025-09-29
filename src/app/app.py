import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Harga Rumah Kab. Tangerang", layout="wide")

# Tab Navigasi
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "⚠️ Disclaimer", "📊 House Price Prediction", "🔎 House Specs Prediction"])

# Tab Home
with tab1:
    st.title("🏠 Selamat Datang di Aplikasi Prediksi Harga Rumah di Kabupaten Tangerang")
    st.write("""
    Aplikasi ini menggunakan model Machine Learning (XGBoost + Optuna) untuk memprediksi harga rumah di Kabupaten Tangerang berdasarkan spesifikasi rumah.
    
    **Cara Menggunakan:**
    1. Buka tab **📊 House Price Prediction**.
    2. Isi semua input yang tersedia.
    3. Klik tombol **Prediksi Harga**.
    4. Harga estimasi akan ditampilkan dalam bentuk rupiah.
             
    *Notes: Buka tab Disclaimer untuk membaca beberapa ketentuan dan batasan terkait estimasi harga rumah yang diberikan oleh aplikasi ini.
    
    🚀 **Mulai sekarang dengan membuka tab Prediksi!**
    """)

# Tab Disclaimer
with tab2:
    st.title("⚠️ Disclaimer")
    st.markdown("""
    **Harap Perhatian** 🏠🔍  

    Website ini menyediakan estimasi harga rumah berdasarkan model prediktif yang dikembangkan menggunakan teknik machine learning.  

    ---
    
    ⚠️ **Hanya Sebagai Referensi**  
    Prediksi yang diberikan oleh sistem ini hanya bersifat estimasi dan **tidak dapat dijadikan acuan pasti** dalam transaksi jual beli properti.  

    📊 **Ketergantungan pada Data**  
    Akurasi prediksi bergantung pada data yang digunakan dalam pelatihan model.

    🚫 **Tidak Menjamin Akurasi**  
    Meskipun model telah dioptimalkan, hasil prediksi bisa berbeda dengan harga pasar sebenarnya.  

    💡 **Bukan Saran Keuangan atau Properti**  
    Website ini **bukan merupakan saran investasi**. Disarankan untuk **berkonsultasi dengan agen properti** atau profesional terkait sebelum mengambil keputusan.  

    🔐 **Privasi Data**  
    Website ini **tidak menyimpan atau membagikan data** yang diinput pengguna tanpa izin.  

    ---
    
    Dengan menggunakan layanan ini, pengguna **menyetujui bahwa pengembang website tidak bertanggung jawab atas keputusan yang dibuat berdasarkan hasil prediksi**.
    """, unsafe_allow_html=True)

# Tab Prediksi
with tab3:
    st.title("📊 Prediksi Harga Rumah")

    # Layout dengan 2 kolom untuk input user
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏡 Informasi Umum")
        kamar_tidur = st.number_input("Jumlah Kamar Tidur:", min_value=1, max_value=10, value=3)
        kamar_mandi = st.number_input("Jumlah Kamar Mandi:", min_value=1, max_value=10, value=2)
        luas_tanah = st.number_input("Luas Tanah (m²):", min_value=10, max_value=1000, value=100)
        luas_bangunan = st.number_input("Luas Bangunan (m²):", min_value=10, max_value=1000, value=80)
        daya_listrik = st.number_input("Daya Listrik (Watt):", min_value=450, max_value=6600, value=1300)

    with col2:
        st.subheader("⚡ Spesifikasi Rumah")
        jumlah_lantai = st.number_input("Jumlah Lantai:", min_value=1, max_value=5, value=1)
        carport = st.number_input("Carport (Jumlah Mobil):", min_value=0, max_value=5, value=1)
        kamar_tidur_pembantu = st.number_input("Kamar Tidur Pembantu:", min_value=0, max_value=5, value=0)
        kamar_mandi_pembantu = st.number_input("Kamar Mandi Pembantu:", min_value=0, max_value=5, value=0)

    # Pilihan Kecamatan
    # Kecamatan yang tersedia dalam model (harus sesuai dengan saat model dilatih)
    kecamatan_tersedia = [
        'Balaraja', 'Cikupa', 'Cisauk', 'Curug', 'Jatiuwung', 'Jayanti', 'Kadu',
        'Kelapa Dua', 'Kosambi', 'Kresek', 'Legok', 'Mauk', 'Pagedangan',
        'Panongan', 'Pasar Kemis', 'Rajeg', 'Sepatan', 'Sindang Jaya', 'Solear',
        'Teluk Naga', 'Tigaraksa'
    ]

    # Kecamatan yang tidak memiliki data harga rumah
    kecamatan_tidak_tersedia = [
        'Gunung Kaler', 'Jambe', 'Kemiri', 'Mekar Baru', 
        'Pakuhaji', 'Sepatan Timur', 'Sukadiri', 'Sukamulya'
    ]

    # Pilih kecamatan berdasarkan daftar yang tersedia
    selected_kecamatan = st.selectbox("🏙️ Pilih Kecamatan:", kecamatan_tersedia + kecamatan_tidak_tersedia)

    # Tombol Prediksi di Tengah
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        predict_button = st.button("🔍 Prediksi Harga", use_container_width=True)

    # Jika tombol ditekan
    if predict_button:
        if selected_kecamatan in kecamatan_tidak_tersedia:
            st.error(f"❌ Mohon Maaf! Data Harga Rumah untuk Kecamatan **{selected_kecamatan}** belum tersedia.")
        else:
            try:
                # One-Hot Encoding hanya untuk kecamatan yang tersedia
                kecamatan_encoded = {f'kec_{kec}': 0 for kec in kecamatan_tersedia}
                kecamatan_encoded[f'kec_{selected_kecamatan}'] = 1  # Set kecamatan yang dipilih ke 1

                #Load Ordinal Encoder
                with open("Model/encoder_data_listrik.pkl", "rb") as f:
                    watt_enc = pickle.load(f)

                # Gabungkan input user ke DataFrame
                df_input = pd.DataFrame({
                    'Kamar Tidur': [kamar_tidur],
                    'Kamar Mandi': [kamar_mandi],
                    'Luas Tanah': [np.log1p(luas_tanah)],  # Transformasi log1p
                    'Luas Bangunan': [np.log1p(luas_bangunan)],  # Transformasi log1p
                    'Daya Listrik': [daya_listrik],
                    'Jumlah Lantai': [jumlah_lantai],
                    'Carport': [carport],
                    'Kamar Tidur Pembantu': [kamar_tidur_pembantu],
                    'Kamar Mandi Pembantu': [kamar_mandi_pembantu]
                })

                # Encoding kolom Daya Listrik 
                df_input[['Daya Listrik']] = watt_enc.transform(df_input[['Daya Listrik']])

                df_final = pd.concat([df_input, pd.DataFrame([kecamatan_encoded])], axis=1)

                # Pastikan urutan fitur sesuai dengan yang digunakan saat model dilatih
                expected_columns = [
                    'Kamar Tidur', 'Kamar Mandi', 'Luas Tanah', 'Luas Bangunan', 'Daya Listrik',
                    'Jumlah Lantai', 'Carport', 'Kamar Tidur Pembantu', 'Kamar Mandi Pembantu'
                ] + [f'kec_{kec}' for kec in kecamatan_tersedia]

                df_final = df_final[expected_columns]

                # Load model
                with open("Model/xgboost_optuna.pkl", "rb") as f:
                    model_xgb = pickle.load(f)

                # Prediksi harga rumah dalam bentuk log
                predicted_price_log = model_xgb.predict(df_final)

                # Konversi kembali ke harga asli
                predicted_price = np.expm1(predicted_price_log) 

                # Menampilkan hasil prediksi
                st.subheader("💰 Estimasi Harga Rumah:")
                st.markdown(
                    f"""
                    <div style='background-color: #d4edda; padding: 10px; border-radius: 5px; color: #155724; text-align: center; font-weight: bold;'>
                        Rp {predicted_price[0]:,.0f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

with tab4:
    st.title("🔎 Prediksi Spesifikasi Rumah Berdasarkan Harga dan Lokasi")

    # Load dataset rumah
    df_rumah = pd.read_csv("Dataset/Data Harga Rumah Kabupaten Tangerang.csv")

    # Buat kategori range harga
    def categorize_price(harga):
        if harga < 500_000_000:
            return "< 500 Juta"
        elif harga < 1_000_000_000:
            return "500 Juta - 1 Miliar"
        elif harga < 1_500_000_000:
            return "1 - 1.5 Miliar"
        elif harga < 2_000_000_000:
            return "1.5 - 2 Miliar"
        elif harga < 2_500_000_000:
            return "2 - 2.5 Miliar"
        elif harga < 3_000_000_000:
            return "2.5 - 3 Miliar"
        elif harga < 4_000_000_000:
            return "3 - 4 Miliar"
        elif harga < 5_000_000_000:
            return "4 - 5 Miliar"
        else:
            return "> 5 Miliar"

    df_rumah["Range Harga"] = df_rumah["Harga"].apply(categorize_price)

    # Dropdown filter
    col1, col2 = st.columns(2)
    with col1:
        lokasi_list = sorted(df_rumah["Kecamatan"].dropna().unique())
        lokasi_filter = st.selectbox("📍 Pilih Kecamatan:", lokasi_list if lokasi_list else ["(Tidak tersedia)"])

    with col2:
        harga_options = [
            "< 500 Juta", "500 Juta - 1 Miliar", "1 - 1.5 Miliar", "1.5 - 2 Miliar",
            "2 - 2.5 Miliar", "2.5 - 3 Miliar", "3 - 4 Miliar", "4 - 5 Miliar", "> 5 Miliar"
        ]
        df_subset = df_rumah[df_rumah["Kecamatan"] == lokasi_filter]
        harga_tersedia = df_subset["Range Harga"].unique().tolist()
        harga_filter = st.selectbox("💰 Pilih Rentang Harga:", [h for h in harga_options if h in harga_tersedia])

    # Filter data
    if lokasi_filter not in df_rumah["Kecamatan"].values:
        st.error(f"❌ Kecamatan **{lokasi_filter}** tidak tersedia dalam database.")
    else:
        df_filtered = df_rumah[(df_rumah["Kecamatan"] == lokasi_filter) & (df_rumah["Range Harga"] == harga_filter)]

        if not df_filtered.empty:
            st.subheader("📌 Ringkasan Spesifikasi Rumah:")

            colA, colB = st.columns(2)

            def tampilkan_range(label, data):
                min_val, max_val = data.min(), data.max()
                if min_val == max_val:
                    st.write(f"**{label}:** {min_val}")
                else:
                    st.write(f"**{label}:** {min_val} - {max_val}")

            with colA:
                tampilkan_range("Jumlah Kamar Tidur", df_filtered["Kamar Tidur"])
                tampilkan_range("Jumlah Kamar Mandi", df_filtered["Kamar Mandi"])
                tampilkan_range("Luas Tanah (m²)", df_filtered["Luas Tanah"])
                tampilkan_range("Luas Bangunan (m²)", df_filtered["Luas Bangunan"])
                tampilkan_range("Daya Listrik (Watt)", df_filtered["Daya Listrik"])

            with colB:
                tampilkan_range("Jumlah Lantai", df_filtered["Jumlah Lantai"])
                tampilkan_range("Carport (Mobil)", df_filtered["Carport"])
                tampilkan_range("Kamar Tidur Pembantu", df_filtered["Kamar Tidur Pembantu"])
                tampilkan_range("Kamar Mandi Pembantu", df_filtered["Kamar Mandi Pembantu"])

            # Tabel data rumah
            df_filtered_display = df_filtered.copy()
            df_filtered_display["Harga"] = df_filtered_display["Harga"].apply(lambda x: f"Rp {x:,.0f}")

            tampilkan_kolom = [
                "Harga", "Kecamatan", "Kamar Tidur", "Kamar Mandi", "Luas Tanah",
                "Luas Bangunan", "Daya Listrik", "Jumlah Lantai", "Carport",
                "Kamar Tidur Pembantu", "Kamar Mandi Pembantu"
            ]

            st.subheader("🏘️ Daftar Rumah Sesuai Kriteria:")
            st.dataframe(df_filtered_display[tampilkan_kolom], use_container_width=True)
        else:
            st.info("Tidak ada rumah yang sesuai dengan filter yang dipilih.")
