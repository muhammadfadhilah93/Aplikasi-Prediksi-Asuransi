import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from io import BytesIO
from babel.numbers import format_currency

# Konfigurasi halaman
st.set_page_config(page_title="Aplikasi Prediksi Asuransi", layout="wide")

# Memuat dataset
@st.cache_data
def load_data():
    data = pd.read_csv('Regression.csv')
    return data

data = load_data()

# Judul dan deskripsi dengan gaya center
st.markdown(
    """
    <h1 style="text-align: center;">Selamat datang di Aplikasi Prediksi Asuransi</h1>
    <p style="text-align: center; font-size: 18px;">
        Analisis, visualisasi, dan prediksi biaya asuransi berdasarkan data demografi dan kesehatan.
    </p>
    """,
    unsafe_allow_html=True
)

# Logo
st.sidebar.image("logo.png", use_container_width=True)

st.sidebar.header("Input Pengguna")
nama_pengguna = st.sidebar.text_input("Nama Anda", "Pengguna")

# Form utama untuk input
with st.sidebar.form(key="form_input_pengguna"):
    st.header("Form Input Pengguna")
    umur = st.number_input("Umur", min_value=int(data['age'].min()), max_value=int(data['age'].max()), value=30, step=1)
    bmi = st.number_input("BMI", min_value=float(data['bmi'].min()), max_value=float(data['bmi'].max()), value=25.0, step=0.1)
    jenis_kelamin = st.selectbox("Jenis Kelamin", data['sex'].unique())
    anak = st.number_input("Jumlah Anak", min_value=int(data['children'].min()), max_value=int(data['children'].max()), value=0, step=1)
    perokok = st.selectbox("Perokok", data['smoker'].unique())
    wilayah = st.selectbox("Wilayah", data['region'].unique())
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    st.sidebar.markdown(f"### Halo, {nama_pengguna}!")

X = data.drop(columns=['charges'])
y = data['charges']

fitur_kategorikal = ['sex', 'smoker', 'region']
fitur_numerikal = ['age', 'bmi', 'children']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), fitur_numerikal),
        ('cat', OneHotEncoder(drop='first'), fitur_kategorikal)
    ]
)
X_processed = preprocessor.fit_transform(X)

# Pelatihan model
@st.cache_data
def train_models():
    models = {
        "Regresi Linear": LinearRegression(),
        "Pohon Keputusan": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Support Vector Regression": SVR()
    }
    trained_models = {name: model.fit(X_processed, y) for name, model in models.items()}
    return trained_models

models = train_models()

# Fungsi untuk format IDR
def format_idr(value):
    return format_currency(value, 'IDR', locale='id_ID')

# Memproses data pengguna
if submit_button:
    data_pengguna = pd.DataFrame({
        'age': [umur],
        'sex': [jenis_kelamin],
        'bmi': [bmi],
        'children': [anak],
        'smoker': [perokok],
        'region': [wilayah]
    })
    data_pengguna_processed = preprocessor.transform(data_pengguna)

# Tab
tab0, tab1, tab2, tab3 = st.tabs(["👋 Selamat Datang", "🔍 Eksplorasi Data", "📊 Prediksi", "📈 Perbandingan Metrik"])

# Tab 0: Selamat Datang
with tab0:
    st.header(f"Selamat Datang, {nama_pengguna}!")
    st.write("Aplikasi ini memungkinkan Anda untuk:")
    st.markdown("- Mengeksplorasi dataset asuransi.")
    st.markdown("- Memprediksi biaya asuransi berdasarkan input pengguna.")
    st.markdown("- Membandingkan kinerja berbagai model machine learning.")
    st.image("1.png")

# Tab 1: Eksplorasi Data
with tab1:
    st.header("Eksplorasi Data")
    st.write("Eksplorasi dataset dan fitur-fiturnya.")

    if st.checkbox("Tampilkan Dataset"):
        st.dataframe(data)

    st.write("### Korelasi Antar Fitur")
    data_numerik = data.select_dtypes(include=[np.number])
    korelasi = data_numerik.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(korelasi, annot=True, cmap="coolwarm")
    st.pyplot(plt)

    st.write("### Distribusi Biaya Asuransi")
    fig, ax = plt.subplots()
    sns.histplot(data['charges'], kde=True, bins=30, ax=ax)
    ax.set_title("Distribusi Biaya Asuransi", fontsize=14)
    st.pyplot(fig)

# Tab 2: Prediksi
with tab2:
    st.header("Prediksi Biaya Asuransi")
    st.write("Masukkan data Anda untuk mendapatkan prediksi dari berbagai model.")

    if submit_button:
        st.write("### Data Pengguna")
        st.dataframe(data_pengguna)

        st.write("### Hasil Prediksi")
        prediksi = {}
        for name, model in models.items():
            hasil_prediksi = model.predict(data_pengguna_processed)[0]
            prediksi[name] = hasil_prediksi

        # Format hasil prediksi mata uang IDR
        pred_df = pd.DataFrame({"Model": list(prediksi.keys()), "Biaya Asuransi": list(prediksi.values())})
        pred_df["Biaya Asuransi"] = pred_df["Biaya Asuransi"].apply(lambda x: format_idr(x))

        # Tampilkan prediksi sebagai tabel
        st.write(pred_df)

        # Visualisasi hasil prediksi
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=pred_df, x="Model", y=[val for val in prediksi.values()], palette="viridis", ax=ax)
        ax.set_title("Hasil Prediksi Biaya Asuransi", fontsize=14)
        ax.set_ylabel("Biaya Asuransi (IDR)")
        ax.set_xlabel("Model")
        for i, bar in enumerate(ax.patches):
            ax.annotate(format_idr(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10, color="black")
        st.pyplot(fig)

        # Menyiapkan file CSV untuk diunduh
        buffer = BytesIO()
        pred_df.to_csv(buffer, index=False)
        buffer.seek(0)

        st.download_button(
            label="Unduh Hasil Prediksi",
            data=buffer,
            file_name="prediksi.csv",
            mime="text/csv",
            key="download-prediksi",
            help="Klik untuk mengunduh hasil prediksi dalam format CSV."
        )

# Tab 3: Perbandingan Metrik
with tab3:
    st.markdown("""
    <h2 style="color: #3498db;">📈 Perbandingan Kinerja Model</h2>
    <p style="font-size: 16px;">
        Di sini Anda dapat mengevaluasi kinerja berbagai model machine learning dalam memprediksi biaya asuransi. 
        Kami menggunakan tiga metrik evaluasi utama:
    </p>
    <ul style="font-size: 16px;">
        <li><b>MSE (Mean Squared Error):</b> Mengukur rata-rata kuadrat dari kesalahan prediksi (semakin rendah semakin baik).</li>
        <li><b>MAE (Mean Absolute Error):</b> Mengukur rata-rata kesalahan absolut (semakin rendah semakin baik).</li>
        <li><b>R² (R-squared):</b> Mengukur seberapa baik model menjelaskan variasi data (semakin tinggi semakin baik).</li>
    </ul>
    """, unsafe_allow_html=True)

    # Perhitungan metrik untuk setiap model
    metrik = []
    for name, model in models.items():
        y_pred = model.predict(X_processed)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        metrik.append({'Model': name, 'MSE': mse, 'MAE': mae, 'R2': r2})

    metrik_df = pd.DataFrame(metrik)

    st.markdown("""
    <h3 style="color: #3498db;">📊 Tabel Perbandingan Kinerja Model</h3>
    <p>Gunakan tabel ini untuk membandingkan nilai MSE, MAE, dan R² untuk setiap model. Pilih model dengan MSE dan MAE terendah serta R² tertinggi.</p>
    """, unsafe_allow_html=True)
    st.dataframe(
        metrik_df.style.format({
            "MSE": "{:,.2f}",
            "MAE": "{:,.2f}",
            "R2": "{:.2%}"
        }).background_gradient(cmap="coolwarm", subset=["R2"]),
        use_container_width=True
    )

    st.markdown("""
    <h3 style="color: #3498db;">📈 Visualisasi Kinerja Model</h3>
    <p>Grafik berikut memvisualisasikan skor R² untuk setiap model. Semakin tinggi skor R², semakin baik model dalam menjelaskan variasi data.</p>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=metrik_df, x="Model", y="R2", palette="viridis", ax=ax)
    ax.set_title("Perbandingan Skor R² untuk Setiap Model", fontsize=14, color="#2c3e50")
    ax.set_ylabel("R²", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    for i, bar in enumerate(ax.patches):
        ax.annotate(f"{bar.get_height():.2%}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, color="black")
    st.pyplot(fig)

    st.markdown("""
    <div style="padding: 15px; border-radius: 5px;">
    <h4 style="color: #3498db;">💡 Kesimpulan</h4>
    <p>
        - Model dengan nilai <b>R² tertinggi</b> memiliki kemampuan terbaik dalam menjelaskan variasi data.<br>
        - Model dengan nilai <b>MSE</b> dan <b>MAE terendah</b> menunjukkan prediksi yang paling akurat.<br>
        Anda dapat menggunakan informasi ini untuk memilih model yang paling sesuai dengan kebutuhan analisis Anda.
    </p>
    </div>
    """, unsafe_allow_html=True)


st.write("UAS DATA MINING (211220010 - MUHAMMAD FADHILAH)")
