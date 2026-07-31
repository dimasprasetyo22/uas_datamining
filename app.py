import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="UAS Data Mining - Prediksi & Clustering",
    page_icon="📊",
    layout="wide"
)

# Sidebar Navigasi
st.sidebar.title("Navigasi Menu")
menu = st.sidebar.radio("Pilih Bagian:", [
    "🏠 Beranda",
    "🩺 1. Klasifikasi Diabetes",
    "☕ 2. Clustering Gerai Kopi & Zona Sepi"
])

# ==========================================
# 1. HALAMAN BERANDA
# ==========================================
if menu == "🏠 Beranda":
    st.title("Aplikasi Web Data Mining - UAS")
    st.markdown("""
    Selamat datang di aplikasi berbasis web interaktif untuk pemenuhan tugas proyek Data Mining.
    
    Aplikasi ini terbagi menjadi dua modul utama:
    1. **Klasifikasi Risiko Diabetes**: Memprediksi status diabetes pasien menggunakan tiga algoritma (*KNN, Naïve Bayes, dan Decision Tree*) beserta evaluasi metrik dan *Confusion Matrix*.
    2. **Clustering Lokasi Gerai Kopi**: Mengelompokkan titik lokasi gerai kopi menggunakan algoritma *K-Means* untuk mendeteksi persebaran serta zona dengan potensi pelanggan rendah (zona sepi).
    
    *Silakan pilih menu di sidebar sebelah kiri untuk mulai menjelajahi aplikasi.*
    """)

# ==========================================
# 2. KLASIFIKASI DIABETES
# ==========================================
elif menu == "🩺 1. Klasifikasi Diabetes":
    st.title("Prediksi Risiko Diabetes Berdasarkan Data Pasien")
    st.markdown("""
    **Deskripsi Proyek:** Modul ini menggunakan *Pima Indians Diabetes Database*. 
    Kami melatih tiga model klasifikasi (*K-Nearest Neighbors, Naïve Bayes, dan Decision Tree*) untuk memprediksi apakah seorang pasien mengidap diabetes berdasarkan fitur klinis.
    """)
    
    @st.cache_data
    def load_diabetes_data():
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv(url, names=columns)
        return df

    try:
        df_diabetes = load_diabetes_data()
        
        X = df_diabetes.drop('Outcome', axis=1)
        y = df_diabetes['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_scaled, y_train)
        y_pred_knn = knn.predict(X_test_scaled)
        
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        y_pred_nb = nb.predict(X_test)
        
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        
        def get_metrics(y_true, y_pred):
            return {
                "Akurasi": accuracy_score(y_true, y_pred),
                "Presisi": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "F1-Score": f1_score(y_true, y_pred, zero_division=0)
            }
            
        metrics_df = pd.DataFrame({
            "KNN": get_metrics(y_test, y_pred_knn),
            "Naïve Bayes": get_metrics(y_test, y_pred_nb),
            "Decision Tree": get_metrics(y_test, y_pred_dt)
        }).T
        
        st.subheader("📊 Perbandingan Metrik Evaluasi Model")
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'))
        
        st.subheader("📉 Visualisasi Confusion Matrix")
        model_choice_cm = st.selectbox("Pilih Model untuk melihat Confusion Matrix:", ["KNN", "Naïve Bayes", "Decision Tree"])
        
        if model_choice_cm == "KNN":
            cm = confusion_matrix(y_test, y_pred_knn)
            title = "Confusion Matrix - KNN"
        elif model_choice_cm == "Naïve Bayes":
            cm = confusion_matrix(y_test, y_pred_nb)
            title = "Confusion Matrix - Naïve Bayes"
        else:
            cm = confusion_matrix(y_test, y_pred_dt)
            title = "Confusion Matrix - Decision Tree"
            
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("🔍 Prediksi Status Pasien Baru")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            p_pregnancies = st.number_input("Jumlah Kehamilan (Pregnancies)", min_value=0, max_value=20, value=2)
            p_glucose = st.number_input("Kadar Glukosa (Glucose)", min_value=0, max_value=200, value=120)
            p_bp = st.number_input("Tekanan Darah (Blood Pressure)", min_value=0, max_value=150, value=70)
        with col2:
            p_skin = st.number_input("Ketebalan Kulit (Skin Thickness)", min_value=0, max_value=100, value=20)
            p_insulin = st.number_input("Kadar Insulin (Insulin)", min_value=0, max_value=900, value=80)
            p_bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=0.0, max_value=70.0, value=25.5)
        with col3:
            p_dpf = st.number_input("Fungsi Silsilah Diabetes (DPF)", min_value=0.0, max_value=3.0, value=0.5)
            p_age = st.number_input("Usia (Age)", min_value=1, max_value=120, value=30)
            
        selected_model_deploy = st.selectbox("Pilih Model untuk Prediksi:", ["KNN", "Naïve Bayes", "Decision Tree"])
        
        if st.button("Prediksi Sekarang"):
            input_data = np.array([[p_pregnancies, p_glucose, p_bp, p_skin, p_insulin, p_bmi, p_dpf, p_age]])
            
            if selected_model_deploy == "KNN":
                input_scaled = scaler.transform(input_data)
                pred = knn.predict(input_scaled)[0]
            elif selected_model_deploy == "Naïve Bayes":
                pred = nb.predict(input_data)[0]
            else:
                pred = dt.predict(input_data)[0]
                
            if pred == 1:
                st.error("Hasil Prediksi: Pasien diprediksi **MENGIDAP DIABETES**.")
            else:
                st.success("Hasil Prediksi: Pasien diprediksi **TIDAK MENGIDAP DIABETES**.")
                
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data atau melatih model: {e}")

# ==========================================
# 3. CLUSTERING GERAI KOPI & ZONA SEPI
# ==========================================
elif menu == "☕ 2. Clustering Gerai Kopi & Zona Sepi":
    st.title("Analisis Klaster Lokasi Gerai Kopi dan Deteksi Zona Sepi")
    st.markdown("""
    **Deskripsi Proyek:** Modul ini menerapkan *K-Means Clustering* menggunakan dataset riil gerai kopi 
    untuk memetakan persebaran titik lokasi serta mengidentifikasi zona dengan potensi rendah (zona sepi).
    """)
    
    try:
        df_coffee = pd.read_csv("dataset/lokasi_gerai_kopi_clean.csv")
        
        k_clusters = st.slider("Pilih Jumlah Klaster (K):", min_value=2, max_value=5, value=3)
        
        feature_cols = ['x', 'y', 'population_density']
        X_cluster = df_coffee[feature_cols]
        scaler_cluster = StandardScaler()
        X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
        
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        df_coffee['Cluster'] = kmeans.fit_predict(X_cluster_scaled)
        
        cluster_means = df_coffee.groupby('Cluster')['population_density'].mean()
        sparse_cluster = cluster_means.idxmin()
        df_coffee['Status_Zona'] = df_coffee['Cluster'].apply(lambda x: '⚠️ Zona Sepi' if x == sparse_cluster else '✅ Zona Ramai')
        
        total_lokasi = len(df_coffee)
        total_zona_sepi = (df_coffee['Status_Zona'] == '⚠️ Zona Sepi').sum()
        total_zona_ramai = (df_coffee['Status_Zona'] == '✅ Zona Ramai').sum()
        
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Total Titik Lokasi", total_lokasi)
        mcol2.metric("Total Zona Ramai", total_zona_ramai)
        mcol3.metric("Total Zona Sepi", total_zona_sepi)
        
        st.markdown("---")
        st.subheader("📍 Visualisasi Persebaran Klaster Gerai Kopi (Scatter Plot)")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=df_coffee, 
            x='x', 
            y='y', 
            hue='Status_Zona', 
            palette={'✅ Zona Ramai': 'blue', '⚠️ Zona Sepi': 'red'},
            style='Cluster',
            s=80, 
            ax=ax
        )
        ax.set_title("Peta Persebaran Klaster Gerai Kopi & Deteksi Zona Sepi")
        ax.set_xlabel("Koordinat X")
        ax.set_ylabel("Koordinat Y")
        st.pyplot(fig)
        
        with st.expander("📂 Lihat Data Mentah & Hasil Klasterisasi (Tabel)"):
            st.dataframe(df_coffee)
        
        st.markdown("---")
        st.subheader("🗺️ Uji Lokasi Gerai Baru")
        
        col_input1, col_input2, col_input3 = st.columns(3)
        with col_input1:
            new_x = st.number_input("Koordinat X Baru", value=float(df_coffee['x'].mean()))
        with col_input2:
            new_y = st.number_input("Koordinat Y Baru", value=float(df_coffee['y'].mean()))
        with col_input3:
            new_density = st.number_input("Estimasi Kepadatan Penduduk", value=float(df_coffee['population_density'].mean()))
            
        if st.button("Analisis Zona Lokasi Baru"):
            new_data = scaler_cluster.transform([[new_x, new_y, new_density]])
            predicted_cluster = kmeans.predict(new_data)[0]
            
            is_sparse = (predicted_cluster == sparse_cluster)
            
            st.info(f"Lokasi tersebut masuk ke dalam **Klaster {predicted_cluster}**.")
            if is_sparse:
                st.warning("Hasil Analisis: Lokasi ini tergolong **ZONA SEPI** (potensi kepadatan rendah).")
            else:
                st.success("Hasil Analisis: Lokasi ini tergolong **ZONA RAMAI** (potensi kepadatan baik).")
                
    except Exception as e:
        st.error(f"Gagal memuat dataset gerai kopi. Pastikan file 'lokasi_gerai_kopi_clean.csv' berada di dalam folder 'dataset': {e}")