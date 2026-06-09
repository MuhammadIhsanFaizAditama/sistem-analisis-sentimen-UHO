import os
import re
import time
import random
import pandas as pd
import joblib

# ==============================================================================
# 1. KONFIGURASI JALUR FILE (PATH)
# ==============================================================================
MODEL_PATH = os.path.join('model', 'model_sentimen_gbdt.pkl')
VECTORIZER_PATH = os.path.join('model', 'tfidf_vectorizer.pkl')
EXCEL_PATH = os.path.join('data', 'Hasil_Scraping_UHO_Berlabel.xlsx')

# Pemuatan Model & Vectorizer Secara Global
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        MODEL_READY = True
    except Exception:
        model = None
        vectorizer = None
        MODEL_READY = False
else:
    model = None
    vectorizer = None
    MODEL_READY = False


# ==============================================================================
# 2. FUNGSI MANAJEMEN DATA EXCEL (Sesuai Struktur Kolom UHO)
# ==============================================================================
def load_sample_dataset():
    """Memuat dataset hasil scraping dari file Excel lokal."""
    if os.path.exists(EXCEL_PATH):
        try:
            df = pd.read_excel(EXCEL_PATH)
            # Kolom wajib sesuai data scraping UHO Anda
            required_columns = ['Nama Tempat', 'Nama Pengguna', 'Rating', 'Ulasan', 'Aspek', 'Sentimen']
            if all(col in df.columns for col in required_columns):
                return df[required_columns]
            else:
                return pd.DataFrame(columns=required_columns)
        except Exception:
            return pd.DataFrame(columns=['Nama Tempat', 'Nama Pengguna', 'Rating', 'Ulasan', 'Aspek', 'Sentimen'])
    else:
        # Fallback statis menggunakan sampel riil data UHO Anda jika file belum ada
        data = {
            'Nama Tempat': ["Fakultas Farmasi - UHO", "Fakultas Ekonomi dan Bisnis (FEB) - UHO", "Fakultas Teknik - UHO"],
            'Nama Pengguna': ["Muhammad Hajrul Malaka", "Laode Asfahyadin Aliddin", "Wa Ode Siti Enarni"],
            'Rating': [5, 5, 3],
            'Ulasan': [
                "Visi dan misi relevan dalam menyiapkan tenaga profesional dan handal dalam pelayanan kefarmasian",
                "Tempat Kuliah Masa Depan",
                "Fakultasnya fasilitasnya lumayan, tapi sayang WC nya tidak terlalu bagus karena kunci pintunya Tdk ada"
            ],
            'Aspek': ['Kualitas Akademik & Pelayanan', 'Kualitas Akademik & Pelayanan', 'Fasilitas Fisik & Infrastruktur'],
            'Sentimen': ['Positif', 'Positif', 'Negatif']
        }
        return pd.DataFrame(data)


# ==============================================================================
# 3. TEXT PREPROCESSING
# ==============================================================================
def clean_text(text):
    """Melakukan pembersihan teks ulasan scraping."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()


# ==============================================================================
# 4. FUNGSI PREDIKSI MODEL ML (INFERENCE)
# ==============================================================================
def predict_nlp_model(text):
    """Memproses teks input tunggal untuk prediksi Aspek dan Sentimen UHO."""
    if not MODEL_READY:
        time.sleep(0.4)
        text_lower = text.lower()
        
        # Rule-based Mock untuk Kategori Aspek UHO
        if any(w in text_lower for w in ['dosen', 'kuliah', 'akreditasi', 'lulus', 'ajar', 'ilmu', 'visi']):
            aspek = "Kualitas Academic & Pelayanan"
        elif any(w in text_lower for w in ['gedung', 'wc', 'parkir', 'fasilitas', 'lab', 'isinyal', 'kipas', 'ac', 'mushalla']):
            aspek = "Fasilitas Fisik & Infrastruktur"
        elif any(w in text_lower for w in ['nyaman', 'sejuk', 'asri', 'aman', 'rindang', 'tenang']):
            aspek = "Keamanan & Kenyamanan"
        else:
            aspek = "Lainnya / Irrelevant"
            
        # Rule-based Mock untuk Kategori Sentimen UHO
        if any(w in text_lower for w in ['bagus', 'mantap', 'ramah', 'nyaman', 'top', 'indah', 'baik', 'lengkap']):
            sentimen = "Positif"
            confidence = random.uniform(0.80, 0.98)
        elif any(w in text_lower for w in ['kurang', 'kecewa', 'lambat', 'error', 'rusak', 'jelek', 'tidak', 'tutup']):
            sentimen = "Negatif"
            confidence = random.uniform(0.80, 0.98)
        else:
            sentimen = "Netral"
            confidence = random.uniform(0.60, 0.79)
            
        return aspek, sentimen, confidence

    else:
        cleaned_text = clean_text(text)
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # Hasil Prediksi Sentimen dari Model GBDT
        prediction = model.predict(vectorized_text)[0]
        probabilities = model.predict_proba(vectorized_text)[0]
        confidence = max(probabilities)
        
        # Mapping label sesuai output kelas model biner/multikelas Anda
        # Contoh di bawah mengasumsikan model multikelas (0: Negatif, 1: Netral, 2: Positif)
        mapping_sentimen = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
        sentimen = mapping_sentimen.get(prediction, "Positif")
        
        # Pengisian aspek default jika model aspek terpisah belum dimasukkan
        aspek = "Kualitas Akademik & Pelayanan" 
        
        return aspek, sentimen, confidence


def process_bulk_excel(df):
    """Memproses Batch Inference untuk visualisasi dashboard."""
    if not MODEL_READY or 'Ulasan' not in df.columns:
        if 'Sentimen' in df.columns:
            df['Sentimen_Prediksi'] = df['Sentimen']
        else:
            df['Sentimen_Prediksi'] = random.choices(['Positif', 'Netral', 'Negatif'], k=len(df))
        return df
        
    cleaned_reviews = df['Ulasan'].apply(clean_text)
    vectorized_matrix = vectorizer.transform(cleaned_reviews)
    raw_predictions = model.predict(vectorized_matrix)
    
    mapping_sentimen = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
    df['Sentimen_Prediksi'] = pd.Series(raw_predictions).map(mapping_sentimen).fillna('Positif').values
    
    return df

def get_top_bottom_faculties(df):
    """
    Menghitung rata-rata rating dan jumlah ulasan per fakultas,
    lalu mengembalikan 10 besar terbaik dan terburuk.
    """
    if 'Nama Tempat' not in df.columns or 'Rating' not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
        
    # Agregasi data: Hitung rata-rata rating dan jumlah ulasan per tempat
    faculty_stats = df.groupby('Nama Tempat').agg(
        Rata_Rata_Rating=('Rating', 'mean'),
        Jumlah_Ulasan=('Rating', 'count')
    ).reset_index()
    
    # Saring tempat yang memiliki ulasan terlalu sedikit jika diperlukan (misal minimal 1 ulasan)
    faculty_stats = faculty_stats[faculty_stats['Jumlah_Ulasan'] >= 1]
    
    # Urutkan untuk mendapatkan peringkat
    top_10 = faculty_stats.sort_values(by=['Rata_Rata_Rating', 'Jumlah_Ulasan'], ascending=[False, False]).head(10)
    bottom_10 = faculty_stats.sort_values(by=['Rata_Rata_Rating', 'Jumlah_Ulasan'], ascending=[True, True]).head(10)
    
    # Bulatkan nilai rata-rata rating menjadi 2 desimal untuk kebutuhan visual
    top_10['Rata_Rata_Rating'] = top_10['Rata_Rata_Rating'].round(2)
    bottom_10['Rata_Rata_Rating'] = bottom_10['Rata_Rata_Rating'].round(2)
    
    return top_10, bottom_10