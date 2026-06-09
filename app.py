import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from model_utils import load_sample_dataset, predict_nlp_model, process_bulk_excel, get_top_bottom_faculties

# ==============================================================================
# 1. CONFIG UI & CUSTOM CSS
# ==============================================================================
st.set_page_config(
    page_title="UHO Sentiment & Aspect Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
    h1 { font-weight: 800; color: #0F172A; letter-spacing: -1px; }
    h2 { font-weight: 700; color: #1E293B; margin-top: 2.5rem; border-bottom: 2px solid #E2E8F0; padding-bottom: 10px; margin-bottom: 1.5rem;}
    h3 { font-weight: 600; color: #334155; margin-top: 1.5rem;}
    
    .stButton>button { 
        width: 100%; border-radius: 8px; 
        background: linear-gradient(135deg, #0284C7, #0EA5E9); 
        color: white; border: none; font-weight: 600; transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        background: linear-gradient(135deg, #0369A1, #0284C7); 
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 800; color: #0284C7; }
    
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    
    .stat-label { font-size: 1rem; color: #64748B; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;}
    .stat-value { font-size: 2.5rem; color: #0F172A; font-weight: 800; margin-top: 0.5rem;}
    
    .sentiment-pos { color: #10B981; }
    .sentiment-neg { color: #EF4444; }
    .sentiment-neu { color: #F59E0B; }
    
    /* Smooth scrolling for anchor links */
    html {
        scroll-behavior: smooth;
    }
    </style>
""", unsafe_allow_html=True)

# Set visual style for plots
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

# Memuat data
@st.cache_data
def load_data():
    df_raw = load_sample_dataset()
    return process_bulk_excel(df_raw.copy())

df_dataset = load_data()

# ==============================================================================
# 2. SIDEBAR NAVIGASI
# ==============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/id/8/87/Logo_Universitas_Halu_Oleo.png", width=100)
    st.markdown("<h3 style='text-align: left; color: #0284C7; margin-bottom: 20px; font-weight: 800;'>UHO Sentiment Dashboard</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    /* Membuat link anchor di sidebar terlihat seperti tombol */
    section[data-testid="stSidebar"] a[href^="#"] {
        display: block;
        padding: 12px 15px;
        margin: 8px 0;
        background-color: #F8FAFC;
        color: #1E293B !important;
        text-decoration: none !important;
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #E2E8F0;
        transition: all 0.2s ease-in-out;
    }
    section[data-testid="stSidebar"] a[href^="#"]:hover {
        background-color: #0284C7;
        color: white !important;
        border-color: #0284C7;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    [🏠 1. Beranda & Maps](#beranda)
    
    [🔍 2. Pencarian & Dataset](#pencarian)
    
    [🏆 3. Peringkat & Visualisasi NLP](#peringkat)
    
    [🚀 4. Uji Coba Model](#uji-coba)
    """)
    st.markdown("---")
    
    st.markdown("### Ringkasan Data")
    st.write(f"**Total Ulasan:** {len(df_dataset)}")
    pos_count = len(df_dataset[df_dataset['Sentimen_Prediksi'] == 'Positif']) if 'Sentimen_Prediksi' in df_dataset.columns else 0
    neg_count = len(df_dataset[df_dataset['Sentimen_Prediksi'] == 'Negatif']) if 'Sentimen_Prediksi' in df_dataset.columns else 0
    neu_count = len(df_dataset[df_dataset['Sentimen_Prediksi'] == 'Netral']) if 'Sentimen_Prediksi' in df_dataset.columns else 0
    
    st.markdown(f"- 🟢 Positif: {pos_count}")
    st.markdown(f"- 🔴 Negatif: {neg_count}")
    st.markdown(f"- 🟡 Netral: {neu_count}")
    
    st.markdown("---")
    st.caption("© 2024 UHO Sentiment App v2.0")

# ==============================================================================
# 3. KONTEN UTAMA (SATU HALAMAN / ONE PAGE)
# ==============================================================================

# ------------------------------------------------------------------------------
# SECTION 1: Beranda & Maps
# ------------------------------------------------------------------------------
st.header("Beranda & Maps", anchor="beranda")
st.title("🎓 Sistem Analisis Sentimen Berbasis Aspek")
st.markdown("<h4 style='color: #64748B; font-weight: 400; margin-top: -15px; margin-bottom: 30px;'>Universitas Halu Oleo (UHO)</h4>", unsafe_allow_html=True)

st.markdown("""
    <div style='background-color: #F8FAFC; padding: 20px; border-radius: 10px; border: 1px solid #E2E8F0; margin-bottom: 30px;'>
    Aplikasi ini dibangun khusus untuk menganalisis data ulasan publik mengenai berbagai fakultas di <b>Universitas Halu Oleo (UHO)</b>. 
    Sistem menggunakan model <i>Natural Language Processing (NLP)</i> untuk mengekstrak sentimen dan aspek dari setiap ulasan.
    </div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-card'><div class='stat-label'>Total Ulasan</div><div class='stat-value'>{len(df_dataset)}</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><div class='stat-label'>Sentimen Positif</div><div class='stat-value sentiment-pos'>{pos_count}</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><div class='stat-label'>Sentimen Negatif</div><div class='stat-value sentiment-neg'>{neg_count}</div></div>", unsafe_allow_html=True)
with col4:
    avg_rating = df_dataset['Rating'].mean() if 'Rating' in df_dataset.columns else 0
    st.markdown(f"<div class='metric-card'><div class='stat-label'>Rata-Rata Rating</div><div class='stat-value' style='color:#F59E0B;'>{avg_rating:.1f} ⭐</div></div>", unsafe_allow_html=True)

st.markdown("### Peta Lokasi Universitas Halu Oleo")
# Menggunakan iframe Google Maps untuk UHO
map_html = """
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3980.124578580632!2d122.51815127581105!3d-4.008401395964177!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x2d988d5e82b7db3d%3A0x6b77242c7bc015f3!2sHalu%20Oleo%20University!5e0!3m2!1sen!2sid!4v1700000000000!5m2!1sen!2sid" width="100%" height="450" style="border:0; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
"""
st.components.v1.html(map_html, height=470)


# ------------------------------------------------------------------------------
# SECTION 2: Pencarian & Dataset
# ------------------------------------------------------------------------------
st.header("Pencarian & Dataset", anchor="pencarian")
st.markdown("### Tentang Fakultas")
st.write("Gunakan panel di bawah ini untuk mencari ulasan berdasarkan fakultas tertentu, memfilter sentimen, dan melihat dataset lengkap hasil scraping dan NLP.")

with st.expander("🛠️ Panel Filter Pencarian", expanded=True):
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        if 'Nama Tempat' in df_dataset.columns:
            tempat_options = ['Semua'] + sorted(list(df_dataset['Nama Tempat'].astype(str).unique()))
        else:
            tempat_options = ['Semua']
        selected_tempat = st.selectbox("📍 Pilih Fakultas / Tempat", tempat_options, key='filter_tempat')
        
    with col_f2:
        if 'Sentimen_Prediksi' in df_dataset.columns:
            sentimen_options = ['Semua', 'Positif', 'Netral', 'Negatif']
        else:
            sentimen_options = ['Semua']
        selected_sentimen = st.selectbox("😊 Kategori Sentimen", sentimen_options, key='filter_sentimen')
        
    with col_f3:
        search_keyword = st.text_input("📝 Cari Kata Kunci di Ulasan", placeholder="Cth: dosen, fasilitas...", key='filter_keyword')

df_filtered = df_dataset.copy()

if selected_tempat != 'Semua' and 'Nama Tempat' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['Nama Tempat'] == selected_tempat]
    
if selected_sentimen != 'Semua' and 'Sentimen_Prediksi' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['Sentimen_Prediksi'] == selected_sentimen]
    
if search_keyword and 'Ulasan' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['Ulasan'].astype(str).str.contains(search_keyword, case=False, na=False)]

st.markdown(f"**Ditemukan {len(df_filtered)} ulasan yang sesuai kriteria.**")

if not df_filtered.empty:
    st.dataframe(df_filtered, use_container_width=True, hide_index=True)
else:
    st.info("Tidak ada ulasan yang cocok dengan filter pencarian.")


# ------------------------------------------------------------------------------
# SECTION 3: Peringkat & Visualisasi NLP
# ------------------------------------------------------------------------------
st.header("Peringkat & Visualisasi NLP", anchor="peringkat")

st.markdown("### 🏆 Peringkat Fakultas (Berdasarkan Rating Google Maps)")
st.markdown("<p style='color: #64748B;'>Analisis urutan tempat terbaik dan terburuk berdasarkan akumulasi rata-rata rating ulasan.</p>", unsafe_allow_html=True)

df_top, df_bottom = get_top_bottom_faculties(df_dataset)

if not df_top.empty and not df_bottom.empty:
    col_rank1, col_space, col_rank2 = st.columns([1, 0.05, 1])
    
    with col_rank1:
        st.markdown("<div style='background-color: #F0FDF4; padding: 15px; border-radius: 8px; border-left: 5px solid #16A34A; margin-bottom: 20px;'><h3 style='color: #16A34A; margin: 0;'>🟢 Top 10 Terbaik</h3></div>", unsafe_allow_html=True)
        fig_top, ax_top = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Rata_Rata_Rating', y='Nama Tempat', data=df_top, palette='viridis', ax=ax_top)
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        ax_top.set_xlabel("Rata-rata Rating", fontweight='bold')
        ax_top.set_ylabel("")
        st.pyplot(fig_top)
        
    with col_rank2:
        st.markdown("<div style='background-color: #FEF2F2; padding: 15px; border-radius: 8px; border-left: 5px solid #DC2626; margin-bottom: 20px;'><h3 style='color: #DC2626; margin: 0;'>🔴 Top 10 Terburuk</h3></div>", unsafe_allow_html=True)
        fig_bot, ax_bot = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Rata_Rata_Rating', y='Nama Tempat', data=df_bottom, palette='magma', ax=ax_bot)
        ax_bot.spines['top'].set_visible(False)
        ax_bot.spines['right'].set_visible(False)
        ax_bot.set_xlabel("Rata-rata Rating", fontweight='bold')
        ax_bot.set_ylabel("")
        st.pyplot(fig_bot)
else:
    st.warning("Data tidak mencukupi untuk menghitung peringkat.")

st.markdown("### ☁️ Visualisasi Word Cloud Fakultas")
st.markdown("Pilih fakultas untuk melihat kata-kata yang paling sering muncul dalam ulasan (Word Cloud).")

if 'Nama Tempat' in df_dataset.columns and 'Ulasan' in df_dataset.columns:
    wc_tempat = st.selectbox("Pilih Fakultas untuk Word Cloud:", sorted(list(df_dataset['Nama Tempat'].astype(str).unique())), key='wc_tempat')
    wc_text = " ".join(df_dataset[df_dataset['Nama Tempat'] == wc_tempat]['Ulasan'].astype(str).tolist())
    
    if wc_text.strip() and wc_text.strip() != "nan":
        try:
            # Menggunakan colormap yang estetik
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='ocean', max_words=100).generate(wc_text)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)
        except Exception as e:
            st.error(f"Gagal membuat Word Cloud: {str(e)}")
    else:
        st.info("Tidak ada teks ulasan yang cukup untuk membuat Word Cloud pada fakultas ini.")


# ------------------------------------------------------------------------------
# SECTION 4: Uji Coba Model NLP
# ------------------------------------------------------------------------------
st.header("Uji Coba Prediksi Model NLP", anchor="uji-coba")
st.markdown("Masukkan kalimat ulasan manual untuk memprediksi aspek dan sentimen berdasarkan model Machine Learning yang ada di dalam folder `model`.")

user_input = st.text_area("Tulis ulasan Anda di sini:", height=120, placeholder="Contoh: Fasilitas lab sangat lengkap namun dosen sering terlambat datang.")

if st.button("🚀 Prediksi Sentimen & Aspek", key='predict_btn'):
    if user_input.strip() == "":
        st.warning("Mohon masukkan teks terlebih dahulu!")
    else:
        with st.spinner("Model sedang memproses..."):
            aspek, sentimen, confidence = predict_nlp_model(user_input)
            
        st.markdown("#### Hasil Prediksi:")
        
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.markdown(f"<div class='metric-card'><div class='stat-label'>Prediksi Aspek</div><div class='stat-value' style='font-size: 1.2rem; margin-top: 1rem;'>{aspek}</div></div>", unsafe_allow_html=True)
            
        with col_r2:
            if sentimen == 'Positif':
                sen_class = "sentiment-pos"
                icon = "✅"
            elif sentimen == 'Negatif':
                sen_class = "sentiment-neg"
                icon = "❌"
            else:
                sen_class = "sentiment-neu"
                icon = "⚠️"
            
            st.markdown(f"<div class='metric-card'><div class='stat-label'>Sentimen</div><div class='stat-value {sen_class}' style='font-size: 1.5rem; margin-top: 1rem;'>{icon} {sentimen}</div></div>", unsafe_allow_html=True)
            
        with col_r3:
            st.markdown(f"<div class='metric-card'><div class='stat-label'>Tingkat Keyakinan (Confidence)</div><div class='stat-value' style='font-size: 1.5rem; margin-top: 1rem; color: #0EA5E9;'>{confidence:.2%}</div></div>", unsafe_allow_html=True)