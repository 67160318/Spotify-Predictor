import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 1. ตั้งค่าหน้าเว็บแบบกว้างและใช้ Theme สีมืด
st.set_page_config(page_title="Spotify Predictor Pro", layout="wide", page_icon="🎵", initial_sidebar_state="expanded")

# --- CUSTOM CSS: แต่ง Theme และอ่านง่าย ---
st.markdown("""
<style>
    /* เปลี่ยนสีพื้นหลังหน้าเว็บและ Sidebar */
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #121212; /* Spotify Dark */
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] {
        background-color: #000000; /* Sidebar Darker */
        border-right: 1px solid #282828;
    }
    
    /* แต่งข้อความและหัวข้อทั่วไป */
    h1, h2, h3, .stMarkdown, .stText {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
    }
    h1 {
        text-shadow: 0 0 10px #1DB954; /* Glow Title */
    }

    /* 🔥 แกปัญหาเรื่องสีในกล่องคำอธิบาย (Expander) ให้อ่านง่ายสุดๆ! 🔥 */
    [data-testid="stExpander"] {
        background-color: #181818 !important; /* พื้นหลังเข้ม */
        border: 1px solid #282828;
        border-radius: 10px;
        box-shadow: 0 0 5px rgba(29, 185, 84, 0.3); /* Subtle green glow */
    }
    
    /* บังคับสีข้อความในกล่องคำอธิบายให้เป็นสีขาวสว่าง อ่านง่าย */
    [data-testid="stExpander"] p, [data-testid="stExpander"] li {
        color: #FFFFFF !important;
        font-size: 14px;
    }
    
    /* บังคับสีข้อความหัวข้อในกล่องคำอธิบาย */
    [data-testid="stExpander"] label {
        color: #FFFFFF !important;
        font-weight: bold !important;
    }

    /* แต่ง Slider และปุ่ม */
    .stSlider > div > div > div > div {
        background-color: #1DB954 !important; /* Spotify Green */
    }
    .stButton > button {
        background-color: #1DB954 !important; /* Spotify Green */
        color: #FFFFFF !important;
        border-radius: 20px !important;
        border: none !important;
        font-weight: bold !important;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.05); /* Zoom Effect */
        background-color: #1ED760 !important; /* Lighter Green */
    }

    /* แต่งตารางแนะนำเพลงแบบ HTML Bypass */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
        background-color: #181818;
        border-radius: 10px;
        overflow: hidden;
    }
    .custom-table th {
        background-color: #282828;
        padding: 15px;
        text-align: left;
        color: #B3B3B3;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 1px;
    }
    .custom-table td {
        padding: 15px;
        border-bottom: 1px solid #282828;
        color: #FFFFFF;
    }
    .custom-table tr:hover {
        background-color: #2A2A2A; /* Highlight Row */
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ฟังก์ชันโหลดข้อมูลและโมเดล
# ==========================================
@st.cache_resource
def load_assets():
    assets = {}
    if os.path.exists('scaler.pkl'):
        assets['scaler'] = joblib.load('scaler.pkl')
    else:
        st.error("❌ ไม่พบไฟล์ scaler.pkl บน GitHub")
        st.stop()

    model_names = ['linear_regression', 'decision_tree', 'random_forest', 'knn']
    assets['models'] = {}
    for name in model_names:
        filename = f"{name}_model.pkl"
        if os.path.exists(filename):
            assets['models'][name.replace('_', ' ').title()] = joblib.load(filename)
        else:
            st.warning(f"⚠️ ไม่พบไฟล์โมเดล {filename}")

    if os.path.exists('spotify_songs.csv'):
        try:
            assets['df_raw'] = pd.read_csv('spotify_songs.csv', on_bad_lines='skip')
        except Exception:
            assets['df_raw'] = None
    else:
        assets['df_raw'] = None
    return assets

assets = load_assets()

# ==========================================
# 3. หน้าตาหน้าเว็บ (UI)
# ==========================================
# ใช้ st.markdown กับ CSS ที่เราแต่งไว้
st.markdown("<h1>🎵 Spotify Track Popularity Pro Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #B3B3B3;'>ทำนายความนิยมของเพลงจากองค์ประกอบของเสียง และค้นหาเพลงที่มีลักษณะใกล้เคียงกัน</p>", unsafe_allow_html=True)

# ==========================================
# 4. ส่วนอธิบาย Features (เพิ่มคะแนนความโปร่งใสและ UX)
# ==========================================
with st.expander("ℹ️ คำอธิบายองค์ประกอบเสียง (Audio Features Explained)"):
    st.markdown("""
    * **Danceability (0.0 - 1.0):** ความเหมาะในการเต้น ยิ่งค่ามาก จังหวะยิ่งชัดเจนและเต้นตามได้ง่าย
    * **Energy (0.0 - 1.0):** ความเร้าใจและพลังของเพลง ค่าที่สูงจะรู้สึกถึงความเร็ว ดัง และหนักหน่วง (เช่น เพลงร็อคหรือ EDM)
    * **Loudness (dB):** ความดังเบาโดยรวมของเพลง (มักติดลบ) ยิ่งใกล้ 0 แปลว่ายิ่งดังมาก
    * **Speechiness (0.0 - 1.0):** สัดส่วนของเสียงพูดในเพลง ค่าที่สูงมักจะเป็นเพลงแร็ป ทอล์คโชว์ หรือพอดแคสต์
    * **Acousticness (0.0 - 1.0):** ความเป็นอคูสติก (ไม่ใช้เครื่องดนตรีไฟฟ้า) ยิ่งใกล้ 1 แปลว่าใช้เครื่องดนตรีสดสูง
    * **Instrumentalness (0.0 - 1.0):** การไม่มีเสียงร้อง ยิ่งใกล้ 1 แปลว่าเป็นเพลงบรรเลงล้วนๆ
    * **Liveness (0.0 - 1.0):** การแสดงสด ยิ่งค่าสูง แปลว่าเพลงนี้มีแนวโน้มถูกบันทึกเสียงจากการเล่นสด (มีเสียงคนดู)
    * **Valence (0.0 - 1.0):** อารมณ์ความอารมณ์ดีของเพลง ค่าสูง = ร่าเริง/มีความสุข, ค่าต่ำ = เศร้า/หดหู่/โกรธ
    * **Tempo (BPM):** ความเร็วของจังหวะเพลง (Beats Per Minute)
    """)

st.sidebar.markdown("<h2 style='color: #1DB954;'>🕹️ Features & Settings</h2>", unsafe_allow_html=True)

if not assets['models']:
    st.sidebar.error("❌ ไม่พบไฟล์โมเดล กรุณาอัปโหลดลง GitHub")
    st.stop()

selected_model_name = st.sidebar.selectbox('Select Machine Learning Model', list(assets['models'].keys()))
selected_model = assets['models'][selected_model_name]

st.sidebar.markdown("<hr style='border: 1px solid #282828;'>", unsafe_allow_html=True)
st.sidebar.subheader("Adjust Audio Features")

danceability = st.sidebar.slider('Danceability', 0.0, 1.0, 0.5)
energy = st.sidebar.slider('Energy', 0.0, 1.0, 0.5)
loudness = st.sidebar.slider('Loudness (dB)', -60.0, 0.0, -10.0)
speechiness = st.sidebar.slider('Speechiness', 0.0, 1.0, 0.1)
acousticness = st.sidebar.slider('Acousticness', 0.0, 1.0, 0.2)
instrumentalness = st.sidebar.slider('Instrumentalness', 0.0, 1.0, 0.0)
liveness = st.sidebar.slider('Liveness', 0.0, 1.0, 0.1)
valence = st.sidebar.slider('Valence', 0.0, 1.0, 0.5)
tempo = st.sidebar.slider('Tempo (BPM)', 50.0, 200.0, 120.0)

st.sidebar.markdown("<hr style='border: 1px solid #282828;'>", unsafe_allow_html=True)
predict_btn = st.sidebar.button('🔮 Predict & Recommend', type='primary')

col_result, col_recommend = st.columns([1, 2])

if predict_btn:
    feature_names = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                     'instrumentalness', 'liveness', 'valence', 'tempo']
    input_features = [[danceability, energy, loudness, speechiness, acousticness, 
                      instrumentalness, liveness, valence, tempo]]
    input_df = pd.DataFrame(input_features, columns=feature_names)
    
    try:
        input_scaled = assets['scaler'].transform(input_df.values)
        prediction = selected_model.predict(input_scaled)[0]
        predicted_popularity = max(0, min(100, prediction))
        
        with col_result:
            st.markdown("<h3 style='border-bottom: 2px solid #1DB954; padding-bottom: 10px;'>📊 Prediction Result</h3>", unsafe_allow_html=True)
            st.balloons()
            # ใช้ CSS แต่งกล่องคะแนนให้เฟี้ยว!
            st.markdown(f"""
            <div style="background-color: #000000; padding: 40px; border_radius: 20px; text_align: center; border: 3px solid #1DB954; box-shadow: 0 0 20px #1DB954;">
                <p style="font-size: 18px; color: #B3B3B3; margin: 0; text-transform: uppercase; letter-spacing: 1px;">Predicted Popularity Score</p>
                <p style="font-size: 80px; font-weight: bold; color: #FFFFFF; margin: 15px 0; text-shadow: 0 0 15px #FFFFFF;">
                    {predicted_popularity:.2f}
                </p>
                <p style="color: #B3B3B3; margin: 0;">out of 100</p>
                <p style="font-size: 12px; color: #888; margin_top: 20px; font-style: italic;">Powered by {selected_model_name} Model</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_recommend:
            st.markdown("<h3 style='border-bottom: 2px solid #1DB954; padding-bottom: 10px;'>🎶 Recommended Similar Songs</h3>", unsafe_allow_html=True)
            
            if assets['df_raw'] is not None:
                try:
                    df_songs = assets['df_raw'].dropna(subset=feature_names)
                    
                    # ✨ โค้ดปราบเพลงแฝด (drop duplicates) ก็ยังมีอยู่นะจ๊ะ! ✨
                    df_songs = df_songs.drop_duplicates(subset=['track_name', 'track_artist']).reset_index(drop=True)
                    
                    raw_features = df_songs[feature_names].values
                    raw_features_scaled = assets['scaler'].transform(raw_features)
                    
                    distances = np.linalg.norm(raw_features_scaled - input_scaled, axis=1)
                    nearest_indices = np.argsort(distances)[:5]
                    
                    recommendations = df_songs.iloc[nearest_indices][['track_name', 'track_artist', 'track_popularity']]
                    
                    clean_recom = pd.DataFrame({
                        'Track Name': recommendations['track_name'].astype(str).tolist(),
                        'Artist': recommendations['track_artist'].astype(str).tolist(),
                        'Popularity': recommendations['track_popularity'].tolist()
                    })
                    
                    # ✨ ใช้ HTML + CSS แต่งตารางให้ดูดีแบบ Spotify Pro อ่านง่าย 100% ✨
                    html_table = clean_recom.to_html(index=False, justify='left', border=0, classes='custom-table')
                    # ✨ บีบโค้ดเป็นบรรทัดเดียว ป้องกัน Streamlit มองเป็น Code Block ✨
                    html_code = f"<div class='custom-table-container'>{html_table}</div>"
                    
                    st.markdown(html_code, unsafe_allow_html=True)
                    
                except KeyError:
                    st.warning("⚠️ โชว์คะแนนได้ปกติ แต่ไม่สามารถแนะนำเพลงได้ (ไฟล์ข้อมูลไม่สมบูรณ์)")
            else:
                st.warning("⚠️ ไม่พบไฟล์ spotify_songs.csv สำหรับดึงข้อมูลแนะนำเพลง")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
