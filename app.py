import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ต้องเป็นบรรทัดแรกสุด
st.set_page_config(page_title="Spotify Predictor", page_icon="🎵")

st.title("🎵 Spotify Track Popularity Predictor")
st.write("แอปทำนายความนิยมเพลงจากองค์ประกอบของเสียง (9 Features)")

# ฟังก์ชันโหลดโมเดล
@st.cache_resource
def load_models():
    if os.path.exists('best_model.pkl') and os.path.exists('scaler.pkl'):
        m = joblib.load('best_model.pkl')
        s = joblib.load('scaler.pkl')
        return m, s
    return None, None

model, scaler = load_models()

if model is None or scaler is None:
    st.error("❌ ไม่พบไฟล์โมเดลบน GitHub")
else:
    st.subheader("📊 ปรับค่าองค์ประกอบของเพลง")
    col1, col2 = st.columns(2)

    with col1:
        danceability = st.slider('Danceability', 0.0, 1.0, 0.5)
        energy = st.slider('Energy', 0.0, 1.0, 0.5)
        # ตัด Key ออกจากตรงนี้
        loudness = st.slider('Loudness (dB)', -60.0, 0.0, -5.0)
        speechiness = st.slider('Speechiness', 0.0, 1.0, 0.1)

    with col2:
        acousticness = st.slider('Acousticness', 0.0, 1.0, 0.2)
        instrumentalness = st.slider('Instrumentalness', 0.0, 1.0, 0.0)
        liveness = st.slider('Liveness', 0.0, 1.0, 0.1)
        valence = st.slider('Valence', 0.0, 1.0, 0.5)
        tempo = st.slider('Tempo (BPM)', 50.0, 200.0, 120.0)

    st.markdown("---")
    if st.button('🔮 ทำนายความนิยม'):
        # ลำดับ 9 ตัวตามที่ Scaler และ Model ต้องการเป๊ะๆ
        feature_names = [
            'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
        ]
        
        features = [[danceability, energy, loudness, speechiness, 
                     acousticness, instrumentalness, liveness, valence, tempo]]
        
        input_df = pd.DataFrame(features, columns=feature_names)
        
        try:
            # ใช้ .values เพื่อป้องกันปัญหาชื่อคอลัมน์ไม่ตรง
            scaled_features = scaler.transform(input_df.values)
            prediction = model.predict(scaled_features)
            
            st.balloons()
            st.success(f'### 🎉 คะแนนความนิยมที่คาดเดา: {prediction[0]:.2f} / 100')
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")
