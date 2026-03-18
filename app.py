import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ต้องเป็นบรรทัดแรกสุดของ Streamlit commands
st.set_page_config(page_title="Spotify Predictor", page_icon="🎵")

st.title("🎵 Spotify Track Popularity Predictor")
st.write("แอปนี้จะช่วยทำนายว่าเพลงของคุณจะฮิตแค่ไหน (0-100) จากองค์ประกอบของเสียง!")

# ฟังก์ชันโหลดโมเดล
@st.cache_resource
def load_models():
    # เช็คว่ามีไฟล์อยู่ในโฟลเดอร์จริงไหม
    if os.path.exists('best_model.pkl') and os.path.exists('scaler.pkl'):
        m = joblib.load('best_model.pkl')
        s = joblib.load('scaler.pkl')
        return m, s
    return None, None

model, scaler = load_models()

if model is None or scaler is None:
    st.error("❌ ไม่พบไฟล์โมเดล! ตรวจสอบว่าคุณอัปโหลดไฟล์ 'best_model.pkl' และ 'scaler.pkl' ไว้ในหน้าแรกของ GitHub หรือยัง")
else:
    # สร้างแถบเลื่อนปรับค่า
    st.subheader("📊 ปรับค่าองค์ประกอบของเพลง")
    col1, col2 = st.columns(2)

    with col1:
        danceability = st.slider('Danceability', 0.0, 1.0, 0.5)
        energy = st.slider('Energy', 0.0, 1.0, 0.5)
        key = st.slider('Key', 0, 11, 5)
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
        feature_names = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 
                         'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        features = [[danceability, energy, key, loudness, speechiness, 
                     acousticness, instrumentalness, liveness, valence, tempo]]
        
        input_df = pd.DataFrame(features, columns=feature_names)
        
        try:
            scaled_features = scaler.transform(input_df)
            prediction = model.predict(scaled_features)
            st.balloons()
            st.success(f'### 🎉 คะแนนความนิยมที่คาดเดา: {prediction[0]:.2f} / 100')
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")
