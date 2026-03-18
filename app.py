import streamlit as st
import pandas as pd
import joblib
import numpy as np

# โหลดโมเดล
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    st.error("ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่ามี best_model.pkl และ scaler.pkl ในโฟลเดอร์")

st.title("🎵 Spotify Track Popularity Predictor")
st.write("แอปนี้จะช่วยทำนายว่าเพลงของคุณจะฮิตแค่ไหน (คะแนนความนิยม 0-100) จากองค์ประกอบของเสียง!")

# สร้างแถบเลื่อนปรับค่า
danceability = st.slider('ความสามารถในการเต้น (Danceability)', 0.0, 1.0, 0.5)
energy = st.slider('พลังงานความคึกคัก (Energy)', 0.0, 1.0, 0.5)
loudness = st.slider('ความดังของเสียง (Loudness dB)', -60.0, 0.0, -5.0)
speechiness = st.slider('สัดส่วนเสียงพูด (Speechiness)', 0.0, 1.0, 0.1)
acousticness = st.slider('ความเป็นดนตรีอคูสติก (Acousticness)', 0.0, 1.0, 0.2)
instrumentalness = st.slider('สัดส่วนดนตรีบรรเลง (Instrumentalness)', 0.0, 1.0, 0.0)
liveness = st.slider('ความเป็นดนตรีสด (Liveness)', 0.0, 1.0, 0.1)
valence = st.slider('อารมณ์บวกของเพลง (Valence)', 0.0, 1.0, 0.5)
tempo = st.slider('จังหวะความเร็ว (Tempo BPM)', 50.0, 200.0, 120.0)

if st.button('🔮 ทำนายความนิยม'):
    # จัดเรียงข้อมูลให้ตรงกับตอนเทรน
    features = [[danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo]]
    scaled_features = scaler.transform(features)
    
    # ทำนาย
    prediction = model.predict(scaled_features)
    st.success(f'🎉 คะแนนความนิยมที่คาดเดา: {prediction[0]:.2f} / 100')