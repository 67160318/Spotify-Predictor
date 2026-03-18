import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. โหลดโมเดลและตัวปรับสเกล
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    st.error("❌ ไม่พบไฟล์โมเดล! ตรวจสอบว่ามีไฟล์ best_model.pkl และ scaler.pkl บน GitHub หรือยัง")

# 2. ส่วนหัวของหน้าเว็บ
st.set_page_config(page_title="Spotify Predictor", page_icon="🎵")
st.title("🎵 Spotify Track Popularity Predictor")
st.write("แอปนี้จะช่วยทำนายว่าเพลงของคุณจะฮิตแค่ไหน (0-100) จากองค์ประกอบของเสียง!")

# 3. สร้างแถบเลื่อนปรับค่า (10 ตัวแปรตามที่โมเดลต้องการ)
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

# 4. ปุ่มทำนายผล
st.markdown("---")
if st.button('🔮 ทำนายความนิยม'):
    # ลำดับชื่อคอลัมน์ต้องตรงกับตอนเทรนเป๊ะๆ
    feature_names = [
        'danceability', 'energy', 'key', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    
    # จัดข้อมูลเข้าลิสต์
    features = [[danceability, energy, key, loudness, speechiness, 
                 acousticness, instrumentalness, liveness, valence, tempo]]
    
    # แปลงเป็น DataFrame
    input_df = pd.DataFrame(features, columns=feature_names)
    
    try:
        # ปรับสเกลและทำนาย
        scaled_features = scaler.transform(input_df)
        prediction = model.predict(scaled_features)
        
        # แสดงผลลัพธ์
        st.balloons()
        st.success(f'### 🎉 คะแนนความนิยมที่คาดเดา: {prediction[0]:.2f} / 100')
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")
