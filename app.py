import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 1. ตั้งค่าหน้าเว็บแบบกว้าง (Wide Mode)
st.set_page_config(page_title="Spotify Predictor Pro", layout="wide", page_icon="🎵")

# ==========================================
# 2. ฟังก์ชันโหลดข้อมูลและโมเดล (พร้อม Caching เพื่อความเร็ว)
# ==========================================
@st.cache_resource
def load_assets():
    assets = {}
    
    # โหลด Scaler
    if os.path.exists('scaler.pkl'):
        assets['scaler'] = joblib.load('scaler.pkl')
    else:
        st.error("❌ ไม่พบไฟล์ scaler.pkl บน GitHub")
        st.stop()

    # โหลดโมเดลทั้ง 4 ตัว
    model_names = ['linear_regression', 'decision_tree', 'random_forest', 'knn']
    assets['models'] = {}
    
    for name in model_names:
        filename = f"{name}_model.pkl"
        if os.path.exists(filename):
            # โหลดโมเดลมาเก็บไว้โดยใช้ชื่อเล่นที่ดูดี
            assets['models'][name.replace('_', ' ').title()] = joblib.load(filename)
        else:
            st.warning(f"⚠️ ไม่พบไฟล์โมเดล {filename}")

    # โหลดข้อมูล CSV ดิบ (จำเป็นสำหรับระบบแนะนำเพลง)
    if os.path.exists('spotify_songs.csv'):
        # โหลดมาเฉพาะคอลัมน์ที่จำเป็นเพื่อประหยัด RAM
        cols_to_load = ['track_name', 'track_artist', 'track_popularity', 'danceability', 'energy', 
                        'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        df_raw = pd.read_csv('spotify_songs.csv', usecols=cols_to_load).dropna()
        assets['df_raw'] = df_raw
    else:
        st.error("❌ ไม่พบไฟล์ spotify_songs.csv บน GitHub (จำเป็นสำหรับระบบแนะนำเพลง)")
        assets['df_raw'] = None
        
    return assets

assets = load_assets()

# ==========================================
# 3. หน้าตาหน้าเว็บ (UI Design - เหมือนตัวอย่าง)
# ==========================================
st.title("🎵 Spotify Track Popularity Pro Predictor")
st.markdown("ทำนายความนิยมของเพลงจากองค์ประกอบของเสียง และค้นหาเพลงที่มีลักษณะใกล้เคียงกัน")

# ส่วนที่ 1: Sidebar (Features & Settings)
st.sidebar.header("🕹️ Features & Settings")

# 1.1 เลือกโมเดล
selected_model_name = st.sidebar.selectbox(
    'Select Machine Learning Model',
    list(assets['models'].keys())
)
selected_model = assets['models'][selected_model_name]

st.sidebar.markdown("---")
st.sidebar.subheader("Adjust Audio Features")

# 1.2 แถบเลื่อนปรับค่า (9 Features ตามที่เราเทรน)
danceability = st.sidebar.slider('Danceability', 0.0, 1.0, 0.5)
energy = st.sidebar.slider('Energy', 0.0, 1.0, 0.5)
loudness = st.sidebar.slider('Loudness (dB)', -60.0, 0.0, -10.0)
speechiness = st.sidebar.slider('Speechiness', 0.0, 1.0, 0.1)
acousticness = st.sidebar.slider('Acousticness', 0.0, 1.0, 0.2)
instrumentalness = st.sidebar.slider('Instrumentalness', 0.0, 1.0, 0.0)
liveness = st.sidebar.slider('Liveness', 0.0, 1.0, 0.1)
valence = st.sidebar.slider('Valence', 0.0, 1.0, 0.5)
tempo = st.sidebar.slider('Tempo (BPM)', 50.0, 200.0, 120.0)

# ปุ่มทำนาย
st.sidebar.markdown("---")
predict_btn = st.sidebar.button('🔮 Predict & Recommend', type='primary')

# ส่วนที่ 2: Main Area (แสดงผล)
# เราจะแบ่งหน้าจอเป็น 2 คอลัมน์ คอลัมน์ซ้ายแสดงผลทำนาย คอลัมน์ขวาแสดงเพลงแนะนำ
col_result, col_recommend = st.columns([1, 2])

if predict_btn:
    # --- ขั้นตอนทำนาย (Prediction Logic) ---
    feature_names = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                     'instrumentalness', 'liveness', 'valence', 'tempo']
    
    input_features = [[danceability, energy, loudness, speechiness, acousticness, 
                      instrumentalness, liveness, valence, tempo]]
    
    # 1. เตรียมข้อมูล Input DataFrame (เพื่อให้ Scaler ไม่งงเรื่องชื่อ)
    input_df = pd.DataFrame(input_features, columns=feature_names)
    
    try:
        # 2. Scaling ข้อมูล Input (ใช้ .values เพื่อป้องกันปัญหา Error เรื่องชื่อคอลัมน์ไม่ตรง)
        input_scaled = assets['scaler'].transform(input_df.values)
        
        # 3. ทำนายผลด้วยโมเดลที่เลือก
        prediction = selected_model.predict(input_scaled)[0]
        
        # จำกัดค่าผลลัพธ์ให้อยู่ในช่วง 0-100
        predicted_popularity = max(0, min(100, prediction))
        
        # แสดงผลในคอลัมน์ซ้าย (Result Card)
        with col_result:
            st.subheader("📊 Prediction Result")
            st.balloons()
            
            # กำหนดสีตามคะแนน
            score_color = "#1DB954" if predicted_popularity > 70 else "#FFA500" if predicted_popularity > 40 else "#FF4B4B"
            
            # สร้างการ์ดแสดงผลสวยๆ (ใช้ HTML นิดหน่อย)
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 30px; border_radius: 15px; text_align: center; border: 2px solid {score_color};">
                <p style="font-size: 20px; color: #5f6368; margin: 0;">Predicted Popularity Score</p>
                <p style="font-size: 72px; font-weight: bold; color: {score_color}; margin: 10px 0;">
                    {predicted_popularity:.2f}
                </p>
                <p style="color: #5f6368; margin: 0;">out of 100</p>
                <p style="font-size: 14px; color: #999; margin_top: 15px;">Model Used: {selected_model_name}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # --- ขั้นตอนแนะนำเพลง (Recommendation Logic) ---
        if assets['df_raw'] is not None:
            with col_recommend:
                st.subheader("🎶 Recommended Similar Songs")
                st.write("เพลงในฐานข้อมูลที่มีลักษณะทางดนตรีใกล้เคียงกับค่าที่คุณปรับมากที่สุด")
                
                # ใช้หลักการ Euclidean Distance เพื่อหาเพลงที่ "ใกล้" ที่สุด
                df_songs = assets['df_raw']
                
                # ดึงเฉพาะคอลัมน์ฟีเจอร์ออกมาเพื่อคำนวณ
                raw_features = df_songs[feature_names].values
                
                # ปรับสเกลข้อมูลเพลงทั้งหมดใน CSV (เพื่อให้เปรียบเทียบกับ Input ได้)
                raw_features_scaled = assets['scaler'].transform(raw_features)
                
                # คำนวณระยะห่าง (Distance) ระหว่าง Input (scaled) กับ ข้อมูลเพลงทั้งหมด (scaled)
                # axis=1 หมายถึงคำนวณระยะห่างระหว่าง "แถว"
                distances = np.linalg.norm(raw_features_scaled - input_scaled, axis=1)
                
                # หา Index ของเพลงที่มีระยะห่างน้อยที่สุด 5 อันดับแรก
                nearest_indices = np.argsort(distances)[:5]
                
                # ดึงข้อมูลเพลงแนะนำมาแสดง
                recommendations = df_songs.iloc[nearest_indices][['track_name', 'track_artist', 'track_popularity']]
                
                # เปลี่ยนชื่อคอลัมน์ให้ดูดีและแสดงผลในรูปแบบตาราง
                recommendations.columns = ['Track Name', 'Artist', 'Actual Popularity']
                
                # แสดงตาราง (use_container_width เพื่อให้ตารางเต็มหน้าจอ)
                st.dataframe(recommendations, use_container_width=True, hide_index=True)
                st.caption("💡 ระบบจะค้นหาเพลงที่ค่า Audio Features ทั้ง 9 อย่างรวมกันแล้วมีค่าใกล้เคียงกับที่คุณปรับมากที่สุด ไม่ใช่แค่เพลงที่ฮิตที่สุด")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")
