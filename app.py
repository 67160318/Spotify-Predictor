import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title="Spotify Predictor Pro", layout="wide", page_icon="🎵")

@st.cache_resource
def load_assets():
    assets = {}
    if os.path.exists('scaler.pkl'):
        assets['scaler'] = joblib.load('scaler.pkl')
    else:
        st.error("❌ ไม่พบไฟล์ scaler.pkl")
        st.stop()

    model_names = ['linear_regression', 'decision_tree', 'random_forest', 'knn']
    assets['models'] = {}
    for name in model_names:
        filename = f"{name}_model.pkl"
        if os.path.exists(filename):
            assets['models'][name.replace('_', ' ').title()] = joblib.load(filename)

    if os.path.exists('spotify_songs.csv'):
        try:
            # โหลดข้อมูลแบบเผื่อเหนียว ป้องกันไฟล์พัง
            assets['df_raw'] = pd.read_csv('spotify_songs.csv', on_bad_lines='skip')
        except Exception:
            assets['df_raw'] = None
    else:
        assets['df_raw'] = None
        
    return assets

assets = load_assets()

st.title("🎵 Spotify Track Popularity Pro Predictor")
st.markdown("ทำนายความนิยมของเพลงจากองค์ประกอบของเสียง และค้นหาเพลงที่มีลักษณะใกล้เคียงกัน")

st.sidebar.header("🕹️ Features & Settings")
selected_model_name = st.sidebar.selectbox('Select Machine Learning Model', list(assets['models'].keys()))
selected_model = assets['models'][selected_model_name]

st.sidebar.markdown("---")
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

st.sidebar.markdown("---")
predict_btn = st.sidebar.button('🔮 Predict & Recommend', type='primary')

col_result, col_recommend = st.columns([1, 2])

if predict_btn:
    feature_names = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 
                     'instrumentalness', 'liveness', 'valence', 'tempo']
    input_features = [[danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo]]
    input_df = pd.DataFrame(input_features, columns=feature_names)
    
    try:
        input_scaled = assets['scaler'].transform(input_df.values)
        prediction = selected_model.predict(input_scaled)[0]
        predicted_popularity = max(0, min(100, prediction))
        
        with col_result:
            st.subheader("📊 Prediction Result")
            st.balloons()
            score_color = "#1DB954" if predicted_popularity > 70 else "#FFA500" if predicted_popularity > 40 else "#FF4B4B"
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 30px; border_radius: 15px; text_align: center; border: 2px solid {score_color};">
                <p style="font-size: 20px; color: #5f6368; margin: 0;">Predicted Popularity Score</p>
                <p style="font-size: 72px; font-weight: bold; color: {score_color}; margin: 10px 0;">
                    {predicted_popularity:.2f}
                </p>
                <p style="color: #5f6368; margin: 0;">out of 100</p>
                <p style="font-size: 14px; color: #999; margin_top: 15px;">Model: {selected_model_name}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_recommend:
            st.subheader("🎶 Recommended Similar Songs")
            # ✨ ระบบดักจับ Error ตรงนี้!
            if assets['df_raw'] is not None:
                try:
                    df_songs = assets['df_raw'].dropna(subset=feature_names) # เช็คว่ามีคอลัมน์ครบไหม
                    raw_features = df_songs[feature_names].values
                    raw_features_scaled = assets['scaler'].transform(raw_features)
                    distances = np.linalg.norm(raw_features_scaled - input_scaled, axis=1)
                    nearest_indices = np.argsort(distances)[:5]
                    
                    recommendations = df_songs.iloc[nearest_indices][['track_name', 'track_artist', 'track_popularity']]
                    recommendations.columns = ['Track Name', 'Artist', 'Actual Popularity']
                    st.dataframe(recommendations, use_container_width=True, hide_index=True)
                except KeyError:
                    st.warning("⚠️ โชว์คะแนนได้ปกติ แต่ไม่สามารถแนะนำเพลงได้ (ไฟล์ spotify_songs.csv ข้อมูลคอลัมน์ไม่ถูกต้อง)")
            else:
                st.warning("⚠️ ไม่พบไฟล์ spotify_songs.csv สำหรับดึงข้อมูลแนะนำเพลง")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
