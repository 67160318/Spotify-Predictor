import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# 1. ตั้งค่าหน้าเว็บแบบกว้าง
st.set_page_config(page_title="Spotify Predictor Pro", layout="wide", page_icon="🎵")

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
st.title("🎵 Spotify Track Popularity Pro Predictor")
st.markdown("ทำนายความนิยมของเพลงจากองค์ประกอบของเสียง และค้นหาเพลงที่มีลักษณะใกล้เคียงกัน")

st.sidebar.header("🕹️ Features & Settings")

if not assets['models']:
    st.sidebar.error("❌ ไม่พบไฟล์โมเดล กรุณาอัปโหลดลง GitHub")
    st.stop()

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
    input_features = [[danceability, energy, loudness, speechiness, acousticness, 
                      instrumentalness, liveness, valence, tempo]]
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
            
            if assets['df_raw'] is not None:
                try:
                    df_songs = assets['df_raw'].dropna(subset=feature_names)
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
                    
                    # ✨ บีบโค้ดเป็นบรรทัดเดียว ป้องกัน Streamlit มองเป็น Code Block ✨
                    html_table = clean_recom.to_html(index=False, justify='center', border=0)
                    html_code = f"<style>.custom-table {{ width: 100%; border-collapse: collapse; font-family: sans-serif; }} .custom-table th {{ background-color: #f0f2f6; padding: 12px; text-align: left; border-bottom: 2px solid #ddd; }} .custom-table td {{ padding: 10px; border-bottom: 1px solid #eee; }} .custom-table tr:hover {{ background-color: #f9f9f9; }}</style><div class='custom-table'>{html_table}</div>"
                    
                    st.markdown(html_code, unsafe_allow_html=True)
                    
                except KeyError:
                    st.warning("⚠️ โชว์คะแนนได้ปกติ แต่ไม่สามารถแนะนำเพลงได้ (ไฟล์ข้อมูลไม่สมบูรณ์)")
            else:
                st.warning("⚠️ ไม่พบไฟล์ spotify_songs.csv สำหรับดึงข้อมูลแนะนำเพลง")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
