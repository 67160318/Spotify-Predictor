if st.button('🔮 ทำนายความนิยม'):
    # ลำดับนี้ต้องเป๊ะตามที่โมเดลเรียนรู้มา (เช็คจาก Error mismatch ก่อนหน้า)
    # เราจะเรียงตามที่มาตรฐาน Spotify Dataset ส่วนใหญ่ใช้ครับ
    feature_names = [
        'danceability', 'energy', 'key', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    
    # สร้างข้อมูล input ในรูปแบบ List
    features = [[danceability, energy, key, loudness, speechiness, 
                 acousticness, instrumentalness, liveness, valence, tempo]]
    
    # สร้าง DataFrame และบังคับลำดับคอลัมน์ให้ถูกต้อง
    input_df = pd.DataFrame(features, columns=feature_names)
    
    try:
        # ใช้ scaler ตัวเดิมที่คุณอัปโหลดไว้
        scaled_features = scaler.transform(input_df)
        prediction = model.predict(scaled_features)
        
        st.success(f'🎉 คะแนนความนิยมที่คาดเดา: {prediction[0]:.2f} / 100')
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")
