import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

DATA_FILE = 'poll_data.csv'
# ⚠️ อย่าลืมไฟล์ฟอนต์นะครับ
THAI_FONT_PATH = 'Sarabun-Regular.ttf' 

def load_data():
    if os.path.exists(DATA_FILE):
        # โหลดมาทั้งหมด แล้วแปลงเป็น String ให้หมดป้องกัน Error
        df = pd.read_csv(DATA_FILE)
        df['vote'] = df['vote'].astype(str) 
        return df
    else:
        return pd.DataFrame(columns=['vote'])

def save_data(vote_text):
    df = load_data()
    new_entry = pd.DataFrame({'vote': [vote_text]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

st.title("☁️ Word Cloud (แบบนับทั้งประโยค)")
st.write("พิมพ์อะไรก็ได้ วรรคตอนก็ได้ ระบบจะนับเป็น 1 คำตอบทันที")

# Input
user_input = st.text_input("กรอกคำตอบ (เช่น '1.1 ดีมาก'):")

if st.button("ส่งคำตอบ"):
    if user_input.strip():
        save_data(user_input.strip())
        st.success("บันทึกแล้ว!")
    else:
        st.warning("กรุณากรอกข้อความ")

st.divider()

# --- ส่วนแสดงผลที่แก้ไขใหม่ ---
st.subheader("ผลลัพธ์")

df = load_data()

if not df.empty:
    # 1. นับจำนวนคำตอบที่ซ้ำกัน และแปลงเป็น Dictionary
    # ตัวอย่างผลลัพธ์: {'1.1 ดีมาก': 5, 'เฉยๆ': 2, 'ชอบ (ที่สุด)': 1}
    vote_counts = df['vote'].value_counts().to_dict()

    # 2. สร้าง WordCloud จากผลนับ (Frequency) โดยตรง
    # วิธีนี้จะไม่สนใจ Regex แล้ว เพราะเราระบุมาแล้วว่าคำไหนมีค่าเท่าไหร่
    try:
        wc = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            font_path=THAI_FONT_PATH,
            # ไม่ต้องใส่ regexp แล้ว
            collocations=False
        ).generate_from_frequencies(vote_counts) # ✅ ใช้คำสั่งนี้แทน

        # 3. แสดงผล
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
        
        # แสดงตารางข้อมูลดิบประกอบ เพื่อเช็คความถูกต้อง
        with st.expander("ดูตารางสรุปจำนวน"):
            st.write(vote_counts)
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
        st.info("💡 อย่าลืมเช็คไฟล์ Font ภาษาไทย")

else:
    st.info("รอข้อมูลแรก...")
