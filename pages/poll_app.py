import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud

DATA_FILE = 'poll_data.csv'
# ⚠️ แก้ชื่อไฟล์ฟอนต์ให้ตรงกับที่คุณโหลดมา
# ถ้ายังไม่มีไฟล์ฟอนต์ ให้คอมเมนต์บรรทัดนี้ แล้วภาษาไทยจะเป็นสี่เหลี่ยม
THAI_FONT_PATH = 'Sarabun-Regular.ttf' 

def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=['vote'])

def save_data(vote_text):
    df = load_data()
    new_entry = pd.DataFrame({'vote': [vote_text]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

st.title("☁️ ระบบ Poll แบบ Word Cloud")
st.write("พิมพ์ความรู้สึกของคุณสั้นๆ (คำซ้ำเยอะ ตัวยิ่งใหญ่)")

# Input
user_input = st.text_input("กรอกคำตอบ (เช่น: สนุก, ดีมาก, ง่วง):")

if st.button("ส่งคำตอบ"):
    if user_input.strip():
        save_data(user_input.strip())
        st.success("บันทึกแล้ว!")
    else:
        st.warning("กรุณากรอกข้อความ")

st.divider()

# Visualization
st.subheader("ผลลัพธ์แบบ Word Cloud")

df = load_data()

if not df.empty:
    # 1. รวมคำตอบทั้งหมดให้เป็นข้อความยาวๆ ข้อความเดียว คั่นด้วยเว้นวรรค
    # เพราะ WordCloud ต้องการ Text ก้อนใหญ่ก้อนเดียว
    text_data = " ".join(df['vote'].astype(str))

    # 2. ตั้งค่า WordCloud
    # regexp=r"[ก-๙a-zA-Z]+" ช่วยให้จับตัวอักษรไทยได้ดีขึ้น
    try:
        wc = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            font_path=THAI_FONT_PATH, # ใส่ path ฟอนต์ไทยตรงนี้
            regexp=r"[ก-๙a-zA-Z]+",    # Regular Expression เพื่อให้รองรับภาษาไทย
            collocations=False         # ปิดการจับคู่คำซ้ำซ้อน
        ).generate(text_data)

        # 3. แสดงผลด้วย Matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off") # ปิดแกน x, y ไม่ให้รก
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการสร้าง Word Cloud: {e}")
        st.info("💡 คำแนะนำ: คุณอาจจะยังไม่ได้วางไฟล์ Font ภาษาไทย (.ttf) ในโฟลเดอร์")
    
    # แสดงจำนวนคนตอบทั้งหมด
    st.caption(f"จำนวนความคิดเห็นทั้งหมด: {len(df)} คน")

else:
    st.info("ยังไม่มีข้อมูล ส่งคำตอบแรกเพื่อเริ่มสร้าง Word Cloud เลย!")
