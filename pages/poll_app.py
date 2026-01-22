import streamlit as st
import pandas as pd
import os

# ตั้งชื่อไฟล์สำหรับเก็บข้อมูล (ในที่นี้ใช้ CSV ง่ายๆ)
DATA_FILE = 'poll_data.csv'

# ฟังก์ชันสำหรับโหลดข้อมูล
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=['vote'])

# ฟังก์ชันสำหรับบันทึกข้อมูล
def save_data(vote_text):
    df = load_data()
    new_entry = pd.DataFrame({'vote': [vote_text]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

# --- ส่วนหน้าตาของ Web App ---
st.title("📊 ระบบ Poll ความเห็นแบบ Real-time")
st.write("พิมพ์ข้อความสั้นๆ เพื่อโหวต หรือแสดงความคิดเห็น")

# 1. ส่วนรับข้อมูล (Input)
user_input = st.text_input("กรอกคำตอบของคุณ (เช่น: ชอบ, เฉยๆ, ไม่ชอบ):")

if st.button("ส่งคำตอบ"):
    if user_input.strip():
        save_data(user_input.strip()) # บันทึกข้อมูลและตัดช่องว่าง
        st.success(f"บันทึกคำตอบ: '{user_input}' เรียบร้อยแล้ว!")
    else:
        st.warning("กรุณากรอกข้อความก่อนส่ง")

st.divider() # เส้นคั่น

# 2. ส่วนประมวลผลและแสดงผล (Processing & Display)
st.subheader("ผลลัพธ์ล่าสุด")

df = load_data()

if not df.empty:
    # นับจำนวนคำตอบที่ซ้ำกัน (Frequency Count)
    # value_counts() ของ Pandas จะนับให้อัตโนมัติ
    result_counts = df['vote'].value_counts()
    
    # แสดงข้อมูลดิบเป็นตาราง (ถ้าต้องการดู)
    with st.expander("ดูข้อมูลดิบ"):
        st.dataframe(result_counts)

    # แสดงผลเป็นกราฟแท่ง
    st.bar_chart(result_counts)
    
    # แสดงสรุปอันดับ 1
    top_answer = result_counts.idxmax()
    top_count = result_counts.max()
    st.metric(label="คำตอบยอดนิยมสูงสุด", value=top_answer, delta=f"{top_count} โหวต")

else:
    st.info("ยังไม่มีข้อมูลในระบบ ลองส่งคำตอบแรกดูสิ!")
