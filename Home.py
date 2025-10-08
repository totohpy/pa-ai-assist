import streamlit as st
from style import load_css

# --- Page Configuration ---
st.set_page_config(
    page_title="PA Planning Studio",
    page_icon="🧭",
    layout="wide"
)

# --- Load CSS ---
load_css()

# --- 🎯 Set Current Page State 🎯 ---
st.session_state.current_page = "Home"

# --- Sidebar Content ---
with st.sidebar:
    st.title("เมนูหลัก")
    
    # --- ใช้ st.button และ st.switch_page พร้อมเช็ค Active State ---
    # ปุ่มจะเปลี่ยนเป็น type="primary" เมื่อ st.session_state.current_page ตรงกับชื่อของมัน
    if st.button("หน้าหลัก (Home)", use_container_width=True, type="primary" if st.session_state.current_page == "Home" else "secondary"):
        st.switch_page("Home.py")

    if st.button("Audit Design Assistant", use_container_width=True, type="primary" if st.session_state.current_page == "Design Assistant" else "secondary"):
        st.switch_page("pages/2_Design_Assistant.py")

    if st.button("Audit Plan Generator", use_container_width=True, type="primary" if st.session_state.current_page == "Plan Generator" else "secondary"):
        st.switch_page("pages/3_Plan_Generator.py")

    if st.button("PA Assistant Chat", use_container_width=True, type="primary" if st.session_state.current_page == "Chat" else "secondary"):
        st.switch_page("pages/4_PA_Assistant_Chat.py")
        
    # ... Footer ...

# --- Homepage Layout ---
# ... เนื้อหาส่วนที่เหลือของหน้า Home เหมือนเดิม ...

# --- Footer Information ---
st.markdown("---")
st.info("⚙️ การใช้ฟีจอร์ AI อาจผิดพลาดได้ โปรดตรวจสอบคำตอบอีกครั้ง และระบบจะแสดงข้อมูลขณะใช้งานเท่านั้นไม่มีการจัดเก็บข้อมูลไว้")
