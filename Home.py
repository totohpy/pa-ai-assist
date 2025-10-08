import streamlit as st
from style import load_css  # Import ฟังก์ชันจากไฟล์ style.py

# --- Page Configuration ---
st.set_page_config(
    page_title="PA Planning Studio",
    page_icon="🧭",
    layout="wide"
)

# --- Load CSS ---
load_css()  # เรียกใช้ฟังก์ชันเพื่อโหลดสไตล์ทั้งหมด

# --- บอกแอปว่าตอนนี้อยู่ที่หน้า "Home" ---
# (สำคัญมากสำหรับทำให้ปุ่ม Active ทำงานถูกต้อง)
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# --- Sidebar Content ---
with st.sidebar:
    st.title("เมนูหลัก")

    # --- สร้างปุ่มเมนู ---
    # ปุ่มจะเปลี่ยนเป็น type="primary" (สีเข้ม) เมื่อ active
    if st.button("หน้าหลัก (Home)", use_container_width=True, type="primary" if st.session_state.current_page == "Home" else "secondary"):
        st.session_state.current_page = "Home"
        st.rerun() # สั่งให้โหลดหน้าใหม่เพื่ออัปเดตสถานะปุ่ม

    if st.button("Audit Design Assistant", use_container_width=True, type="primary" if st.session_state.current_page == "Design Assistant" else "secondary"):
        st.session_state.current_page = "Design Assistant"
        st.switch_page("pages/2_Design_Assistant.py")

    if st.button("Audit Plan Generator", use_container_width=True, type="primary" if st.session_state.current_page == "Plan Generator" else "secondary"):
        st.session_state.current_page = "Plan Generator"
        st.switch_page("pages/3_Plan_Generator.py")

    if st.button("PA Assistant Chat", use_container_width=True, type="primary" if st.session_state.current_page == "Chat" else "secondary"):
        st.session_state.current_page = "Chat"
        st.switch_page("pages/4_PA_Assistant_Chat.py")

    # --- Footer ---
    st.markdown("""
        <div class="sidebar-footer">
            <p>
                <span style="color: grey;">By PAO1 </span><br>
                <span style="font-size: 16px; letter-spacing: 0.5px;">
                    <span style="color: red; font-weight: bold;">A</span>udit
                    <span style="color: red; font-weight: bold;">I</span>ntelligence
                    <span style="color: red; font-weight: bold;">T</span>eam
                </span>
            </p>
        </div>
    """, unsafe_allow_html=True)


# --- Homepage Layout ---
st.title("🧭 Planning Studio – Performance Audit")
st.markdown(
    "<h3 class='subtitle'>⚒ Achieve More, Faster. Your Intelligent Efficiency Tools ᯓ★</h3>",
    unsafe_allow_html=True
)
st.write("")

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown(
        """
        <div class="feature-box">
            <span class="emoji">🏳️</span>
            <h3>Audit Design Assistant</h3>
            <p>แนะนำประเด็นตรวจสอบที่น่าสนใจ จากการวิเคราะห์ข้อมูลแผน, 6W2H, Flowchart Logic Model และฐานข้อมูลข้อตรวจพบในอดีต</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="feature-box">
            <span class="emoji">🧾</span>
            <h3>Audit Plan Generator</h3>
            <p>ช่วยร่างแผนและแนวการตรวจสอบ พร้อมระบบ AI ช่วยสร้างเนื้อหาในแต่ละส่วน และส่งออกเป็นเอกสารได้</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="feature-box">
            <span class="emoji">💬</span>
            <h3>PA Assistant Chat</h3>
            <p>ผู้ช่วยอัจฉริยะที่สามารถถาม-ตอบข้อสงสัยจากคลังข้อมูลการตรวจสอบต่างๆ ช่วยสนับสนุนการทำงาน</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Footer Information ---
st.markdown("---")
st.info("⚙️ การใช้ฟีเจอร์ AI อาจผิดพลาดได้ โปรดตรวจสอบคำตอบอีกครั้ง และระบบจะแสดงข้อมูลขณะใช้งานเท่านั้นไม่มีการจัดเก็บข้อมูลไว้")
