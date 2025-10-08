import streamlit as st
from style import load_css # 1. Import ฟังก์ชันจากไฟล์ style.py

# --- Page Configuration ---
st.set_page_config(
    page_title="PA Planning Studio",
    page_icon="🧭",
    layout="wide"
)

# --- Load CSS ---
load_css() # 2. เรียกใช้ฟังก์ชันเพื่อโหลดสไตล์ทั้งหมด

# --- Sidebar Content ---
with st.sidebar:
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

# --- Create Tabs ---
tab1, tab2, tab3 = st.tabs([
    "🏳️ Audit Design Assistant",
    "🧾 Audit Plan Generator",
    "💬 PA Assistant Chat"
])

# --- Content for Tab 1 ---
with tab1:
    st.markdown(
        """
        <a href="Design_Assistant" target="_self" class="feature-link">
            <div class="feature-box">
                <span class="emoji">🏳️</span>
                <h3>Audit Design Assistant</h3>
                <p>แนะนำประเด็นตรวจสอบที่น่าสนใจ จากการวิเคราะห์ข้อมูลแผน, 6W2H, Flowchart Logic Model และฐานข้อมูลข้อตรวจพบในอดีต</p>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

# --- Content for Tab 2 ---
with tab2:
    st.markdown(
        """
        <a href="Plan_Generator" target="_self" class="feature-link">
            <div class="feature-box">
                <span class="emoji">🧾</span>
                <h3>Audit Plan Generator</h3>
                <p>ช่วยร่างแผนและแนวการตรวจสอบ พร้อมระบบ AI ช่วยสร้างเนื้อหาในแต่ละส่วน และส่งออกเป็นเอกสารได้</p>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

# --- Content for Tab 3 ---
with tab3:
    st.markdown(
        """
        <a href="PA_Assistant_Chat" target="_self" class="feature-link">
            <div class="feature-box">
                <span class="emoji">💬</span>
                <h3>PA Assistant Chat</h3>
                <p>ผู้ช่วยอัจฉริยะที่สามารถถาม-ตอบข้อสงสัยจากคลังข้อมูลการตรวจสอบต่างๆ ช่วยสนับสนุนการทำงาน</p>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

# --- Footer Information ---
st.markdown("---")
st.info("⚙️ การใช้ฟีเจอร์ AI อาจผิดพลาดได้ โปรดตรวจสอบคำตอบอีกครั้ง และระบบจะแสดงข้อมูลขณะใช้งานเท่านั้นไม่มีการจัดเก็บข้อมูลไว้")
