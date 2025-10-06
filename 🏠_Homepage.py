import streamlit as st

st.set_page_config(
    page_title="PA Planning Studio",
    page_icon="🧭",
    layout="wide"
)

st.title("🧭 Planning Studio – Performance Audit")
st.markdown("---")

st.markdown(
    """
    <style>
    .feature-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        transition: transform 0.2s;
        height: 100%;
    }
    .feature-box:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .feature-box h3 {
        color: #007bff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header("เครื่องมือดิจิทัลสำหรับงานตรวจสอบผลการดำเนินงาน", anchor=False)
st.write("")

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    with st.container(border=True):
        st.markdown(
            """
            <div class="feature-box">
                <h3>✨ Design Assistant</h3>
                <p>แนะนำประเด็นตรวจสอบที่น่าสนใจ โดยการวิเคราะห์จากข้อมูลแผน, 6W2H, Logic Model, และฐานข้อมูลข้อตรวจพบในอดีต</p>
            </div>
            """,
            unsafe_allow_html=True
        )

with col2:
    with st.container(border=True):
        st.markdown(
            """
            <div class="feature-box">
                <h3>📜 Plan Generator</h3>
                <p>ช่วยร่างแผนและแนวการตรวจสอบ พร้อมระบบ AI ช่วยสร้างเนื้อหาในแต่ละส่วน และส่งออกเป็นเอกสารได้</p>
            </div>
            """,
            unsafe_allow_html=True
        )

with col3:
    with st.container(border=True):
        st.markdown(
            """
            <div class="feature-box">
                <h3>💬 PA Assistant Chat</h3>
                <p>ผู้ช่วยอัจฉริยะที่สามารถถาม-ตอบข้อสงสัยจากคลังข้อมูลและเอกสารที่อัปโหลด เพื่อช่วยสนับสนุนการทำงาน</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")
st.info("💡 กรุณาเลือกเมนูจากแถบด้านข้าง (Sidebar) เพื่อเริ่มต้นใช้งาน")
