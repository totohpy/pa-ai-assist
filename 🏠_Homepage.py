import streamlit as st

st.set_page_config(
    page_title="PA Planning Studio",
    page_icon="🧭",
    layout="wide"
)

# --- New, Enhanced CSS for a beautiful homepage ---
st.markdown(
    """
    <style>
    /* Main container styling */
    .main-container {
        padding: 2rem;
        background-color: #f8f9fa;
        border-radius: 15px;
    }

    /* Remove Streamlit's default top padding */
    .block-container {
        padding-top: 2rem;
    }

    /* Clickable Feature Box Styling */
    .feature-link {
        text-decoration: none;
    }
    .feature-box {
        background-color: #ffffff;
        padding: 2.5rem 2rem;
        border-radius: 10px;
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        border: 1px solid #e0e0e0;
    }
    .feature-box:hover {
        transform: translateY(-10px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    .feature-box .emoji {
        font-size: 4rem;
        line-height: 1;
    }
    .feature-box h3 {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-size: 1.5rem;
    }
    .feature-box p {
        color: #6c757d;
        font-size: 1rem;
    }

    /* Color Overrides for each box */
    .box-1 h3 { color: #A93C2D; } /* Red from Design Assistant tabs */
    .box-2 h3 { color: #2E8B57; } /* A nice green for Plan Generator */
    .box-3 h3 { color: #4A6A8A; } /* Blue from PA Assistant Chat tab */
    </style>
    """,
    unsafe_allow_html=True
)

# --- New Homepage Layout ---
with st.container():
    st.title("🧭 Planning Studio – Performance Audit")
    st.header("เครื่องมือดิจิทัลสำหรับงานตรวจสอบผลการดำเนินงาน", anchor=False)
    st.write("")
    st.write("")

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        # The entire markdown block is wrapped in a page link
        st.page_link(
            "pages/2_✨_Design_Assistant.py",
            label="""
            <div class="feature-box box-1">
                <span class="emoji">✨</span>
                <h3>Design Assistant</h3>
                <p>แนะนำประเด็นตรวจสอบที่น่าสนใจ จากการวิเคราะห์ข้อมูลแผน, 6W2H, และฐานข้อมูลข้อตรวจพบในอดีต</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.page_link(
            "pages/3_📜_Plan_Generator.py",
            label="""
            <div class="feature-box box-2">
                <span class="emoji">📜</span>
                <h3>Plan Generator</h3>
                <p>ช่วยร่างแผนและแนวการตรวจสอบ พร้อมระบบ AI ช่วยสร้างเนื้อหาในแต่ละส่วน และส่งออกเป็นเอกสารได้</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.page_link(
            "pages/4_💬_PA_Assistant_Chat.py",
            label="""
            <div class="feature-box box-3">
                <span class="emoji">💬</span>
                <h3>PA Assistant Chat</h3>
                <p>ผู้ช่วยอัจฉริยะที่สามารถถาม-ตอบข้อสงสัยจากคลังข้อมูลและเอกสารต่างๆ เพื่อช่วยสนับสนุนการทำงาน</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Remove the old footer from here, as it's now managed in the sub-pages' sidebars
