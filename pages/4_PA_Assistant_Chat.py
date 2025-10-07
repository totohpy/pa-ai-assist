import streamlit as st

st.set_page_config(
    page_title="PA Planning Studio",
    page_icon="🧭",
    layout="wide"
)

# --- Add the credit to the sidebar on the homepage ---
with st.sidebar:
    # This content will be pushed to the bottom by the CSS below
    st.markdown("""
        <div class="sidebar-footer">
            <hr>
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


# --- Enhanced CSS for homepage and sidebar ---
st.markdown(
    """
    <style>
    /* --- Overall App Color Theme --- */
    /* Main app background */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #e0f2f1;
    }
    /* Sidebar background and initial width */
    [data-testid="stSidebar"] {
        background-color: #e0f2f1;
        width: 350px !important;
    }

    /* --- NEW: CSS to push footer to the bottom of the sidebar --- */
    .sidebar-footer {
        position: absolute;
        bottom: 20px;
        width: 100%;
        padding-left: 1rem; /* Adjust to align with sidebar content */
        padding-right: 1rem;
    }

    /* Remove Streamlit's default top padding */
    .block-container {
        padding-top: 2rem;
    }

    /* Styling for the link to remove underline and inherit color */
    .feature-link {
        text-decoration: none !important;
        color: inherit !important;
    }
    .feature-link:hover {
        text-decoration: none !important;
        color: inherit !important;
    }

    /* Clickable Feature Box Styling */
    .feature-box {
        background-color: #e0f2f1;
        padding: 2.5rem 2rem;
        border-radius: 60px;
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
        height: 50%;
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
        font-size: 2rem;
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
    /* --- UPDATED: Style the sidebar navigation --- */
    
    /* Move the whole navigation block down */
    div[data-testid="stSidebarNav"] {
        margin-top: 30px;
    }

    /* Style each navigation link */
    div[data-testid="stSidebarNav"] > ul > li > a {
        padding: 16px 40px !important;      /* เพิ่ม padding ด้านซ้ายเป็น 24px */
        font-size: 22px !important;
        margin-bottom: 10px;
        border-radius: 6px;
        color: #FFFFFF !important;
        background-color: #b2dfdb;
        border: 1px solid #9dbdb9;      /* เพิ่มขอบสีเทาบางๆ */
    }
    
    /* Style the ACTIVE page link */
    div[data-testid="stSidebarNav"] a[aria-current="page"] {
        background-color: #80cbc4;
        color: #FFFFFF !important;         /* เปลี่ยนสีตัวอักษรเป็นสีขาว */
        font-weight: bold;
        border: 1px solid #70b8af;      /* เพิ่มขอบสีเทาเข้มขึ้นเล็กน้อย */
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --- Homepage Layout ---
with st.container():
    st.title("🧭 Planning Studio – Performance Audit")
    st.subheader("เครื่องมือดิจิทัลสำหรับงานตรวจสอบผลการดำเนินงาน", anchor=False)
    st.write("")
    st.write("")

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown(
            """
            <a href="Design_Assistant" target="_self" class="feature-link">
                <div class="feature-box box-1">
                    <span class="emoji">❔</span>
                    <h3>Design Assistant</h3>
                    <p>แนะนำประเด็นตรวจสอบที่น่าสนใจ จากการวิเคราะห์ข้อมูลแผน, 6W2H, และฐานข้อมูลข้อตรวจพบในอดีต</p>
                </div>
            </a>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <a href="Plan_Generator" target="_self" class="feature-link">
                <div class="feature-box box-2">
                    <span class="emoji">🧾</span>
                    <h3>Plan Generator</h3>
                    <p>ช่วยร่างแผนและแนวการตรวจสอบ พร้อมระบบ AI ช่วยสร้างเนื้อหาในแต่ละส่วน และส่งออกเป็นเอกสารได้</p>
                </div>
            </a>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <a href="PA_Assistant_Chat" target="_self" class="feature-link">
                <div class="feature-box box-3">
                    <span class="emoji">💬</span>
                    <h3>PA Assistant Chat</h3>
                    <p>ผู้ช่วยอัจฉริยะที่สามารถถาม-ตอบข้อสงสัยจากคลังข้อมูลและเอกสารการตรวจสอบต่างๆ เพื่อช่วยสนับสนุนการทำงาน</p>
                </div>
            </a>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")
st.info("⚙️ การใช้ฟีเจอร์ AI อาจทำผิดพลาดได้ ดังนั้น โปรดตรวจสอบคำตอบอีกครั้ง")

