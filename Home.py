import streamlit as st

st.set_page_config(
    page_title="PA Planning Studio",
    page_icon="🧭",
    layout="wide"
)

# --- Add the credit to the bottom of the sidebar ---
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


# --- Enhanced CSS for homepage and sidebar ---
st.markdown(
    """
    <style>
    /* --- Overall App Color Theme --- */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #e0f2f1;
    }
    [data-testid="stSidebar"] {
        background-color: #e0f2f1;
        width: 250px !important;
    }
    
    /* --- Flexbox layout for Sidebar --- */
    /* This targets the inner container of the sidebar */
    [data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    /* This makes the navigation take up all available space, pushing the footer down */
    [data-testid="stSidebarNav"] {
        flex-grow: 1;
        margin-top: 20px; /* Move navigation down */
    }
    .sidebar-footer {
        width: 100%;
        padding: 1rem;
        text-align: center; /* Center the footer content */
    }

    /* Remove Streamlit's default top padding */
    .block-container {
        padding-top: 2rem;
    }

    /* --- Feature Box Styling (Main Page) --- */
    .feature-link { text-decoration: none !important; color: inherit !important; }
    .feature-link:hover { text-decoration: none !important; color: inherit !important; }
    .feature-box {
        background-color: #e0f2f1;
        padding: 2rem 1rem;
        border-radius: 20px;
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        border: 2px solid #d0e0df;
    }
    .feature-box:hover {
        transform: translateY(-10px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    .feature-box .emoji { font-size: 2.5rem; line-height: 1.5; }
    .feature-box h3 { margin-top: 0.1rem; margin-bottom: 0.1rem; font-size: 1.2rem; }
    .feature-box p { color: #6c757d; font-size: 1rem; }
    
    /* --- Style the sidebar navigation --- */
    div[data-testid="stSidebarNav"] > ul > li > a {
        padding: 18px 40px !important; /* Increased padding for more height */
        font-size: 20px !important;    /* Larger font size */
        margin-bottom: 10px;
        border-radius: 8px;
        color: #263238 !important;     /* Darker text for inactive links */
        background-color: #80deea;     /* Light teal for inactive links */
        border: 3px solid #9dbdb9;
        font-weight: 500;
    }
    
    /* Style the ACTIVE page link */
    div[data-testid="stSidebarNav"] a[aria-current="page"] {
        background-color: #80cbc4;     /* Dark teal for active link */
        color: #FFFFFF !important;     /* White text for active link */
        font-weight: 600;
        border: 1px solid #00796b;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Homepage Layout ---
st.title("🧭 Planning Studio – Performance Audit")
st.subheader("", anchor=False)
st.markdown(
    "<h2 style='font-style: italic; color: #2baf2b; font-size: 22px'> ⚒ Achieve More, Faster. Your Intelligent Efficiency Tools  ᯓ★ </h3>",
    unsafe_allow_html=True
)
st.write("")
st.write("")

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown(
        """
        <a href="Design_Assistant" target="_self" class="feature-link">
            <div class="feature-box box-1">
                <span class="emoji">🏳️</span>
                <h3>Audit Design Assistant</h3>
                <p>แนะนำประเด็นตรวจสอบที่น่าสนใจ จากการวิเคราะห์ข้อมูลแผน, 6W2H, Flowchart Logic Model  และฐานข้อมูลข้อตรวจพบในอดีต</p>
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
                <h3>Audit Plan Generator</h3>
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
                <p>ผู้ช่วยอัจฉริยะที่สามารถถาม-ตอบข้อสงสัยจากคลังข้อมูลการตรวจสอบต่างๆ ช่วยสนับสนุนการทำงาน</p>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.info("⚙️ การใช้ฟีเจอร์ AI อาจผิดพลาดได้ โปรดตรวจสอบคำตอบอีกครั้ง และระบบจะแสดงข้อมูลขณะใช้งานเท่านั้นไม่มีการจัดเก็บข้อมูลไว้")
