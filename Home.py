import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="PA Planning Studio",
    page_icon="🧭",
    layout="wide"
)

# --- CSS Styling for the entire app ---
st.markdown(
    """
    <style>
    /* --- Overall App Color Theme --- */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #e0f2f1;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1 { 
        font-size: 38px !important; 
    }
    .subtitle {
        font-style: italic; 
        color: #2baf2b; 
        font-size: 18px;
    }
    
    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background-color: #e0f2f1;
        width: 250px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    [data-testid="stSidebarNav"] {
        flex-grow: 1;
        margin-top: 20px;
    }
    .sidebar-footer {
        width: 100%;
        padding: 1rem;
        text-align: center;
    }

    /* --- Sidebar Navigation Links --- */
    div[data-testid="stSidebarNav"] > ul > li > a {
        padding: 18px 40px !important;
        font-size: 20px !important;
        margin-bottom: 10px;
        border-radius: 8px;
        color: #26328 !important;     /* Inactive link text color */
        background-color: #b2dfdb;     /* Inactive link background */
        border: 1px solid #9dbdb9;
        font-weight: 500;
        transition: background-color 0.2s ease, color 0.2s ease;
    }
    div[data-testid="stSidebarNav"] > ul > li > a:hover {
        background-color: #80cbc4;     /* Hover background color */
        color: #FFFFFF !important;       /* Hover text color */
    }
    div[data-testid="stSidebarNav"] a[aria-current="page"] {
        background-color: #00796b;     /* Active page link background */
        color: #FFFFFF !important;       /* Active page link text */
        font-weight: 600;
        border: 1px solid #004d40;
    }

    /* --- Custom Tab Styling --- */
    button[data-baseweb="tab"] {
        border-radius: 2px;
        padding: 8px 18px;
        margin: 0px;
        font-size: 16px;
        letter-spacing: 0.3px;
        font-weight: normal;
        color: white !important;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s ease-in-out;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        transform: translateY(-2px);
        opacity: 1;
        color: #000000 !important;
        background-color: #FFFFFF;
    }
    button[data-baseweb="tab"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
        opacity: 0.95;
    }
    div[data-baseweb="tab-list"] button:nth-of-type(1) { background-color: #A93C2D; }
    div[data-baseweb="tab-list"] button:nth-of-type(2) { background-color: #4D8076; }
    div[data-baseweb="tab-list"] button:nth-of-type(3) { background-color: #4A6A8A; }
    div[data-baseweb="tab-list"] {
        border-bottom: none !important;
        margin-bottom: 2rem;
        flex-wrap: wrap;
        gap: 4px;
    }

    /* --- Feature Box Styling --- */
    .feature-link { text-decoration: none !important; color: inherit !important; }
    .feature-box {
        background-color: #ffffff;
        padding: 2rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
        height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        border: 1px solid #d0e0df;
    }
    .feature-box:hover {
        transform: translateY(-10px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    .feature-box .emoji { font-size: 2.5rem; line-height: 1.5; }
    .feature-box h3 { margin-top: 0.5rem; margin-bottom: 0.5rem; font-size: 1.4rem; }
    .feature-box p { color: #6c757d; font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True
)

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
