import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="PA Planning Studio",
    page_icon="🧭",
    layout="wide"
)

# --- Helper Function for Feature Boxes ---
def feature_box(emoji, title, description, link):
    """Creates a clickable feature box using Markdown."""
    st.markdown(
        f"""
        <a href="{link}" target="_self" class="feature-link">
            <div class="feature-box">
                <span class="emoji">{emoji}</span>
                <h3>{title}</h3>
                <p>{description}</p>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

# --- CSS Styling ---
st.markdown(
    """
    <style>
    /* --- General Theme --- */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #e0f2f1;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1 {
        font-size: 38px !important;
    }
    .subtitle-text {
        font-style: italic;
        color: #00897b; /* A slightly darker, more professional green */
        font-size: 20px;
        font-weight: 500;
        text-align: left;
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
        color: #263238 !important;
        background-color: #b2dfdb; /* Inactive link background */
        border: 1px solid #9dbdb9;
        font-weight: 500;
        transition: background-color 0.2s ease, color 0.2s ease;
    }
    /* ADDED: Hover effect for sidebar links */
    div[data-testid="stSidebarNav"] > ul > li > a:hover {
        background-color: #80cbc4; /* Darker teal on hover */
        color: #FFFFFF !important;
    }
    div[data-testid="stSidebarNav"] a[aria-current="page"] {
        background-color: #00796b; /* Active link background */
        color: #FFFFFF !important;
        font-weight: 600;
        border: 1px solid #004d40;
    }

    /* --- Homepage Feature Boxes --- */
    .feature-link {
        text-decoration: none !important;
        color: inherit !important;
    }
    .feature-box {
        background-color: #ffffff; /* White background for contrast */
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
    .feature-box .emoji {
        font-size: 2.5rem;
        line-height: 1.5;
    }
    .feature-box h3 {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 1.4rem; /* Slightly larger title */
    }
    .feature-box p {
        color: #6c757d;
        font-size: 1rem;
    }
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
    "<p class='subtitle-text'>⚒️ Achieve More, Faster. Your Intelligent Efficiency Tools</p>",
    unsafe_allow_html=True
)
st.write("") # Add vertical space

# Create 3 columns for the feature boxes
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    feature_box(
        emoji="🏳️",
        title="Audit Design Assistant",
        description="แนะนำประเด็นตรวจสอบที่น่าสนใจ จากการวิเคราะห์ข้อมูลแผน, 6W2H, Flowchart Logic Model และฐานข้อมูลข้อตรวจพบในอดีต",
        link="Design_Assistant"
    )

with col2:
    feature_box(
        emoji="🧾",
        title="Audit Plan Generator",
        description="ช่วยร่างแผนและแนวการตรวจสอบ พร้อมระบบ AI ช่วยสร้างเนื้อหาในแต่ละส่วน และส่งออกเป็นเอกสารได้",
        link="Plan_Generator"
    )

with col3:
    feature_box(
        emoji="💬",
        title="PA Assistant Chat",
        description="ผู้ช่วยอัจฉริยะที่สามารถถาม-ตอบข้อสงสัยจากคลังข้อมูลการตรวจสอบต่างๆ ช่วยสนับสนุนการทำงาน",
        link="PA_Assistant_Chat"
    )

st.markdown("---")
st.info("⚙️ การใช้ฟีเจอร์ AI อาจผิดพลาดได้ โปรดตรวจสอบคำตอบอีกครั้ง และระบบจะแสดงข้อมูลขณะใช้งานเท่านั้นไม่มีการจัดเก็บข้อมูลไว้")
