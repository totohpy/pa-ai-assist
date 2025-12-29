import streamlit as st
import streamlit.components.v1 as components

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Dashboard - Envi Audit SAO")

# --- Sidebar Configuration (Matching your theme) ---
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

# --- Main Title ---
st.title("📊 Environmental Audit Dashboard ")
st.markdown("Dashboard สรุปสภาพปัญหาด้านสิ่งแวดล้อมในพื้นที่และการวางแผนตรวจสอบสิ่งแวดล้อม")

# --- Custom CSS (Copied from your code for consistency) ---
st.markdown("""
<style>
    /* --- Overall App Color Theme --- */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #e0f2f1;
    }
    h1 { font-size: 36px !important; }
    [data-testid="stSidebar"] {
        background-color: #e0f2f1;
        width: 250px !important;
    }
    
    /* --- Flexbox layout for Sidebar --- */
    [data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    [data-testid="stSidebarNav"] {
        flex-grow: 1;
        margin-top: 10px;
    }
    .sidebar-footer {
        width: 100%;
        padding: 1rem;
        text-align: center;
    }

    /* Remove Streamlit's default top padding */
    .block-container {
        padding-top: 2.5rem;
    }
    
    /* --- Style the sidebar navigation --- */
    div[data-testid="stSidebarNav"] > ul > li > a {
        padding: 18px 40px !important;
        font-size: 20px !important;
        margin-bottom: 10px;
        border-radius: 8px;
        color: #263238 !important;
        background-color: #b2dfdb;
        border: 1px solid #9dbdb9;
        font-weight: 500;
    }

    /* Style the ACTIVE page link */
    div[data-testid="stSidebarNav"] a[aria-current="page"] {
        background-color: #80cbc4;
        color: #FFFFFF !important;
        font-weight: 600;
        border: 1px solid #00796b;
    }
</style>
""", unsafe_allow_html=True)

# --- Power BI Dashboard Section ---
# ใช้ Container เพื่อจัดระเบียบ (Optional)
with st.container(border=True):
    # iframe Code (Set width to 100% for responsiveness)
    iframe_code = """
    <iframe title="Envi_Audit_SAO2026"
    width="100%" height="612"
    src="https://app.powerbi.com/view?r=eyJrIjoiMzBmODQ2MTgtMGYwMy00NTc3LWI4ZTAtOWE1NzY3MjRkMGMwIiwidCI6ImI3NWFiN2IzLTU4YmEtNGZkNy1iYTU1LTMyNmY0ZWRmYzllOSIsImMiOjEwfQ%3D%3D"
    frameborder="0" allowFullScreen="true"></iframe>
    """
    
    # แสดงผล
    components.html(iframe_code, height=612)
