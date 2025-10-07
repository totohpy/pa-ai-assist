import streamlit as st
from datetime import datetime
import re
from openai import OpenAI
# ... (import อื่นๆ เหมือนเดิม)

st.set_page_config(layout="wide", page_title="AI Plan Generator")
st.title("🤖 AI Plan Generator")
st.markdown("เครื่องมือช่วยสร้างแผนและแนวการตรวจสอบ พร้อมระบบ AI ช่วยร่างเนื้อหา")

# --- UPDATED: CSS to match Homepage.py ---
st.markdown(
    """
    <style>
    /* Overall App Color Theme */
    [data-testid="stAppViewContainer"] > .main { background-color: #e0f2f1; }
    [data-testid="stSidebar"] { background-color: #e0f2f1; width: 250px !important; }

    /* Sidebar Navigation Styling */
    div[data-testid="stSidebarNav"] { margin-top: 20px; }
    div[data-testid="stSidebarNav"] > ul > li > a {
        padding: 18px 40px !important; font-size: 20px !important; margin-bottom: 10px;
        border-radius: 8px; color: #263238 !important; background-color: #b2dfdb;
        border: 1px solid #9dbdb9; font-weight: 500;
    }
    div[data-testid="stSidebarNav"] a[aria-current="page"] {
        background-color: #80cbc4; color: #FFFFFF !important;
        font-weight: 600; border: 1px solid #00796b;
    }

    /* Original Styling for this page */
    .ai-expander [data-testid="stExpander"] > div:first-of-type { 
        background-color: #D1E8FF !important; border: 1px solid #007BFF; border-radius: 0.5rem; 
    }
    .ai-expander [data-testid="stExpander"] p { 
        color: #004085; font-weight: bold; 
    }
    .ai-button-container .stButton > button { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; font-weight: bold; border-radius: 0.5rem; width: 100%; }
    .ai-button-container .stButton > button:hover { background-color: #c3e6cb; color: #155724; border-color: #b1dfbb; }
    </style>
    """,
    unsafe_allow_html=True
)


# ... (โค้ดส่วนที่เหลือทั้งหมดของไฟล์นี้เหมือนเดิมทุกประการ) ...
