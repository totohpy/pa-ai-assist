# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
import io
from PyPDF2 import PdfReader
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(page_title="Design Assistant", page_icon="✨", layout="wide")

# (Sidebar is managed from the main Homepage.py file)
with st.sidebar:
    pass

# --- Helper Functions (No changes needed) ---
# ... (All functions like init_state, next_id, etc. remain the same) ...

# Initialize session state and variables
init_state()
plan = st.session_state["plan"]
logic_df = st.session_state["logic_items"]
methods_df = st.session_state["methods"]
kpis_df = st.session_state["kpis"]
risks_df = st.session_state["risks"]
audit_issues_df = st.session_state["audit_issues"]

# --- Main App UI ---
st.title("✨ Design Assistant")

with st.expander("💡 คำแนะนำการใช้งาน"):
    st.info("กรุณาระบุข้อมูล อย่างน้อย **ระบุ แผน & 6W2H** ส่วนใดส่วนหนึ่ง เพื่อค้นหาข้อตรวจพบที่ผ่านมาและให้ PA Assistant แนะนำ ได้แม่นยำที่สุด")

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

    /* Tab Styling from the original file */
    button[data-baseweb="tab"] { border-radius: 2px; padding: 8px 6px; margin: 0px; font-size: 16px; letter-spacing: 0.3px; font-weight: normal; color: white !important; border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: all 0.2s ease-in-out; }
    button[data-baseweb="tab"][aria-selected="true"] { box-shadow: 0 4px 12px rgba(0,0,0,0.25); transform: translateY(-2px); opacity: 1; }
    button[data-baseweb="tab"]:hover { transform: translateY(-1px); box-shadow: 0 3px 8px rgba(0,0,0,0.15); }
    div[data-baseweb="tab-list"] button:nth-of-type(-n+5) { background-color: #A93C2D; }
    div[data-baseweb="tab-list"] button:nth-of-type(6), div[data-baseweb="tab-list"] button:nth-of-type(7) { background-color: #4D8076; }
    div[data-baseweb="tab-list"] button:nth-of-type(8) { background-color: #4A6A8A; }
    div[data-baseweb="tab-list"] { border-bottom: none !important; margin-bottom: 15px; flex-wrap: wrap; gap: 2px; } 
    h4 { color: #007bff !important; border-bottom: 2px solid #e0e0e0; padding-bottom: 5px; } 
    </style>
    """,
    unsafe_allow_html=True
)

tab_plan, tab_logic, tab_method, tab_kpi, tab_risk, tab_issue, tab_preview, tab_assist = st.tabs([
    "1.&nbsp;ระบุ แผน & 6W2H", "2.&nbsp;ระบุ Logic Model", "3.&nbsp;ระบุ Methods", 
    "4.&nbsp;ระบุ KPIs", "5.&nbsp;ระบุ Risks", "🔍 ค้นหาข้อตรวจพบที่ผ่านมา", 
    "📋 สรุปข้อมูล (Preview)", "✨ PA Assistant แนะนำประเด็น"
]) 

# ... (โค้ดส่วนที่เหลือของ tab ทั้งหมดเหมือนเดิมทุกประการ) ...
