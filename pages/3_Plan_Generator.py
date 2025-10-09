import streamlit as st
from datetime import datetime
import re
from openai import OpenAI
import os
import html
import io
import docx
from docx.enum.section import WD_ORIENT
from docx.shared import Pt
import base64
import json
import streamlit.components.v1 as components
from style import load_css

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Plan Generator")

# --- Load CSS and Set Page State ---
load_css()
st.session_state.current_page = "Plan Generator"

# --- Sidebar ---
with st.sidebar:
    st.title("เมนูหลัก")
    if st.button("หน้าหลัก (Home)", use_container_width=True, type="primary" if st.session_state.get("current_page") == "Home" else "secondary"):
        st.switch_page("Home.py")
    if st.button("Audit Design Assistant", use_container_width=True, type="primary" if st.session_state.get("current_page") == "Design Assistant" else "secondary"):
        st.switch_page("pages/2_Design_Assistant.py")
    if st.button("Audit Plan Generator", use_container_width=True, type="primary" if st.session_state.get("current_page") == "Plan Generator" else "secondary"):
        st.rerun()
    if st.button("PA Assistant Chat", use_container_width=True, type="primary" if st.session_state.get("current_page") == "Chat" else "secondary"):
        st.switch_page("pages/4_PA_Assistant_Chat.py")
    st.markdown("""<div class="sidebar-footer">
        <p><span style="color: grey;">By PAO1 </span><br><span style="font-size: 16px; letter-spacing: 0.5px;"><span style="color: red; font-weight: bold;">A</span>udit <span style="color: red; font-weight: bold;">I</span>ntelligence <span style="color: red; font-weight: bold;">T</span>eam</span></p>
    </div>""", unsafe_allow_html=True)

# --- Main Content ---
st.title("🔮 Plan Generator")
# ... (วางเนื้อหาเดิมของ 3_Plan_Generator.py ทั้งหมดต่อจากตรงนี้) ...
