# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
# ... (โค้ดส่วนบนอื่นๆ) ...

st.title("🧭 Planning Studio – Performance Audit")

# ----------------- START: Custom CSS for Styling and Responsiveness (ปรับปรุงแท็บและมือถือ) -----------------
st.markdown("""
<style>
/* 1. GLOBAL FONT/BACKGROUND ADJUSTMENTS */
body {
    font-family: 'Kanit', sans-serif;
}

/* 2. STYLE TABS AS COLORED BUTTONS (Custom Tabs) */

/* A. Container for all tabs: REMOVE the default bottom bar and set spacing */
div[data-baseweb="tab-list"] {
    border-bottom: none !important; /* *** กำจัดเส้นแนวนอนใต้แท็บเริ่มต้น *** */
    margin-bottom: 15px;
    flex-wrap: wrap; 
    gap: 10px; /* เพิ่มช่องว่างระหว่างปุ่ม */
}

/* B. Base style for ALL tab buttons */
button[data-baseweb="tab"] {
    /* Override Streamlit's default tab styling */
    border-bottom: none !important;
    background-color: #ffffff !important; 
    
    /* Apply button-like styling (Non-Equal Width is default here) */
    border: 1px solid #007bff; 
    border-radius: 8px;
    padding: 10px 15px; 
    transition: all 0.3s;
    font-weight: bold;
    color: #007bff !important; /* สีตัวอักษรเริ่มต้น */
    box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1); 
    
    /* ต้องบังคับให้ background เป็นสีขาว/ใส */
    background-color: white !important;
    
    /* กำจัดส่วนขยายที่ Streamlit ใช้ในการทำไฮไลต์ใต้แท็บ */
    &::after {
        content: none !important;
    }
}

/* C. Style for the ACTIVE tab (when selected) */
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #007bff !important; /* พื้นหลังสีน้ำเงินเข้ม */
    color: white !important; /* ตัวอักษรสีขาว */
    border: 1px solid #007bff;
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
}

/* 3. MOBILE RESPONSIVENESS ADJUSTMENTS */
@media (max-width: 768px) {
    .st-emotion-cache-18ni2cb, .st-emotion-cache-1jm69l4 {
        width: 100% !important;
        margin-bottom: 1rem;
    }
}

/* 4. STYLE HEADERS */
h4 {
    color: #007bff !important;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)
# ----------------- END: Custom CSS -----------------

tab_plan, tab_logic, tab_method, tab_kpi, tab_risk, tab_issue, tab_preview, tab_assist, tab_chatbot = st.tabs([
# ... (โค้ดส่วนล่างเหมือนเดิม) ...
