# นี่คือเนื้อหาทั้งหมดของไฟล์ style.py (เวอร์ชันอัปเดต)
import streamlit as st

def load_css():
    st.markdown(
        """
        <style>
        /* --- General Theme & Sidebar Base --- */
        [data-testid="stAppViewContainer"] > .main { background-color: #e0f2f1; }
        .block-container { padding-top: 2rem; }
        [data-testid="stSidebar"] { background-color: #e0f2f1; width: 250px !important; }

        /* --- 🎯 HIDE Streamlit's Default Main Menu 🎯 --- */
        div[data-testid="stSidebarNav"] {
            display: none;
        }

        /* --- Custom Sidebar Buttons --- */
        div[data-testid="stSidebarContent"] .stButton > button {
            width: 100%;
            border: 1px solid transparent; /* No border for inactive */
            border-radius: 8px;
            background-color: transparent; /* Transparent background for inactive */
            color: #263238;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 8px;
            text-align: left; /* Align text to the left */
            padding: 10px 20px;
            transition: all 0.2s ease-in-out;
        }

        /* Style for button on hover */
        div[data-testid="stSidebarContent"] .stButton > button:hover {
            background-color: #cce8e6;
            color: #004d40;
            border-color: #cce8e6;
        }
        
        /* --- 🎯 Style for the ACTIVE button (using type="primary") 🎯 --- */
        div[data-testid="stSidebarContent"] .stButton > button[kind="primary"] {
            border: 1px solid #004d40;
            background-color: #00796b;
            color: white;
        }

        /* --- Other existing styles --- */
        .sidebar-footer { /* ... style as before ... */ }
        h1 { font-size: 38px !important; }
        .subtitle { font-style: italic; color: #2baf2b; font-size: 18px; }
        /* ... All other styles for tabs, feature boxes, etc. remain the same ... */

        </style>
        """,
        unsafe_allow_html=True
    )
