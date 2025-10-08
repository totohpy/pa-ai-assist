# นี่คือเนื้อหาทั้งหมดของไฟล์ style.py
import streamlit as st

def load_css():
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
            color: #263238 !important;     /* Inactive link text color */
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

        /* --- Custom Tab Styling on Home Page --- */
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
        </style>
        """,
        unsafe_allow_html=True
    )
