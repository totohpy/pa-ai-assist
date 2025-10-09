import streamlit as st

def load_css():
    st.markdown(
        """
        <style>
        /* --- General Theme & Sidebar Base --- */
        [data-testid="stAppViewContainer"] > .main { background-color: #e0f2f1; }
        .block-container { padding-top: 2rem; }
        [data-testid="stSidebar"] { background-color: #e0f2f1; width: 280px !important; }

        /* --- ซ่อนเมนูหลักของ Streamlit --- */
        div[data-testid="stSidebarNav"] {
            display: none;
        }

        /* --- ดีไซน์ปุ่มเมนูที่เราสร้างเองใน Sidebar --- */
        div[data-testid="stSidebarContent"] .stButton > button {
            width: 100%;
            border: 1px solid #b2dfdb; /* เพิ่มขอบสีอ่อน */
            border-radius: 8px;
            background-color: transparent;
            color: #263238;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 8px;
            text-align: left;
            padding: 10px 20px;
            transition: all 0.2s ease-in-out;
        }

        /* ดีไซน์ปุ่มเมื่อเอาเมาส์ไปวาง (Hover) */
        div[data-testid="stSidebarContent"] .stButton > button:hover {
            background-color: #cce8e6;
            color: #004d40;
            border-color: #80cbc4;
        }

        /* --- ดีไซน์สำหรับปุ่มที่กำลังถูกเลือก (Active) --- */
        /* This targets the button when its type is "primary" */
        div[data-testid="stSidebarContent"] .stButton > button.st-emotion-cache-19n6ohb,
        div[data-testid="stSidebarContent"] .stButton > button:focus:not(:hover) {
            border: 1px solid #004d40;
            background-color: #00796b;
            color: white;
        }

        /* --- Footer ใน Sidebar --- */
        .sidebar-footer {
            width: 100%;
            padding: 1rem;
            text-align: center;
            position: absolute;
            bottom: 0;
            left: 0;
        }

        /* --- สไตล์อื่นๆ (คงเดิม) --- */
        h1 { font-size: 38px !important; }
        .subtitle { font-style: italic; color: #2baf2b; font-size: 18px; }
        .feature-box {
            background-color: #ffffff; padding: 2rem 1.5rem; border-radius: 20px;
            text-align: center; transition: transform 0.3s, box-shadow 0.3s;
            height: 250px; display: flex; flex-direction: column;
            justify-content: center; align-items: center; border: 1px solid #d0e0df;
        }
        .feature-box:hover { transform: translateY(-10px); box-shadow: 0 8px 30px rgba(0,0,0,0.12); }
        </style>
        """,
        unsafe_allow_html=True
    )

