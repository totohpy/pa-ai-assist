# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from io import BytesIO
# ... (import อื่นๆ เหมือนเดิม)

st.set_page_config(page_title="Design Assistant", page_icon="✨", layout="wide")

# ----------------- ⚙️ การตั้งค่ากลาง -----------------
with st.sidebar:
    # --- vvv ลบ st.info และลายเซ็นออกจากตรงนี้ vvv ---
    try:
        st.session_state.api_key_global = st.secrets["api_key"]
    except KeyError:
        st.session_state.api_key_global = ""
        st.warning("ฟีเจอร์ AI ยังไม่พร้อมใช้งาน กรุณาติดต่อผู้ดูแลระบบ")
    except Exception as e:
        st.session_state.api_key_global = ""
        st.error(f"เกิดข้อผิดพลาดในการโหลด API Key: {e}")
    # --- ^^^ สิ้นสุดการลบ ^^^ ---

# (โค้ดส่วนที่เหลือของไฟล์นี้ไม่ต้องเปลี่ยนแปลง)
# ...
