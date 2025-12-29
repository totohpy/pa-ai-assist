import streamlit as st
import streamlit.components.v1 as components

# ตั้งค่าหน้าเว็บให้กว้าง (Optional: เพื่อให้แสดงผล Dashboard ได้เต็มตาขึ้น)
st.set_page_config(layout="wide", page_title="Environmental Audit Dashboard")

st.title("Environmental Audit Dashboard")

# โค้ด HTML iframe ที่คุณให้มา
# แนะนำให้เปลี่ยน width="1024" เป็น width="100%" เพื่อให้ยืดหดตามหน้าจอ
iframe_code = """
<iframe title="Envi_Audit_SAO2026"
width="100%" height="612"
src="https://app.powerbi.com/view?r=eyJrIjoiMzBmODQ2MTgtMGYwMy00NTc3LWI4ZTAtOWE1NzY3MjRkMGMwIiwidCI6ImI3NWFiN2IzLTU4YmEtNGZkNy1iYTU1LTMyNmY0ZWRmYzllOSIsImMiOjEwfQ%3D%3D"
frameborder="0" allowFullScreen="true"></iframe>
"""

# การแสดงผล
# ต้องกำหนด height ในฟังก์ชัน python ให้ตรงหรือมากกว่าใน iframe เพื่อไม่ให้เกิด scrollbar ซ้อน
components.html(iframe_code, height=612)
