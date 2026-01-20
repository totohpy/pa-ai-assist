import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Config ---
st.set_page_config(page_title="Custom Dashboard Builder", page_icon="🧱", layout="wide")

# --- Custom Style ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] > .main { background-color: #f8f9fa; }
    .block-container { padding-top: 2rem; }
    .stButton > button { border-radius: 8px; }
    div[data-testid="stExpander"] { background-color: white; border-radius: 10px; border: 1px solid #ddd; }
</style>
""", unsafe_allow_html=True)

st.title("🧱 สร้าง Dashboard ด้วยตัวเอง (Dashboard Builder)")
st.markdown("อัปโหลดไฟล์แล้วกด **'เพิ่มกราฟ'** เพื่อจัดวางหน้าจอ Dashboard ของคุณเอง")

# --- Session State สำหรับเก็บกราฟที่ User สร้าง ---
if 'dashboard_charts' not in st.session_state:
    st.session_state.dashboard_charts = []

# --- 1. Upload Section ---
with st.sidebar:
    st.header("1. ข้อมูลตั้งต้น")
    uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์ Excel/CSV", type=['xlsx', 'csv'])
    
    if st.button("🗑️ ล้าง Dashboard ทั้งหมด", type="primary"):
        st.session_state.dashboard_charts = []
        st.rerun()

# --- Helper Function ---
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'): return pd.read_csv(file)
        else: return pd.read_excel(file)
    except: return None

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        # --- ส่วนควบคุมการเพิ่มกราฟ ---
        with st.expander("➕ เพิ่มกราฟใหม่ (คลิกที่นี่)", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            chart_type = c1.selectbox("ประเภทกราฟ", ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Area Chart"])
            x_col = c2.selectbox("แกน X (กลุ่มข้อมูล)", df.columns)
            
            # กรองเฉพาะคอลัมน์ตัวเลขสำหรับแกน Y
            num_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
            y_col = c3.selectbox("แกน Y (ค่าตัวเลข)", num_cols if num_cols else df.columns)
            
            chart_title = c4.text_input("ชื่อกราฟ", value=f"{chart_type} of {y_col} by {x_col}")
            
            if st.button("✅ ยืนยันเพิ่มกราฟ"):
                # บันทึกการตั้งค่ากราฟลง Session State
                new_chart = {
                    "id": len(st.session_state.dashboard_charts) + 1,
                    "type": chart_type,
                    "x": x_col,
                    "y": y_col,
                    "title": chart_title
                }
                st.session_state.dashboard_charts.append(new_chart)
                st.success("เพิ่มกราฟเรียบร้อย!")
                st.rerun()

        st.divider()

        # --- แสดงผล Dashboard (Loop ตามรายการที่ User เพิ่มมา) ---
        if not st.session_state.dashboard_charts:
            st.info("👆 ยังไม่มีกราฟ กดที่ 'เพิ่มกราฟใหม่' ด้านบนเพื่อเริ่มต้นสร้าง Dashboard ของคุณ")
        
        # จัดวางกราฟแบบ Grid (2 กราฟต่อ 1 แถว)
        for i in range(0, len(st.session_state.dashboard_charts), 2):
            cols = st.columns(2)
            
            # ดึงกราฟทีละคู่ (ซ้าย, ขวา)
            batch = st.session_state.dashboard_charts[i:i+2]
            
            for idx, chart_config in enumerate(batch):
                with cols[idx]:
                    with st.container(border=True):
                        st.subheader(chart_config['title'])
                        
                        # สร้างกราฟตาม Config
                        try:
                            if chart_config['type'] == "Bar Chart":
                                fig = px.bar(df, x=chart_config['x'], y=chart_config['y'])
                            elif chart_config['type'] == "Line Chart":
                                fig = px.line(df, x=chart_config['x'], y=chart_config['y'])
                            elif chart_config['type'] == "Pie Chart":
                                fig = px.pie(df, names=chart_config['x'], values=chart_config['y'])
                            elif chart_config['type'] == "Scatter Plot":
                                fig = px.scatter(df, x=chart_config['x'], y=chart_config['y'])
                            elif chart_config['type'] == "Area Chart":
                                fig = px.area(df, x=chart_config['x'], y=chart_config['y'])
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # ปุ่มลบกราฟ
                            if st.button("❌ ลบ", key=f"del_{i+idx}"):
                                st.session_state.dashboard_charts.pop(i+idx)
                                st.rerun()
                        except Exception as e:
                            st.error(f"ไม่สามารถแสดงกราฟได้: {e}")

    else:
        st.error("อ่านไฟล์ไม่ได้")
else:
    st.info("⬅️ กรุณาอัปโหลดไฟล์ที่ Sidebar ด้านซ้าย")
