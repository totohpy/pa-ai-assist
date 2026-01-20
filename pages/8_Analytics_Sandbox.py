import streamlit as st
import pandas as pd
from io import BytesIO

# --- Import PyGWalker ---
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer

# --- Page Config ---
st.set_page_config(page_title="Analytics Sandbox (Power BI Mode)", page_icon="📊", layout="wide")

# --- Custom Style ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] > .main { background-color: #f0f2f6; }
    h1 { color: #263238; }
    .stDataFrame { background-color: white; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Analytics Sandbox (Power BI Mode)")
st.markdown("วิเคราะห์ข้อมูลแบบ Drag & Drop เหมือน Power BI โดยไม่ต้องเขียนโค้ด")

# --- 1. Upload Section ---
with st.container(border=True):
    uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์ Excel หรือ CSV เพื่อเริ่มวิเคราะห์", type=['xlsx', 'csv'])

# --- Function to load data ---
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        return None

# --- Function to get PyGWalker Renderer (Cache Resource เพื่อความเร็ว) ---
@st.cache_resource
def get_pyg_renderer(dataframe):
    # สร้าง Renderer ของ PyGWalker
    return StreamlitRenderer(dataframe, spec="./gw_config.json", spec_io_mode="RW")

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.success(f"✅ โหลดข้อมูลสำเร็จ: {df.shape[0]} รายการ")

        # สร้าง Tabs แยกโหมดการทำงาน
        tab_bi, tab_audit, tab_raw = st.tabs(["🎨 Power BI Mode (Drag & Drop)", "🔍 Audit Tools", "📄 ดูข้อมูลดิบ"])

        # === TAB 1: Power BI Mode (PyGWalker) ===
        with tab_bi:
            st.info("💡 คำแนะนำ: ลากชื่อคอลัมน์ด้านซ้าย ไปวางในช่อง X-Axis หรือ Y-Axis เพื่อสร้างกราฟทันที")
            
            # เรียกใช้ PyGWalker
            renderer = get_pyg_renderer(df)
            renderer.explorer()

        # === TAB 2: Audit Tools (Tools เดิมที่มีประโยชน์) ===
        with tab_audit:
            st.subheader("🛠️ เครื่องมือช่วยตรวจสอบ (Audit Tools)")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🎲 สุ่มตัวอย่าง (Random Sampling)")
                sample_size = st.number_input("จำนวนที่ต้องการสุ่ม", min_value=1, max_value=len(df), value=min(10, len(df)))
                if st.button("สุ่มข้อมูล"):
                    sampled_df = df.sample(n=sample_size)
                    st.write(sampled_df)

            with c2:
                st.markdown("#### 🏆 จัดลำดับสูงสุด (Top N)")
                # หาคอลัมน์ตัวเลข
                num_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
                if num_cols:
                    top_col = st.selectbox("เลือกคอลัมน์ที่จะจัดลำดับ", num_cols)
                    top_n = st.slider("จำนวนลำดับ", 1, 20, 5)
                    st.write(df.nlargest(top_n, top_col))
                else:
                    st.warning("ไม่พบคอลัมน์ตัวเลข")

        # === TAB 3: Raw Data ===
        with tab_raw:
            st.dataframe(df, use_container_width=True)

    else:
        st.error("ไม่สามารถอ่านไฟล์ได้ กรุณาตรวจสอบรูปแบบไฟล์")
else:
    st.info("👆 กรุณาอัปโหลดไฟล์ด้านบนเพื่อเริ่มต้น")
