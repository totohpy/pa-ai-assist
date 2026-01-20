import streamlit as st
import pandas as pd
import sweetviz as sv
import streamlit.components.v1 as components
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer
import os

# --- Page Config ---
st.set_page_config(page_title="Super Analytics Sandbox", page_icon="🕵️", layout="wide")

# --- Custom Style ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] > .main { background-color: #f0f2f6; }
    h1 { color: #263238; }
    .stDataFrame { background-color: white; }
    /* ปรับแต่งปุ่มให้ดูเด่น */
    .stButton > button { border-radius: 8px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🕵️ Super Analytics Sandbox")
st.markdown("ศูนย์รวมเครื่องมือวิเคราะห์ข้อมูล: **Power BI Mode** (วิเคราะห์เอง) และ **Auto Report** (ให้ AI ช่วยวิเคราะห์)")

# --- 1. Upload Section ---
with st.container(border=True):
    uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์ Excel หรือ CSV เพื่อเริ่มงาน", type=['xlsx', 'csv'])

# --- Helper Functions ---
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        return None

@st.cache_resource
def get_pyg_renderer(dataframe):
    return StreamlitRenderer(dataframe, spec="./gw_config.json", spec_io_mode="RW")

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.success(f"✅ โหลดข้อมูลสำเร็จ: {df.shape[0]} รายการ | {len(df.columns)} คอลัมน์")
        
        # --- สร้าง Tabs แยกโหมดการทำงาน ---
        tab_bi, tab_sweetviz, tab_audit = st.tabs([
            "🎨 Power BI Mode (PyGWalker)", 
            "📑 Auto Report (Sweetviz)", 
            "🛠️ Audit Tools (Sampling)"
        ])

        # === TAB 1: PyGWalker (Power BI Style) ===
        with tab_bi:
            st.info("💡 **Tips:** ลากชื่อคอลัมน์ไปวางในแกน X/Y เพื่อสร้างกราฟ หรือกด 'Data' เพื่อดูข้อมูลดิบ")
            renderer = get_pyg_renderer(df)
            renderer.explorer()

        # === TAB 2: Sweetviz (Auto Audit Report) ===
        with tab_sweetviz:
            st.subheader("📑 สร้างรายงานวิเคราะห์อัตโนมัติ (X-Ray ข้อมูล)")
            st.markdown("ระบบจะสแกนข้อมูลทั้งหมดและสรุปค่าทางสถิติ, ค่าที่หายไป (Missing), และความผิดปกติให้ทันที")
            
            if st.button("🚀 เริ่มสร้างรายงาน Sweetviz", type="primary"):
                with st.spinner("กำลังสแกนข้อมูล... (อาจใช้เวลาสักครู่หากไฟล์ใหญ่)"):
                    try:
                        # 1. Analyze Data
                        report = sv.analyze(df)
                        
                        # 2. Save to HTML temporary file
                        report_path = "sweetviz_report.html"
                        report.show_html(report_path, open_browser=False, layout='vertical', scale=1.0)
                        
                        # 3. Read HTML back to display
                        with open(report_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        
                        # 4. Display in Streamlit
                        st.success("สร้างรายงานเสร็จสิ้น! เลื่อนลงเพื่อดูรายละเอียด")
                        components.html(html_content, height=1000, scrolling=True)
                        
                        # 5. Download Button
                        with open(report_path, "rb") as f:
                            st.download_button(
                                label="💾 ดาวน์โหลดไฟล์รายงาน (.html) ไปเปิดดูทีหลัง",
                                data=f,
                                file_name="audit_xray_report.html",
                                mime="text/html"
                            )
                            
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการสร้างรายงาน: {e}")

        # === TAB 3: Audit Tools ===
        with tab_audit:
            st.subheader("🛠️ เครื่องมือช่วยตรวจสอบเพิ่มเติม")
            
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown("#### 🎲 สุ่มตัวอย่าง (Random Sampling)")
                    st.caption("ใช้สำหรับสุ่มรายการเพื่อขอเอกสารตรวจสอบ")
                    sample_size = st.number_input("จำนวนที่ต้องการสุ่ม", min_value=1, max_value=len(df), value=min(10, len(df)))
                    if st.button("สุ่มข้อมูล"):
                        sampled_df = df.sample(n=sample_size)
                        st.dataframe(sampled_df)
                        
            with c2:
                with st.container(border=True):
                    st.markdown("#### 🏆 จัดลำดับสูงสุด (Top N)")
                    st.caption("หาโครงการที่ใช้งบเยอะสุด หรือมีความเสี่ยงสูง")
                    num_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
                    if num_cols:
                        top_col = st.selectbox("เลือกคอลัมน์ที่จะเรียง", num_cols)
                        top_n = st.slider("จำนวนลำดับ", 1, 50, 5)
                        st.dataframe(df.nlargest(top_n, top_col))
                    else:
                        st.warning("ไม่พบคอลัมน์ตัวเลข")

    else:
        st.error("ไม่สามารถอ่านไฟล์ได้")
else:
    st.info("👆 กรุณาอัปโหลดไฟล์ด้านบนเพื่อเริ่มต้น")
