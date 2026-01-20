import streamlit as st
import pandas as pd
import sweetviz as sv
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer

# --- Page Config ---
st.set_page_config(page_title="Super Analytics Sandbox", page_icon="🕵️", layout="wide")

# --- Custom Style ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] > .main { background-color: #f0f2f6; }
    h1 { color: #263238; }
    .stDataFrame { background-color: white; }
    .stButton > button { border-radius: 8px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🕵️ Super Analytics Sandbox")
st.markdown("เครื่องมือวิเคราะห์ข้อมูลครบวงจร: **Power BI Mode** (กราฟ) และ **Deep Scan** (ตรวจสอบคุณภาพข้อมูล)")

# --- 1. Upload Section ---
with st.container(border=True):
    uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์ Excel หรือ CSV", type=['xlsx', 'csv'])

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
        
        # สร้าง Tabs
        tab_bi, tab_ydata, tab_sweetviz, tab_audit = st.tabs([
            "🎨 Power BI Mode", 
            "🔬 Deep Scan (YData)", 
            "📑 Quick Report (Sweetviz)",
            "🛠️ Audit Tools"
        ])

        # === TAB 1: PyGWalker (Power BI Mode) ===
        with tab_bi:
            st.info("💡 **Power BI Mode:** ลากชื่อคอลัมน์ซ้ายมือ มาวางในแกน X / Y เพื่อสร้างกราฟ")
            renderer = get_pyg_renderer(df)
            renderer.explorer()

        # === TAB 2: YData Profiling (แทน D-Tale) ===
        with tab_ydata:
            st.subheader("🔬 Deep Data Profiling (YData)")
            st.markdown("วิเคราะห์ข้อมูลเชิงลึก เหมาะสำหรับตรวจสอบความผิดปกติ, ความสัมพันธ์ (Correlation) และคุณภาพข้อมูล")
            
            if st.button("🚀 เริ่มวิเคราะห์เจาะลึก (Deep Scan)", type="primary"):
                with st.spinner("กำลังประมวลผล... (ระบบนี้ละเอียดมาก อาจใช้เวลา 1-2 นาที)"):
                    try:
                        # สร้าง Profile Report
                        pr = ProfileReport(df, explorative=True, title="Audit Data Profiling")
                        
                        # บันทึกเป็น HTML ชั่วคราว
                        report_path = "ydata_report.html"
                        pr.to_file(report_path)
                        
                        # อ่านไฟล์ HTML กลับมาแสดง
                        with open(report_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        
                        st.success("วิเคราะห์เสร็จสิ้น!")
                        components.html(html_content, height=1000, scrolling=True)
                        
                        # ปุ่มดาวน์โหลด
                        with open(report_path, "rb") as f:
                            st.download_button(
                                label="💾 ดาวน์โหลดรายงานฉบับเต็ม (.html)",
                                data=f,
                                file_name="Deep_Audit_Report.html",
                                mime="text/html"
                            )
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาด: {e}")

        # === TAB 3: Sweetviz (Quick Scan) ===
        with tab_sweetviz:
            st.subheader("📑 Quick Scan Report (Sweetviz)")
            st.markdown("รายงานเปรียบเทียบข้อมูลแบบรวดเร็ว อ่านง่าย สบายตา")
            
            if st.button("🚀 สร้างรายงานด่วน (Quick Scan)"):
                with st.spinner("กำลังสร้างรายงาน..."):
                    report = sv.analyze(df)
                    report.show_html("sweetviz_report.html", open_browser=False, layout='vertical', scale=1.0)
                    
                    with open("sweetviz_report.html", 'r', encoding='utf-8') as f:
                        components.html(f.read(), height=1000, scrolling=True)

        # === TAB 4: Audit Tools (Tools เดิม) ===
        with tab_audit:
            st.subheader("🛠️ เครื่องมือสุ่มและกรองข้อมูล")
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.markdown("#### 🎲 สุ่มตัวอย่าง (Sampling)")
                    sample_size = st.number_input("จำนวนสุ่ม", 1, len(df), 5)
                    if st.button("สุ่มข้อมูล"):
                        st.dataframe(df.sample(sample_size))
            with c2:
                with st.container(border=True):
                    st.markdown("#### 🏆 Top N Ranking")
                    num_cols = df.select_dtypes(include=['number']).columns
                    if not num_cols.empty:
                        col = st.selectbox("เรียงตาม", num_cols)
                        st.dataframe(df.nlargest(5, col))
                    else:
                        st.warning("ไม่พบคอลัมน์ตัวเลข")

    else:
        st.error("ไม่สามารถอ่านไฟล์ได้")
else:
    st.info("👆 กรุณาอัปโหลดไฟล์เพื่อเริ่มต้น")
