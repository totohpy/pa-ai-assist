import streamlit as st
import pandas as pd
import sweetviz as sv
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns # เพิ่ม seaborn เพื่อบังคับ style

# --- Page Config ---
st.set_page_config(page_title="Super Analytics Sandbox", page_icon="🕵️", layout="wide")

# --- Custom Style ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] > .main { background-color: #f0f2f6; }
    h1, h2, h3, p, div { font-family: 'Sarabun', sans-serif !important; } 
    .stDataFrame { background-color: white; }
</style>
""", unsafe_allow_html=True)

# --- 🛠️ FIX v2: ระบบค้นหาและบังคับใช้ Font ภาษาไทย (แบบละเอียด) ---
def setup_thai_font_robust():
    # ลิสต์รายชื่อที่อาจจะเป็นที่อยู่ของไฟล์ (หาทั้งในโฟลเดอร์ปัจจุบัน และถอยออกไป 1 ชั้น)
    possible_paths = [
        "Sarabun-Regular.ttf",          # กรณีอยู่โฟลเดอร์เดียวกับไฟล์นี้
        "../Sarabun-Regular.ttf",       # กรณีไฟล์นี้อยู่ใน pages/ แล้ว font อยู่ข้างนอก
        "./Sarabun-Regular.ttf",
        "Sarabun-Bold.ttf",
        "../Sarabun-Bold.ttf"
    ]
    
    font_path = None
    for path in possible_paths:
        if os.path.exists(path):
            font_path = path
            break
            
    if font_path:
        # 1. Add Font to Matplotlib Manager
        fm.fontManager.addfont(font_path)
        
        # 2. Get Font Name (ชื่อจริงๆ ของ Font หลังจากแอดแล้ว)
        prop = fm.FontProperties(fname=font_path)
        font_name = prop.get_name()
        
        # 3. Force Global Settings (บังคับทุกจุด)
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False
        sns.set(font=font_name) # บังคับ Seaborn ด้วย
        
        return font_name, True # ส่งชื่อฟอนต์กลับไปใช้ต่อ
    else:
        return None, False

# เรียกใช้ฟังก์ชันทันที
thai_font_name, font_found = setup_thai_font_robust()

# แจ้งเตือน User ถ้าหาไม่เจอจริงๆ
if not font_found:
    st.toast("⚠️ ไม่พบไฟล์ฟอนต์ Sarabun-Regular.ttf ภาษาไทยในกราฟอาจแสดงผลผิดพลาด", icon="⚠️")
# -----------------------------------------------------------------------

st.title("🕵️ Super Analytics Sandbox")
st.markdown("เครื่องมือวิเคราะห์ข้อมูลครบวงจร")

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
        st.success(f"✅ โหลดข้อมูลสำเร็จ: {df.shape[0]} รายการ")
        
        # สร้าง Tabs
        tab_bi, tab_ydata, tab_sweetviz, tab_audit = st.tabs([
            "🎨 Power BI Mode", 
            "🔬 Deep Scan (YData)", 
            "📑 Quick Report",
            "🛠️ Audit Tools"
        ])

        # === TAB 1: PyGWalker ===
        with tab_bi:
            renderer = get_pyg_renderer(df)
            renderer.explorer()

        # === TAB 2: YData Profiling (จุดที่แก้ปัญหา) ===
        with tab_ydata:
            st.subheader("🔬 Deep Data Profiling (YData)")
            
            if st.button("🚀 เริ่มวิเคราะห์เจาะลึก (Deep Scan)", type="primary"):
                with st.spinner("กำลังประมวลผล..."):
                    try:
                        # Re-apply font settings before generating
                        if font_found:
                            plt.rcParams['font.family'] = thai_font_name
                            sns.set(font=thai_font_name)

                        # สร้าง Profile Report โดยระบุ Font เข้าไปใน Plot config โดยตรง
                        pr = ProfileReport(
                            df, 
                            explorative=True,
                            title="Audit Data Profiling",
                            plot={'font': {'family': thai_font_name}} if font_found else {}, # บังคับตรงนี้อีกที
                        )
                        
                        # Save & Show
                        report_path = "ydata_report.html"
                        pr.to_file(report_path)
                        
                        with open(report_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        
                        st.success("วิเคราะห์เสร็จสิ้น!")
                        components.html(html_content, height=1000, scrolling=True)
                        
                        with open(report_path, "rb") as f:
                            st.download_button("💾 ดาวน์โหลดรายงาน", f, "Deep_Audit_Report.html", "text/html")
                            
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาด: {e}")

        # === TAB 3: Sweetviz ===
        with tab_sweetviz:
            st.subheader("📑 Quick Scan Report")
            if st.button("🚀 สร้างรายงานด่วน"):
                report = sv.analyze(df)
                report.show_html("sweetviz_report.html", open_browser=False, layout='vertical', scale=1.0)
                with open("sweetviz_report.html", 'r', encoding='utf-8') as f:
                    components.html(f.read(), height=1000, scrolling=True)

        # === TAB 4: Audit Tools ===
        with tab_audit:
            st.subheader("🛠️ Audit Tools")
            c1, c2 = st.columns(2)
            with c1:
                sample_size = st.number_input("จำนวนสุ่ม", 1, len(df), 5)
                if st.button("สุ่มข้อมูล"):
                    st.dataframe(df.sample(sample_size))
            with c2:
                num_cols = df.select_dtypes(include=['number']).columns
                if not num_cols.empty:
                    col = st.selectbox("เรียงตาม", num_cols)
                    st.dataframe(df.nlargest(5, col))
    else:
        st.error("ไม่สามารถอ่านไฟล์ได้")
else:
    st.info("👆 กรุณาอัปโหลดไฟล์")
