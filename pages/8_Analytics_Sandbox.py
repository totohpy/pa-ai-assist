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
import seaborn as sns 

# --- Page Config ---
st.set_page_config(page_title="Super Analytics Sandbox", page_icon="🕵️", layout="wide")

# --- Custom Style ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] > .main { background-color: #f0f2f6; }
    h1, h2, h3, p, div, span { font-family: 'Sarabun', sans-serif !important; } 
    .stDataFrame { background-color: white; }
</style>
""", unsafe_allow_html=True)

# --- 🛠️ FIX v4: ระบบบังคับ Font (Final Boss Edition) ---
def setup_thai_font_final():
    # 1. หาไฟล์ฟอนต์ (เหมือนเดิม)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    font_paths = [
        os.path.join(parent_dir, "Sarabun-Regular.ttf"),
        os.path.join(parent_dir, "Sarabun-Bold.ttf"),
        "Sarabun-Regular.ttf"
    ]
    
    found_path = None
    for p in font_paths:
        if os.path.exists(p):
            found_path = p
            break
            
    if found_path:
        # 2. Add Font
        fm.fontManager.addfont(found_path)
        prop = fm.FontProperties(fname=found_path)
        font_name = prop.get_name() # ได้ชื่อเช่น 'Sarabun'
        
        # 3. 🔥 บังคับค่า Global Matplotlib (จุดสำคัญ)
        # ตั้งค่า Family เป็น sans-serif
        plt.rcParams['font.family'] = 'sans-serif' 
        # แล้วยัดชื่อฟอนต์ไทยไว้ *ตัวแรกสุด* ของรายการ sans-serif
        # (วิธีนี้ Matplotlib จะหยิบตัวแรกมาใช้เสมอ แม้จะโดนรีเซ็ต family)
        plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 4. บังคับ Seaborn
        sns.set_theme(font=font_name)
        
        return font_name, True
    else:
        return None, False

# เรียกใช้ทันที
thai_font_name, font_found = setup_thai_font_final()

if font_found:
    st.toast(f"✅ บังคับใช้ฟอนต์: {thai_font_name}", icon="🇹🇭")
else:
    st.error("⚠️ ไม่พบไฟล์ฟอนต์ Sarabun-Regular.ttf ใน Project")

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
        if file.name.endswith('.csv'): return pd.read_csv(file)
        else: return pd.read_excel(file)
    except: return None

@st.cache_resource
def get_pyg_renderer(dataframe):
    return StreamlitRenderer(dataframe, spec="./gw_config.json", spec_io_mode="RW")

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.success(f"✅ โหลดข้อมูลสำเร็จ: {df.shape[0]} รายการ")
        
        tab_bi, tab_ydata, tab_sweetviz, tab_audit = st.tabs([
            "🎨 Power BI Mode", 
            "🔬 Deep Scan (YData)", 
            "📑 Quick Report",
            "🛠️ Audit Tools"
        ])

        with tab_bi:
            renderer = get_pyg_renderer(df)
            renderer.explorer()

        # === TAB 2: YData Profiling (Final Fix) ===
        with tab_ydata:
            st.subheader("🔬 Deep Data Profiling (YData)")
            
            if st.button("🚀 เริ่มวิเคราะห์เจาะลึก (Deep Scan)", type="primary"):
                with st.spinner("กำลังประมวลผล..."):
                    try:
                        # Re-Execute Font Setup (กันเหนียว)
                        if font_found:
                            plt.rcParams['font.family'] = 'sans-serif'
                            plt.rcParams['font.sans-serif'] = [thai_font_name] + plt.rcParams['font.sans-serif']
                            sns.set_theme(font=thai_font_name)

                        # 🔥 ส่ง Config เข้าไปใน ProfileReport โดยตรง
                        pr = ProfileReport(
                            df, 
                            explorative=True,
                            title="Audit Data Profiling",
                            plot={
                                'dpi': 200,
                                'image_format': 'png',
                                'font': {
                                    'family': 'sans-serif',
                                    'sans-serif': [thai_font_name] # ย้ำตรงนี้อีกที
                                }
                            }
                        )
                        
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

        with tab_sweetviz:
            st.subheader("📑 Quick Scan Report")
            if st.button("🚀 สร้างรายงานด่วน"):
                report = sv.analyze(df)
                report.show_html("sweetviz_report.html", open_browser=False, layout='vertical', scale=1.0)
                with open("sweetviz_report.html", 'r', encoding='utf-8') as f:
                    components.html(f.read(), height=1000, scrolling=True)

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
