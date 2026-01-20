import streamlit as st
import pandas as pd
import dtale
import sweetviz as sv
import streamlit.components.v1 as components
from dtale.app import get_instance

# --- Page Config ---
st.set_page_config(page_title="Deep Analytics (D-Tale)", page_icon="🔬", layout="wide")

# --- Custom Style ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] > .main { background-color: #f0f2f6; }
    h1 { color: #263238; }
    .stButton > button { border-radius: 8px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🔬 Deep Analytics Sandbox")
st.markdown("วิเคราะห์ข้อมูลเชิงลึกด้วย **D-Tale** (เครื่องมือที่ละเอียดที่สุดสำหรับ Data Scientist)")

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

def startup_dtale(data):
    # เริ่มต้น D-Tale instance
    d = dtale.show(data, host='localhost')
    return d

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.success(f"✅ โหลดข้อมูลสำเร็จ: {df.shape[0]} รายการ | {len(df.columns)} คอลัมน์")
        
        # --- สร้าง Tabs ---
        tab_dtale, tab_sweetviz, tab_audit = st.tabs([
            "🔬 D-Tale (Deep Analysis)", 
            "📑 Auto Report (Sweetviz)", 
            "🛠️ Audit Tools"
        ])

        # === TAB 1: D-Tale ===
        with tab_dtale:
            st.info("💡 **D-Tale** คือเครื่องมือที่ทรงพลังมาก สามารถแก้ไขข้อมูล กรอง และวิเคราะห์สถิติขั้นสูงได้")
            
            # ปุ่มเริ่มรัน D-Tale
            if st.button("🚀 เปิด D-Tale Analysis", type="primary"):
                with st.spinner("กำลังเริ่มระบบ D-Tale..."):
                    # Start D-Tale
                    d = startup_dtale(df)
                    
                    # ดึง URL (เนื่องจาก D-Tale รันคนละ Port เราต้องเปิดหน้าใหม่)
                    dtale_url = d.main_url()
                    
                    st.markdown("---")
                    st.success("D-Tale พร้อมใช้งานแล้ว!")
                    
                    # แสดงลิงก์ให้กดเปิด (วิธีนี้เสถียรที่สุดบน Streamlit Cloud)
                    st.markdown(f'''
                        <a href="{dtale_url}" target="_blank" style="text-decoration: none;">
                            <button style="
                                background-color: #FF4B4B;
                                color: white;
                                padding: 10px 24px;
                                border: none;
                                border-radius: 8px;
                                cursor: pointer;
                                font-size: 16px;
                                font-weight: bold;">
                                🌐 คลิกเพื่อเปิดหน้าต่าง D-Tale (Full Screen)
                            </button>
                        </a>
                    ''', unsafe_allow_html=True)
                    
                    st.warning("หมายเหตุ: หากรันบน Server/Cloud บางแห่ง อาจต้องตั้งค่า Port เพิ่มเติม หากเปิดไม่ได้ให้ลองใช้ Auto Report แทน")
                    
                    # ลอง Embed iframe (เผื่อใช้ได้ใน Local)
                    with st.expander("หรือลองดูในหน้าต่างนี้ (Embed View)"):
                        components.iframe(dtale_url, height=800, scrolling=True)

        # === TAB 2: Sweetviz (เหมือนเดิม) ===
        with tab_sweetviz:
            st.subheader("📑 สร้างรายงานสรุปผล (X-Ray)")
            if st.button("🚀 สร้างรายงาน Sweetviz"):
                with st.spinner("กำลังสร้างรายงาน..."):
                    report = sv.analyze(df)
                    report.show_html("sweetviz_report.html", open_browser=False)
                    with open("sweetviz_report.html", 'r', encoding='utf-8') as f:
                        components.html(f.read(), height=1000, scrolling=True)

        # === TAB 3: Audit Tools (เหมือนเดิม) ===
        with tab_audit:
            st.subheader("🛠️ เครื่องมือสุ่มและกรองข้อมูล")
            c1, c2 = st.columns(2)
            with c1:
                sample_size = st.number_input("จำนวนสุ่ม", 1, len(df), 5)
                if st.button("สุ่มข้อมูล"):
                    st.dataframe(df.sample(sample_size))
            with c2:
                num_cols = df.select_dtypes(include=['number']).columns
                if not num_cols.empty:
                    col = st.selectbox("เรียงลำดับตาม", num_cols)
                    st.dataframe(df.nlargest(5, col))

    else:
        st.error("อ่านไฟล์ไม่ได้")
else:
    st.info("👆 กรุณาอัปโหลดไฟล์")
