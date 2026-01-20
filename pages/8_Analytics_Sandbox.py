import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

# --- Page Config ---
st.set_page_config(page_title="Data Analytics Sandbox", page_icon="📈", layout="wide")

# --- Custom Style (เพื่อให้เข้ากับธีมเดิม) ---
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] > .main { background-color: #e0f2f1; }
    h1 { color: #263238; }
    .stDataFrame { background-color: white; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Data Analytics Sandbox")
st.markdown("พื้นที่สำหรับอัปโหลดข้อมูลและวิเคราะห์เบื้องต้นด้วยตัวเอง")

# --- 1. Upload Section ---
with st.container(border=True):
    uploaded_file = st.file_uploader("📂 อัปโหลดไฟล์ Excel หรือ CSV", type=['xlsx', 'csv'])

if uploaded_file:
    # --- Load Data ---
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success(f"✅ โหลดข้อมูลสำเร็จ: {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")
        
        # --- Tabs สำหรับฟังก์ชันต่างๆ ---
        tab_view, tab_viz, tab_audit = st.tabs(["👀 ดูข้อมูล & กรอง", "📊 สร้างกราฟ", "🔍 Audit Tools"])

        # === Tab 1: View Data ===
        with tab_view:
            st.subheader("ตารางข้อมูล")
            
            # Column Selection
            all_cols = df.columns.tolist()
            selected_cols = st.multiselect("เลือกคอลัมน์ที่ต้องการแสดง", all_cols, default=all_cols)
            
            # Simple Filter (ตัวอย่างกรอง 1 คอลัมน์)
            st.markdown("##### ตัวกรองเบื้องต้น")
            col_to_filter = st.selectbox("เลือกคอลัมน์ที่จะกรอง", ["(ไม่กรอง)"] + all_cols)
            if col_to_filter != "(ไม่กรอง)":
                unique_vals = df[col_to_filter].unique()
                val_to_filter = st.multiselect(f"เลือกค่าใน {col_to_filter}", unique_vals)
                if val_to_filter:
                    df_display = df[df[col_to_filter].isin(val_to_filter)][selected_cols]
                else:
                    df_display = df[selected_cols]
            else:
                df_display = df[selected_cols]
                
            st.dataframe(df_display, use_container_width=True)
            
            # สรุปสถิติ
            with st.expander("ดูสถิติเบื้องต้น (Describe)"):
                st.write(df_display.describe())

        # === Tab 2: Visualization ===
        with tab_viz:
            st.subheader("สร้างกราฟด้วยตัวเอง")
            c1, c2, c3 = st.columns(3)
            chart_type = c1.selectbox("ประเภทกราฟ", ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart"])
            x_axis = c2.selectbox("แกน X (แนวนอน)", df.columns)
            # พยายามหาคอลัมน์ตัวเลขมาเป็น default แกน Y
            num_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
            y_axis = c3.selectbox("แกน Y (แนวตั้ง/ค่า)", num_cols if num_cols else df.columns)
            
            if st.button("🚀 สร้างกราฟ"):
                if chart_type == "Bar Chart":
                    fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
                elif chart_type == "Line Chart":
                    fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
                elif chart_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Correlation: {x_axis} vs {y_axis}")
                elif chart_type == "Pie Chart":
                    fig = px.pie(df, names=x_axis, values=y_axis, title=f"Proportion of {y_axis} by {x_axis}")
                
                st.plotly_chart(fig, use_container_width=True)

        # === Tab 3: Audit Tools (Simple) ===
        with tab_audit:
            st.subheader("เครื่องมือช่วยตรวจสอบ")
            
            # 1. Sampling
            st.markdown("#### 🎲 สุ่มตัวอย่าง (Random Sampling)")
            sample_size = st.number_input("จำนวนที่ต้องการสุ่ม (แถว)", min_value=1, max_value=len(df), value=5)
            if st.button("สุ่มข้อมูล"):
                sampled_df = df.sample(n=sample_size)
                st.write(sampled_df)
                
            st.divider()
            
            # 2. Top N Analysis
            st.markdown("#### 🏆 จัดลำดับสูงสุด (Top N)")
            if num_cols:
                top_col = st.selectbox("เลือกคอลัมน์ตัวเลขที่จะจัดลำดับ", num_cols)
                top_n = st.slider("เอามากี่ลำดับ", 1, 20, 5)
                top_df = df.nlargest(top_n, top_col)
                st.bar_chart(top_df.set_index(df.columns[0])[top_col]) # ใช้คอลัมน์แรกเป็น Label
                st.write(top_df)
            else:
                st.info("ไม่พบคอลัมน์ตัวเลขในไฟล์นี้")

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")

else:
    st.info("👆 กรุณาอัปโหลดไฟล์เพื่อเริ่มต้นใช้งาน")
