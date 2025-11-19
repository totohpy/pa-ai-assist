import streamlit as st
import qrcode
from io import BytesIO
from PIL import Image
import os
import base64

# --- 1. ตั้งค่า Page Config ---
st.set_page_config(
    page_title="QR Code Generator",
    page_icon="📱",
    layout="wide"
)

# --- 2. Custom CSS (Styles) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
    }
    
    [data-testid="stAppViewContainer"] > .main {
        background-color: #e0f2f1; 
    }
    
    h1 { 
        font-size: 36px !important; 
        color: #263238; 
        font-weight: 700;
    }
    h2, h3 {
        color: #263238;
        font-weight: 700;
    }
    
    [data-testid="stSidebar"] {
        background-color: #e0f2f1;
        width: 250px !important;
        border-right: 1px solid #b2dfdb;
    }

    [data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    [data-testid="stSidebarNav"] {
        flex-grow: 1;
        margin-top: 20px;
    }
    .sidebar-footer {
        width: 100%;
        padding: 1rem;
        text-align: center;
    }

    .block-container {
        padding-top: 2rem;
    }
    
    div[data-testid="stSidebarNav"] > ul > li > a {
        padding: 18px 40px !important;
        font-size: 20px !important;
        margin-bottom: 10px;
        border-radius: 8px;
        color: #263238 !important;
        background-color: #b2dfdb;
        border: 1px solid #9dbdb9;
        font-weight: 500;
    }
    div[data-testid="stSidebarNav"] a[aria-current="page"] {
        background-color: #80cbc4;
        color: #FFFFFF !important;
        font-weight: 600;
        border: 1px solid #00796b;
    }

    .stButton > button {
        background-color: #2563EB;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .result-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #9dbdb9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-top: 10px;
    }

    /* Style สำหรับ Card เลือกโลโก้ */
    .logo-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        background-color: white;
        height: 100%;
    }
    .logo-selected {
        border: 2px solid #2563EB;
        background-color: #eff6ff;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Helper Functions ---

def generate_qr_code_with_logo(data, logo_file_name=None):
    """สร้าง QR Code จากข้อมูลที่กำหนด และใส่ Logo ตรงกลาง (ถ้ามี)"""
    # ใช้ Error Correction Level High (H) เพื่อรองรับการวาง Logo โดยข้อมูลไม่เสียหาย
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    
    # ถ้ามีการเลือก Logo
    if logo_file_name:
        try:
            # ตรวจสอบว่าไฟล์มีอยู่จริงหรือไม่
            if os.path.exists(logo_file_name):
                logo = Image.open(logo_file_name)
                
                # คำนวณขนาด Logo (ปรับให้ใหญ่ขึ้นเป็น ~30% ของความกว้าง QR Code)
                width, height = img.size
                logo_size = int(width / 3.3) 
                logo = logo.resize((logo_size, logo_size))
                
                # คำนวณตำแหน่งวางตรงกลาง
                pos = ((width - logo_size) // 2, (height - logo_size) // 2)
                
                # วาง Logo ลงไป
                img.paste(logo, pos)
            else:
                print(f"Logo file not found: {logo_file_name}")
        except Exception as e:
            print(f"Error loading logo: {e}")

    # Convert to bytes for Streamlit
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def get_image_base64(image_path):
    """ฟังก์ชันช่วยแปลงรูปภาพเป็น Base64 สำหรับแสดงผลใน HTML"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception:
        return None

# --- 4. Sidebar Section ---
with st.sidebar:
    try:
        st.image("image_e05e9c.png", use_column_width=True) 
    except:
        pass 

    # Sidebar Footer
    st.markdown("""
        <div class="sidebar-footer">
            <p>
                <span style="color: grey;">By PAO1 </span><br>
                <span style="font-size: 16px; letter-spacing: 0.5px;">
                    <span style="color: red; font-weight: bold;">A</span>udit 
                    <span style="color: red; font-weight: bold;">I</span>ntelligence
                    <span style="color: red; font-weight: bold;">T</span>eam
                </span>
            </p>
        </div>
    """, unsafe_allow_html=True)

# --- 5. Main Content ---
st.title("📱 QR Code Generator")
st.markdown("##### เครื่องมือสร้างคิวอาร์โค้ดพร้อมโลโก้หน่วยงาน")

col1, col2 = st.columns([1.2, 0.8], gap="large")

with col1:
    st.info("✏️ **ข้อมูลสำหรับสร้าง QR**")
    
    qr_data = st.text_input("ใส่ URL หรือข้อความที่ต้องการ:", placeholder="https://www.example.com")
    
    st.write("")
    st.markdown("**เลือกรูปแบบโลโก้:**")

    # --- Logo Selection UI แบบรูปภาพ ---
    # ใช้ Session State เพื่อเก็บค่าที่เลือก
    if 'selected_logo_key' not in st.session_state:
        st.session_state['selected_logo_key'] = 'none'

    logo_cols = st.columns([1, 1, 1])
    
    # ฟังก์ชันช่วยแสดงผลตัวเลือก
    def render_option(col, key, label, image_path=None, is_no_logo=False):
        with col:
            is_selected = (st.session_state['selected_logo_key'] == key)
            
            # ใช้สัญลักษณ์แทนตัวเลข
            icon = "🔘" if is_selected else "⚪"
            st.markdown(f"<div style='text-align:center; margin-bottom:5px; font-weight:bold; color: #263238;'>{icon} {label}</div>", unsafe_allow_html=True)
            
            if is_no_logo:
                st.markdown("<div style='height:100px; border:1px dashed #ccc; display:flex; align-items:center; justify-content:center; color:#aaa; border-radius:8px; background:white; margin-bottom:10px;'>No Logo</div>", unsafe_allow_html=True)
            elif image_path and os.path.exists(image_path):
                img_b64 = get_image_base64(image_path)
                if img_b64:
                    st.markdown(
                        f"""<div style='height:100px; display:flex; align-items:center; justify-content:center; margin-bottom:10px; background:white; border-radius:8px; border: 1px solid #eee;'>
                            <img src="data:image/png;base64,{img_b64}" style="max-height:100px; max-width:100%;">
                        </div>""", 
                        unsafe_allow_html=True
                    )
            else:
                 st.warning("ไม่พบไฟล์")

            # ปุ่มเลือก (เปลี่ยนข้อความปุ่มเพื่อความเรียบง่าย)
            if is_selected:
                st.button("เลือกแล้ว", key=f"btn_{key}", type="primary", disabled=True, use_container_width=True)
            else:
                if st.button("เลือก", key=f"btn_{key}_select", use_container_width=True):
                    st.session_state['selected_logo_key'] = key
                    st.rerun()

    # 1. ไม่ใส่โลโก้
    render_option(logo_cols[0], 'none', 'ไม่ใส่โลโก้', is_no_logo=True)

    # 2. โลโก้ขาว-ดำ
    render_option(logo_cols[1], 'bw', 'โลโก้ขาว-ดำ', image_path="logoSAO-BW-TH_0.png")

    # 3. โลโก้สี
    render_option(logo_cols[2], 'color', 'โลโก้สี', image_path="logoSAO-TH-02.png")

    # Map Key to Filename
    logo_file_map = {
        "none": None,
        "bw": "logoSAO-BW-TH_0.png",
        "color": "logoSAO-TH-02.png"
    }
    selected_logo_file = logo_file_map[st.session_state['selected_logo_key']]
    # ---------------------------------------------------

    st.write("")
    st.markdown("---")
    if st.button("🚀 สร้าง QR Code", type="primary", use_container_width=True):
        if qr_data:
            with st.spinner("กำลังสร้าง..."):
                # สร้าง QR Code จากข้อมูลที่กรอกโดยตรง
                qr_image = generate_qr_code_with_logo(qr_data, selected_logo_file)
                
                # เก็บผลลัพธ์ลง Session State
                st.session_state['gen_qr_image'] = qr_image
                st.session_state['gen_qr_data'] = qr_data
                
                st.success("✅ สร้างสำเร็จ!")
        else:
            st.warning("กรุณาใส่ข้อมูลก่อนครับ")

with col2:
    # ตรวจสอบว่ามีการสร้างผลลัพธ์หรือยัง
    if 'gen_qr_image' in st.session_state:
        st.info("📝 **ผลลัพธ์**")
        
        # แสดง QR Code
        st.markdown(f"**QR Code:**")
        st.caption(f"Data: {st.session_state.get('gen_qr_data', '')[:50]}...") # แสดงข้อมูลบางส่วน

        if st.session_state['gen_qr_image']:
            # ปรับขนาดการแสดงผล QR Code ให้ใหญ่ขึ้น (จาก 300 เป็น 400 หรือเต็มความกว้าง)
            st.image(st.session_state['gen_qr_image'], caption="QR Code พร้อมใช้งาน", width=400)
            
            # ปุ่มดาวน์โหลด QR Code
            st.download_button(
                label="💾 ดาวน์โหลด QR Code (PNG)",
                data=st.session_state['gen_qr_image'],
                file_name="qrcode.png",
                mime="image/png"
            )

    else:
        # หน้าจอว่างเปล่าเมื่อยังไม่เริ่ม
        st.container(border=True).markdown(
            """
            <div style="text-align: center; padding: 40px; color: #64748B;">
                <h3>รอการสร้าง...</h3>
                <p>ใส่ข้อมูล เลือกโลโก้ แล้วกดปุ่มสร้างได้เลย</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
