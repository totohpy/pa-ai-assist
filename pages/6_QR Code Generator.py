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

    /* General Button Style */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
        width: 100%;
    }

    /* ---------------------------------------- */
    /* Button Styles Implementation            */
    /* ---------------------------------------- */

    /* 1. Selection Buttons (Secondary) - สีขาว/โปร่ง */
    button[kind="secondary"] {
        background-color: white !important;
        border: 1px solid #CBD5E1 !important; 
        color: #334155 !important;
        box-shadow: none !important;
    }
    button[kind="secondary"]:hover {
        background-color: #F1F5F9 !important;
        border-color: #94A3B8 !important;
        color: #0F172A !important;
    }
    button[kind="secondary"]:focus {
        border-color: #2563EB !important;
        color: #2563EB !important;
        background-color: #EFF6FF !important;
    }

    /* 2. Generate Button (Primary) - สีน้ำเงินทึบ */
    button[kind="primary"] {
        background-color: #2563EB !important;
        border: 1px solid #2563EB !important;
        color: white !important;
        font-weight: bold !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2) !important;
    }
    button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
        box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.3) !important;
    }
    
    .result-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #9dbdb9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-top: 10px;
    }

</style>
""", unsafe_allow_html=True)

# --- 3. Helper Functions ---

def generate_qr_code_with_logo(data, logo_file_name=None, logo_size_factor=3.5):
    """
    สร้าง QR Code จากข้อมูลที่กำหนด และใส่ Logo ตรงกลาง (ถ้ามี)
    ปรับปรุง: รองรับไฟล์ PNG พื้นหลังโปร่งใส (RGBA)
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=2,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    # สร้างรูป QR Code พื้นฐาน (โหมด RGB)
    img = qr.make_image(fill_color="black", back_color="white").convert('RGBA')
    
    if logo_file_name:
        try:
            if os.path.exists(logo_file_name):
                # เปิดไฟล์โลโก้
                logo = Image.open(logo_file_name)
                
                # คำนวณขนาด
                if logo_size_factor <= 0: logo_size_factor = 1
                width, height = img.size
                logo_size = int(width / logo_size_factor)
                logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
                
                # --- สร้างพื้นหลังสีขาวรองรับโลโก้ ---
                # สร้างภาพสี่เหลี่ยมสีขาวขนาดเท่าโลโก้ (บวกขอบนิดหน่อยถ้าต้องการ)
                bg_size = logo_size 
                logo_bg = Image.new("RGBA", (bg_size, bg_size), "white")
                
                # คำนวณตำแหน่งวาง (กึ่งกลาง)
                pos = ((width - bg_size) // 2, (height - bg_size) // 2)
                
                # วางพื้นหลังสีขาวลงไปก่อน
                img.paste(logo_bg, pos)
                
                # วางโลโก้ทับลงไป โดยใช้ตัวเองเป็น Mask เพื่อรักษาความโปร่งใส
                # ถ้าโลโก้เป็น RGBA ให้ใช้ mask
                if logo.mode == 'RGBA':
                    img.paste(logo, pos, mask=logo)
                else:
                    img.paste(logo, pos)
                    
        except Exception as e:
            print(f"Logo Error: {e}")

    # แปลงกลับเป็น RGB ก่อนเซฟ (ถ้าไม่ต้องการ Transparency ใน QR final result) หรือเซฟเป็น PNG ได้เลย
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception:
        return None

# --- 4. Sidebar ---
with st.sidebar:
    try:
        st.image("image_e05e9c.png", use_column_width=True) 
    except:
        pass 

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

with st.container(border=True):
    col_left, col_right = st.columns([1.2, 0.8], gap="large")

    # --- Left Column ---
    with col_left:
        st.subheader("1. ใส่ข้อมูล")
        qr_data = st.text_input("URL หรือข้อความที่ต้องการ:", placeholder="https://www.example.com")
        
        st.write("")
        st.subheader("2. เลือกโลโก้")
        
        # Init State
        if 'selected_logo_key' not in st.session_state:
            st.session_state['selected_logo_key'] = 'none'

        # Layout 3 คอลัมน์
        l1, l2, l3 = st.columns(3)
        
        # --- Helper เพื่อสร้าง Card ---
        def render_logo_selection(col, key, label, image_path=None, is_no_logo=False):
            with col:
                # แสดงรูป
                if is_no_logo:
                     st.markdown("""
                        <div style='height:100px; border:1px dashed #ccc; display:flex; align-items:center; justify-content:center; 
                        color:#aaa; border-radius:8px; background:white; margin-bottom:10px; font-size:0.8rem;'>No Logo</div>
                    """, unsafe_allow_html=True)
                elif image_path and os.path.exists(image_path):
                    b64 = get_image_base64(image_path)
                    if b64:
                        st.markdown(f"""
                            <div style='height:100px; display:flex; align-items:center; justify-content:center; 
                            border:1px solid #eee; border-radius:8px; background:white; margin-bottom:10px;'>
                                <img src="data:image/png;base64,{b64}" style="max-height:80px; max-width:100%;">
                            </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='height:100px; display:flex; align-items:center; justify-content:center; color:red; border:1px solid #eee; border-radius:8px; margin-bottom:10px;'>Missing</div>", unsafe_allow_html=True)
                
                # ปุ่มเลือก
                is_selected = (st.session_state['selected_logo_key'] == key)
                
                # กำหนดสัญลักษณ์
                icon = "🔴" if is_selected else "⭕"
                
                # ใช้ type="secondary" เสมอสำหรับปุ่มเลือก เพื่อให้เป็นพื้นขาวตามที่ขอ
                if st.button(f"{icon} {label}", key=f"btn_{key}", type="secondary", use_container_width=True):
                    st.session_state['selected_logo_key'] = key
                    st.rerun()

        # Render
        render_logo_selection(l1, 'none', 'ไม่ใส่', is_no_logo=True)
        render_logo_selection(l2, 'bw', 'ขาว-ดำ', image_path="logoSAO-BW-TH_0.png")
        render_logo_selection(l3, 'color', 'สี', image_path="logoSAO-TH-02.png")

        # Map selection
        logo_map = {
            "none": None,
            "bw": "logoSAO-BW-TH_0.png",
            "color": "logoSAO-TH-02.png"
        }
        selected_logo = logo_map[st.session_state['selected_logo_key']]
        
        # --- Slider ปรับขนาดโลโก้ (แสดงเฉพาะเมื่อมีการเลือกโลโก้) ---
        if selected_logo is not None:
            st.write("")
            st.markdown("**ปรับขนาดโลโก้:**")
            # ค่าตัวหาร: 5 (เล็ก) -> 2.5 (ใหญ่)
            logo_scale_input = st.slider("ขนาดโลโก้ (เล็ก - ใหญ่)", min_value=1, max_value=4, value=2, step=1)
            logo_divisor = 5.625 + (logo_scale_input * -0.625)
        else:
            logo_divisor = 3.5

        st.markdown("---")
        
        # ปุ่ม Generate ใช้ type="primary" เพื่อให้ CSS จับเป็นสีน้ำเงินทึบ
        if st.button("🚀 สร้าง QR Code", type="primary", use_container_width=True):
            if qr_data:
                with st.spinner("กำลังสร้าง..."):
                    # ส่งค่า logo_divisor ไปด้วย
                    img_buf = generate_qr_code_with_logo(qr_data, selected_logo, logo_divisor)
                    st.session_state['gen_qr_image'] = img_buf
                    st.session_state['gen_qr_data'] = qr_data
            else:
                st.error("กรุณาใส่ URL หรือข้อความก่อนครับ")

    # --- Right Column ---
    with col_right:
        st.subheader("3. ผลลัพธ์")
        
        result_placeholder = st.empty()
        
        if 'gen_qr_image' in st.session_state:
            with result_placeholder.container():
                st.markdown("""
                    <div style="text-align: center; padding: 20px; border: 1px solid #E2E8F0; border-radius: 10px; background-color: #F8FAFC;">
                """, unsafe_allow_html=True)
                
                st.image(st.session_state['gen_qr_image'], caption="QR Code ของคุณ", width=300)
                
                st.success("สร้างเรียบร้อย!")
                st.caption(f"Link: {st.session_state.get('gen_qr_data', '')[:40]}...")
                
                st.download_button(
                    label="💾 ดาวน์โหลดไฟล์ PNG",
                    data=st.session_state['gen_qr_image'],
                    file_name="qrcode.png",
                    mime="image/png",
                    use_container_width=True
                )
                
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            result_placeholder.markdown("""
                <div style="
                    height: 400px; 
                    display: flex; 
                    flex-direction: column;
                    align-items: center; 
                    justify-content: center; 
                    color: #94A3B8; 
                    text-align: center;
                    border: 2px dashed #E2E8F0;
                    border-radius: 10px;
                    background-color: #F8FAFC;
                ">
                    <div style="font-size: 4rem; margin-bottom: 10px;">📷</div>
                    <div style="font-size: 1.1rem; font-weight: 500;">รอการสร้าง QR Code</div>
                    <div style="font-size: 0.9rem;">กรอกข้อมูลและเลือกโลโก้ทางซ้ายมือ<br>แล้วกดปุ่มสร้างได้เลย</div>
                </div>
            """, unsafe_allow_html=True)
