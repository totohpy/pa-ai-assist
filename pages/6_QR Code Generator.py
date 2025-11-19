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
    
    /* ปรับแต่ง Radio ให้ดูดีขึ้น */
    .stRadio > label {
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Helper Functions ---

def generate_qr_code_with_logo(data, logo_file_name=None):
    """สร้าง QR Code จากข้อมูลที่กำหนด และใส่ Logo ตรงกลาง (ถ้ามี)"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=2,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    
    if logo_file_name:
        try:
            if os.path.exists(logo_file_name):
                logo = Image.open(logo_file_name)
                width, height = img.size
                logo_size = int(width / 3.5) 
                logo = logo.resize((logo_size, logo_size))
                pos = ((width - logo_size) // 2, (height - logo_size) // 2)
                img.paste(logo, pos)
        except Exception:
            pass

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

        # --- แสดงรูปตัวอย่างโลโก้ ---
        cols_preview = st.columns(3)
        
        # Helper function to render preview
        def render_preview(col, label, image_path=None):
            with col:
                st.markdown(f"<div style='text-align:center; margin-bottom:5px; font-weight:bold; font-size:0.9rem;'>{label}</div>", unsafe_allow_html=True)
                if image_path and os.path.exists(image_path):
                    img_b64 = get_image_base64(image_path)
                    if img_b64:
                        st.markdown(f"""
                            <div style='height:80px; display:flex; align-items:center; justify-content:center; 
                            border:1px solid #eee; border-radius:8px; background:white; padding:5px;'>
                                <img src="data:image/png;base64,{img_b64}" style="max-height:70px; max-width:100%;">
                            </div>""", unsafe_allow_html=True)
                else:
                    # Placeholder for No Logo or Missing File
                    st.markdown(f"""
                        <div style='height:80px; display:flex; align-items:center; justify-content:center; 
                        border:1px dashed #ccc; border-radius:8px; color:#aaa; background:#f9f9f9;'>
                            {'No Logo' if not image_path else 'File Not Found'}
                        </div>""", unsafe_allow_html=True)

        render_preview(cols_preview[0], "1. ไม่ใส่โลโก้", None)
        render_preview(cols_preview[1], "2. ขาว-ดำ", "logoSAO-BW-TH_0.png")
        render_preview(cols_preview[2], "3. สี", "logoSAO-TH-02.png")

        st.write("")
        # --- Radio Button สำหรับเลือก ---
        logo_option = st.radio(
            "เลือกรูปแบบ:",
            ("ไม่ใส่โลโก้", "โลโก้ขาว-ดำ", "โลโก้สี"),
            horizontal=True,
            label_visibility="collapsed" # ซ่อน Label เพราะมี header ด้านบนแล้ว
        )

        # Map ตัวเลือกกับชื่อไฟล์
        logo_map = {
            "ไม่ใส่โลโก้": None,
            "โลโก้ขาว-ดำ": "logoSAO-BW-TH_0.png",
            "โลโก้สี": "logoSAO-TH-02.png"
        }
        selected_logo = logo_map[logo_option]

        st.markdown("---")
        
        if st.button("🚀 สร้าง QR Code", type="primary", use_container_width=True):
            if qr_data:
                with st.spinner("กำลังสร้าง..."):
                    img_buf = generate_qr_code_with_logo(qr_data, selected_logo)
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
