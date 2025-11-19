import streamlit as st
import requests
import qrcode
from io import BytesIO
from PIL import Image
import os

# --- 1. ตั้งค่า Page Config ---
st.set_page_config(
    page_title="Short Link & QR Code",
    page_icon="🔗",
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
    
    /* กรอบ Preview Logo */
    .logo-preview {
        border: 1px dashed #9dbdb9;
        padding: 10px;
        border-radius: 8px;
        display: inline-block;
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Helper Functions ---

def shorten_url(url):
    """สร้าง Short Link โดยใช้ TinyURL API"""
    try:
        api_url = f"http://tinyurl.com/api-create.php?url={url}"
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            return response.text
        else:
            return None
    except Exception as e:
        return None

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
                
                # คำนวณขนาด Logo (ประมาณ 25% ของความกว้าง QR Code)
                width, height = img.size
                logo_size = int(width / 4) 
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
st.title("🔗 Short Link & QR Code Generator")
st.markdown("##### เครื่องมือสร้างลิงก์สั้นและคิวอาร์โค้ดพร้อมโลโก้")

col1, col2 = st.columns([1.2, 0.8], gap="large")

with col1:
    st.info("✏️ **ข้อมูลสำหรับสร้าง QR**")
    
    long_url = st.text_input("วาง URL ที่นี่ (เช่น https://www.example.com)", placeholder="https://...")
    
    st.write("")
    st.markdown("**ตัวเลือกโลโก้ (Logo Options):**")
    
    # Radio Button สำหรับเลือก Logo
    logo_option = st.radio(
        "เลือกรูปแบบโลโก้:",
        ("ไม่ใส่โลโก้", "logoSAO-BW-TH_0.png (ขาว-ดำ)", "logoSAO-TH-02.png (สี)"),
        index=0,
        horizontal=False
    )
    
    # Map ตัวเลือกกับชื่อไฟล์จริง
    logo_file_map = {
        "ไม่ใส่โลโก้": None,
        "logoSAO-BW-TH_0.png (ขาว-ดำ)": "logoSAO-BW-TH_0.png",
        "logoSAO-TH-02.png (สี)": "logoSAO-TH-02.png"
    }
    selected_logo_file = logo_file_map[logo_option]

    # --- ส่วนแสดง Preview Logo ---
    if selected_logo_file:
        st.markdown("ตัวอย่างโลโก้:")
        try:
            if os.path.exists(selected_logo_file):
                st.image(selected_logo_file, width=100, caption="Logo Preview")
            else:
                st.warning("ไม่พบไฟล์โลโก้ในระบบ (QR จะถูกสร้างโดยไม่มีโลโก้)")
        except Exception:
            st.warning("ไม่สามารถแสดงตัวอย่างโลโก้ได้")
    # -----------------------------

    st.write("")
    if st.button("🚀 สร้าง Short Link & QR Code", type="primary", use_container_width=True):
        if long_url:
            with st.spinner("กำลังสร้าง..."):
                # 1. Shorten URL (สร้าง Short Link เพื่อแสดงผล แต่ไม่ได้ใช้ใน QR)
                short_url = shorten_url(long_url)
                
                # 2. Generate QR Code (ใช้ URL ต้นฉบับตามที่ user ขอ)
                target_url_for_qr = long_url 
                
                # ส่งชื่อไฟล์โลโก้ไปที่ฟังก์ชัน
                qr_image = generate_qr_code_with_logo(target_url_for_qr, selected_logo_file)
                
                # เก็บผลลัพธ์ลง Session State
                st.session_state['gen_short_url'] = short_url
                st.session_state['gen_qr_image'] = qr_image
                st.session_state['gen_original_url'] = long_url
                
                st.success("✅ สร้างสำเร็จ!")
        else:
            st.warning("กรุณาใส่ URL ก่อนครับ")

with col2:
    # ตรวจสอบว่ามีการสร้างผลลัพธ์หรือยัง
    if 'gen_qr_image' in st.session_state:
        st.info("📝 **ผลลัพธ์**")
        
        # แสดง Short Link (ถ้าสร้างได้)
        if st.session_state.get('gen_short_url'):
            st.markdown(f"""
            <div class="result-box">
                <p style="margin-bottom: 5px; font-weight: bold; color: #263238;">Short Link:</p>
                <a href="{st.session_state['gen_short_url']}" target="_blank" style="font-size: 18px; color: #2563EB; text-decoration: none;">
                    {st.session_state['gen_short_url']}
                </a>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("ไม่สามารถสร้าง Short Link ได้ (แต่ QR Code สร้างเรียบร้อยแล้ว)")
        
        st.write("") # Spacer

        # แสดง QR Code
        st.markdown(f"**QR Code (จากลิงก์ต้นฉบับ):**")
        st.caption(f"Target: {st.session_state.get('gen_original_url', '')[:50]}...") # แสดง URL บางส่วน

        if st.session_state['gen_qr_image']:
            st.image(st.session_state['gen_qr_image'], caption="สแกนเพื่อไปยังลิงก์", width=250)
            
            # ปุ่มดาวน์โหลด QR Code
            st.download_button(
                label="💾 ดาวน์โหลด QR Code (PNG)",
                data=st.session_state['gen_qr_image'],
                file_name="qrcode_with_logo.png",
                mime="image/png"
            )

    else:
        # หน้าจอว่างเปล่าเมื่อยังไม่เริ่ม
        st.container(border=True).markdown(
            """
            <div style="text-align: center; padding: 40px; color: #64748B;">
                <h3>รอการสร้าง...</h3>
                <p>ใส่ URL เลือกโลโก้ แล้วกดปุ่มสร้างได้เลย</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
