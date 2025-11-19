import streamlit as st
import requests
import qrcode
from io import BytesIO
from PIL import Image

# --- 1. ตั้งค่า Page Config ---
st.set_page_config(
    page_title="Short Link & QR Code",
    page_icon="🔗",
    layout="wide"
)

# --- 2. Custom CSS (Styles) ---
# ใช้สไตล์เดียวกับหน้า Typhoon OCR (Sarabun, สีเขียวอ่อน theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;700&display=swap');

    /* บังคับใช้ฟอนต์ Sarabun กับทุก Element */
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
    }
    
    /* --- Overall App Color Theme --- */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #e0f2f1; /* สีเขียวอ่อนตามธีม PA Assistant */
    }
    
    /* ปรับหัวข้อ Header */
    h1 { 
        font-size: 36px !important; 
        color: #263238; 
        font-weight: 700;
    }
    h2, h3 {
        color: #263238;
        font-weight: 700;
    }
    
    /* ปรับแต่ง Sidebar */
    [data-testid="stSidebar"] {
        background-color: #e0f2f1;
        width: 250px !important;
        border-right: 1px solid #b2dfdb;
    }

    /* --- Flexbox layout for Sidebar --- */
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

    /* Remove Streamlit's default top padding */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Style Navigation Links */
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

    /* ปรับแต่งปุ่ม Primary */
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
    
    /* Result Box Styling */
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

def shorten_url(url):
    """สร้าง Short Link โดยใช้ TinyURL API (ไม่ต้องใช้ Key)"""
    try:
        api_url = f"http://tinyurl.com/api-create.php?url={url}"
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.text
        else:
            return None
    except Exception as e:
        return None

def generate_qr_code(data):
    """สร้าง QR Code image"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to bytes for Streamlit
    buf = BytesIO()
    img.save(buf)
    buf.seek(0)
    return buf

# --- 4. Sidebar Section ---
with st.sidebar:
    try:
        st.image("image_e05e9c.png", use_column_width=True) 
    except:
        pass # ถ้าไม่มีรูปก็ไม่ต้องแสดงอะไร หรือแสดง Text แทน

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
st.markdown("##### เครื่องมือสร้างลิงก์สั้นและคิวอาร์โค้ดอย่างรวดเร็ว")

col1, col2 = st.columns([1.2, 0.8], gap="large")

with col1:
    st.info("✏️ **ใส่ลิงก์ที่ต้องการแปลง**")
    
    long_url = st.text_input("วาง URL ที่นี่ (เช่น https://www.example.com/very-long-url...)", placeholder="https://...")
    
    if st.button("🚀 สร้าง Short Link & QR Code", type="primary", use_container_width=True):
        if long_url:
            with st.spinner("กำลังสร้าง..."):
                # 1. Shorten URL
                short_url = shorten_url(long_url)
                
                # 2. Generate QR Code (ใช้ Short URL ถ้ามี, ถ้าไม่มีใช้ Long URL)
                target_url_for_qr = short_url if short_url else long_url
                qr_image = generate_qr_code(target_url_for_qr)
                
                # เก็บผลลัพธ์ลง Session State
                st.session_state['gen_short_url'] = short_url
                st.session_state['gen_qr_image'] = qr_image
                st.session_state['gen_original_url'] = long_url
                
                st.success("✅ สร้างสำเร็จ!")
        else:
            st.warning("กรุณาใส่ URL ก่อนครับ")

with col2:
    # ตรวจสอบว่ามีการสร้างผลลัพธ์หรือยัง
    if 'gen_short_url' in st.session_state and st.session_state['gen_short_url']:
        st.info("📝 **ผลลัพธ์**")
        
        # แสดง Short Link
        st.markdown(f"""
        <div class="result-box">
            <p style="margin-bottom: 5px; font-weight: bold; color: #263238;">Short Link:</p>
            <a href="{st.session_state['gen_short_url']}" target="_blank" style="font-size: 18px; color: #2563EB; text-decoration: none;">
                {st.session_state['gen_short_url']}
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("") # Spacer

        # แสดง QR Code
        st.markdown("**QR Code:**")
        if st.session_state['gen_qr_image']:
            st.image(st.session_state['gen_qr_image'], caption="สแกนเพื่อไปยังลิงก์", width=200)
            
            # ปุ่มดาวน์โหลด QR Code
            st.download_button(
                label="💾 ดาวน์โหลด QR Code (PNG)",
                data=st.session_state['gen_qr_image'],
                file_name="qrcode.png",
                mime="image/png"
            )
    
    elif 'gen_original_url' in st.session_state:
        # กรณี Shorten ไม่สำเร็จ แต่สร้าง QR จาก Original ได้
        st.warning("ไม่สามารถสร้าง Short Link ได้ (อาจเกิดข้อผิดพลาดจากเครือข่าย) แต่สร้าง QR Code จากลิงก์เดิมให้แล้ว")
        if st.session_state['gen_qr_image']:
             st.image(st.session_state['gen_qr_image'], caption="QR Code (จากลิงก์เดิม)", width=200)
             st.download_button(
                label="💾 ดาวน์โหลด QR Code",
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
                <p>ผลลัพธ์ Short Link และ QR Code จะแสดงที่นี่</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
