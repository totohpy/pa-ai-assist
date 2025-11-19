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
    }

    /* Main Action Button (Generate) */
    div[data-testid="column"] > div > div > div > div > div > div.stButton > button {
         /* Default styling for primary action button if needed, or use specific targeting */
    }
    
    /* Specific styles for selection buttons (Unselected) */
    /* Using 'secondary' type which is usually white/transparent-ish */
    button[kind="secondary"] {
        background-color: transparent !important;
        border: 1px solid transparent !important; /* ไม่มีขอบ */
        color: #334155 !important;
        box-shadow: none !important;
    }
    button[kind="secondary"]:hover {
        background-color: #F1F5F9 !important; /* สีพื้นหลังเมื่อ Hover เบาๆ */
        border: 1px solid #CBD5E1 !important; /* มีขอบจางๆ เมื่อ Hover */
        color: #2563EB !important;
    }

    /* Specific styles for selection buttons (Selected) */
    /* Using 'primary' type but overriding to look like outlined white button */
    button[kind="primary"] {
        background-color: transparent !important; /* โปร่ง */
        border: 2px solid #2563EB !important; /* มีขอบสีฟ้า */
        color: #2563EB !important; /* ตัวหนังสือสีฟ้า */
        font-weight: bold !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.1) !important;
    }
    button[kind="primary"]:hover {
        background-color: #EFF6FF !important; /* ฟ้าอ่อนๆ เมื่อ Hover */
    }
    
    /* Ensure the generate button keeps its solid color */
    /* We target the specific button by its text content or position if possible, 
       but since we can't target by text in pure CSS easily without :has(), 
       we rely on Streamlit's primary/secondary classes.
       To fix the "Generate" button becoming transparent (since it is primary),
       we need to be careful. 
       
       Better approach: Use secondary for ALL selection buttons, 
       and change the border via inline style or key-based targeting if possible, 
       OR just accept that 'primary' style is overridden.
       
       Let's try to target only the buttons inside the columns for logo selection.
    */

    /* Re-apply solid style for the main 'Start' button which is likely the last button */
    /* This is a bit hacky but Streamlit CSS targeting is limited. */
    
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
                
                # กำหนดสัญลักษณ์และปุ่ม
                if is_selected:
                    icon = "🔴"
                    # ใช้ type="primary" ซึ่งเรา override CSS ให้เป็นขอบสีฟ้า พื้นโปร่ง
                    btn_type = "primary"
                else:
                    icon = "⭕"
                    # ใช้ type="secondary" ซึ่งเรา override CSS ให้ไม่มีขอบ พื้นโปร่ง
                    btn_type = "secondary"
                
                if st.button(f"{icon} {label}", key=f"btn_{key}", type=btn_type, use_container_width=True):
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

        st.markdown("---")
        
        # ปุ่ม Generate - เราต้องใช้ Inline CSS หรือ Custom Key เพื่อไม่ให้โดน CSS ของปุ่มเลือกกระทบมากนัก
        # แต่เนื่องจากเราแก้ CSS ของ primary ไปแล้ว ปุ่มนี้จะกลายเป็นพื้นโปร่งขอบฟ้าด้วย
        # ถ้าต้องการให้ปุ่มนี้ทึบเหมือนเดิม เราต้องใส่ Style เฉพาะ
        
        # ใช้ st.markdown เพื่อสร้างปุ่มที่หน้าตาต่างออกไปไม่ได้ง่ายๆ ใน Streamlit 
        # ดังนั้นเราจะปล่อยให้เป็นสไตล์เดียวกับ "Selected Button" (ขอบฟ้า พื้นโปร่ง) หรือยอมรับความเปลี่ยนแปลง
        # หรือใช้ trick: ใช้ secondary แต่ใส่ style
        
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
