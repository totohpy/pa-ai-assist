import streamlit as st
import requests
import json
from PIL import Image
from io import BytesIO
from docx import Document # ต้องติดตั้ง python-docx

# --- 1. ตั้งค่า Page Config ---
st.set_page_config(
    page_title="OCR",
    page_icon="📄",
    layout="wide"
)

# --- 2. Custom CSS (Styles) ---
# การตั้งค่าสไตล์ทั้งหมดตามธีม PA Assistant Chat (Sarabun, สีเขียวอ่อน)
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
    /* This targets the inner container of the sidebar */
    [data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    /* This makes the navigation take up all available space, pushing the footer down */
    [data-testid="stSidebarNav"] {
        flex-grow: 1;
        margin-top: 20px; /* Move navigation down */
    }
    .sidebar-footer {
        width: 100%;
        padding: 1rem;
        text-align: center; /* Center the footer content */
    }

    /* Remove Streamlit's default top padding */
    .block-container {
        padding-top: 2rem;
    }

    /* --- Feature Box Styling (Main Page) - included for consistency across app --- */
    .feature-link { text-decoration: none !important; color: inherit !important; }
    .feature-link:hover { text-decoration: none !important; color: inherit !important; }
    .feature-box {
        background-color: #e0f2f1;
        padding: 1rem 1rem;
        border-radius: 20px;
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        border: 1px solid #d0e0df;
    }
    .feature-box:hover {
        transform: translateY(-10px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    .feature-box .emoji { font-size: 1.6rem; line-height: 1; }
    .feature-box h3 { margin-top: 0.7rem; margin-bottom: 0.4rem; font-size: 1.2rem; }
    .feature-box p { color: #6c757d; font-size: 0.85rem; }
     
    /* --- Style the sidebar navigation --- */
    div[data-testid="stSidebarNav"] > ul > li > a {
        padding: 18px 40px !important; /* Increased padding for more height */
        font-size: 20px !important;    /* Larger font size */
        margin-bottom: 10px;
        border-radius: 8px;
        color: #263238 !important;      /* Darker text for inactive links */
        background-color: #b2dfdb;      /* Light teal for inactive links */
        border: 1px solid #9dbdb9;
        font-weight: 500;
    }
     
    /* Style the ACTIVE page link */
    div[data-testid="stSidebarNav"] a[aria-current="page"] {
        background-color: #80cbc4;      /* Dark teal for active link */
        color: #FFFFFF !important;      /* White text for active link */
        font-weight: 600;
        border: 1px solid #00796b;
    }

    /* ปรับแต่งปุ่ม Primary (Start OCR) - คงสีน้ำเงินเพื่อให้โดดเด่น */
    .stButton > button {
        background-color: #2563EB; /* สีน้ำเงิน Typhoon */
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
    
    /* ปรับแต่งพื้นที่แสดงผลลัพธ์ */
    .stTextArea textarea {
        background-color: #FFFFFF;
        border: 1px solid #9dbdb9; /* ใช้สีขอบที่เข้ากับธีม */
        border-radius: 8px;
        font-family: 'Sarabun', sans-serif;
        line-height: 1.6;
    }
    
    /* กรอบรูปภาพ */
    [data-testid="stImage"] {
        border: 1px solid #9dbdb9;
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. ฟังก์ชัน Logic การเรียก API ---
def extract_text_from_image(uploaded_file, api_key, model, task_type, max_tokens, temperature, top_p, repetition_penalty, pages=None):
    # Endpoint สำหรับ Typhoon OCR
    url = "https://api.opentyphoon.ai/v1/ocr"
    
    # เตรียมไฟล์ (Streamlit ส่งมาเป็น BytesIO)
    files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
    
    # เตรียม Parameters
    data = {
        'model': model,
        'task_type': task_type,
        'max_tokens': str(max_tokens),
        'temperature': str(temperature),
        'top_p': str(top_p),
        'repetition_penalty': str(repetition_penalty)
    }

    if pages and pages.strip():
        data['pages'] = pages.strip()

    headers = {
        'Authorization': f'Bearer {api_key}'
    }

    try:
        # ส่ง Request ไปยัง API
        response = requests.post(url, files=files, data=data, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            extracted_texts = []
            
            # แกะ JSON Response เพื่อดึงข้อความที่ได้จาก OCR
            for page_result in result.get('results', []):
                if page_result.get('success') and page_result.get('message'):
                    content = page_result['message']['choices'][0]['message']['content']
                    try:
                        # พยายามแปลง String JSON กลับเป็น Object เพื่อดึงเฉพาะ natural_text (ข้อความดิบ)
                        parsed_content = json.loads(content)
                        text = parsed_content.get('natural_text', content)
                        # จัดการกรณีที่ข้อความเป็น dict/list
                        if isinstance(text, (dict, list)):
                            text = json.dumps(text, ensure_ascii=False)
                    except json.JSONDecodeError:
                        # ถ้าไม่ใช่ JSON ให้ใช้ content ดิบ
                        text = content
                    extracted_texts.append(text)
                elif not page_result.get('success'):
                    error_msg = f"Error: {page_result.get('error', 'Unknown error')}"
                    extracted_texts.append(f"[{error_msg}]")
            
            # รวมข้อความจากทุกหน้าเข้าด้วยกัน
            return '\n\n---\n\n'.join(extracted_texts)
        else:
            return f"API Error: {response.status_code}\n{response.text}"
            
    except Exception as e:
        return f"Connection Error: {str(e)}"

# ฟังก์ชันสำหรับสร้างไฟล์ docx
def create_docx(text):
    doc = Document()
    # เพิ่มย่อหน้าข้อความที่ได้จากการ OCR
    for paragraph in text.split('\n'):
        doc.add_paragraph(paragraph)
    
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# --- API Key loader for Streamlit Cloud ---
if 'api_key' not in st.session_state:
    try:
        # โหลดจาก st.secrets["api_key"] (สำหรับ Streamlit Cloud)
        # ใช้ .get() เพื่อป้องกัน KeyError หากไม่มี secrets.toml
        st.session_state['api_key'] = st.secrets.get("api_key", "")
    except Exception:
        # ถ้าไม่มี secret, ตั้งค่าเป็น empty string
        st.session_state['api_key'] = "" 
# ------------------------------------------

# --- 4. Fixed Parameters ---
# กำหนดค่าพารามิเตอร์คงที่ของโมเดล
model = "typhoon-ocr"
task_type = "v1.5"
# ค่าพารามิเตอร์ Max Tokens, Temperature, Top P, Repetition Penalty ถูกย้ายไปกำหนดด้วย Slider ในส่วน Main Content

# --- 5. ส่วน Sidebar (Footer Only) ---
with st.sidebar:
    # --- Sidebar Footer ---
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

# --- 6. ส่วน Main Content ---
st.title("📄 ระบบแปลงภาพเป็นข้อความ (OCR)")
st.markdown("##### เครื่องมือช่วยดึงข้อความจากเอกสารภาษาไทยและอังกฤษด้วย AI")

# พื้นที่ Upload
uploaded_file = st.file_uploader(
    "เลือกไฟล์ภาพ (JPG, PNG) หรือเอกสาร (PDF)", 
    type=['png', 'jpg', 'jpeg', 'webp', 'pdf']
)

# Layout หลัก: 2 คอลัมน์
if uploaded_file:
    col1, col2 = st.columns([1, 1], gap="large")

    # --- Column ซ้าย: Preview & Controls ---
    with col1:
        st.info("🖼️ **ไฟล์ต้นฉบับ**")
        
        # แสดง Preview หรือคำเตือนสำหรับ PDF
        if uploaded_file.type == "application/pdf":
            st.warning("⚠️ ไฟล์ PDF จะไม่แสดงตัวอย่าง แต่สามารถประมวลผลได้ปกติ")
        else:
            st.image(uploaded_file, use_column_width=True)
        
        # ย้าย pages_input มาไว้เป็นลำดับแรกสุดในส่วนควบคุมตามคำขอ
        pages_input = st.text_input("ระบุหน้า (สำหรับ PDF)", placeholder="เช่น 1, 2 หรือ 1-5 (เว้นว่างเพื่อทำทั้งหมด)")
        
        st.markdown("---") # เพิ่มเส้นแบ่งระหว่าง Input หลัก กับ Advanced Settings
        
        # --- Advanced Settings (ตามที่ร้องขอ) ---
        # นำหัวข้อ "### ⚙️ การตั้งค่า (Advanced)" ออก และรวมข้อความเข้ากับหัวข้อ Expander
        with st.expander("⚙️ การตั้งค่า (Advanced) | ปรับแต่งค่า Parameter", expanded=False):
            # ใช้ st.session_state.get เพื่อกำหนดค่าเริ่มต้นและเก็บค่าที่ผู้ใช้ปรับ
            max_tokens = st.slider("Max Tokens", 1000, 16000, st.session_state.get("max_tokens", 16000), 100, key="max_tokens_slider")
            temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.get("temperature", 0.1), 0.1, key="temperature_slider")
            top_p = st.slider("Top P", 0.0, 1.0, st.session_state.get("top_p", 0.6), 0.1, key="top_p_slider")
            repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, st.session_state.get("repetition_penalty", 1.1), 0.1, key="repetition_penalty_slider")
        
        st.markdown("---") # เพิ่มเส้นแบ่งก่อนปุ่ม

        # ปุ่ม Action
        current_api_key = st.session_state.get("api_key", "")

        if st.button("🚀 เริ่มประมวลผล (Start OCR)", type="primary", use_container_width=True):
            if not current_api_key:
                st.error("❌ กรุณาตั้งค่า API Key ใน Streamlit Secrets หรือติดต่อผู้ดูแลระบบเพื่อเข้าถึงฟังก์ชันนี้")
            else:
                # บันทึกค่าพารามิเตอร์ลงใน session state ก่อนเรียก API เพื่อให้ค่าคงที่
                st.session_state["max_tokens"] = max_tokens
                st.session_state["temperature"] = temperature
                st.session_state["top_p"] = top_p
                st.session_state["repetition_penalty"] = repetition_penalty

                with st.spinner("🌀 AI กำลังอ่านเอกสาร... โปรดรอสักครู่"):
                    # เรียกฟังก์ชันประมวลผล โดยใช้ค่าจาก slider
                    result_text = extract_text_from_image(
                        uploaded_file, current_api_key, model, task_type, 
                        max_tokens, temperature, top_p, repetition_penalty, pages_input
                    )
                    st.session_state["ocr_result"] = result_text
                    st.success("✅ เสร็จสิ้น!")

    # --- Column ขวา: Result ---
    with col2:
        st.info("📝 **ผลลัพธ์ข้อความ**")
        
        result_text = st.session_state.get("ocr_result", "")
        
        # Text Area สำหรับแสดงผล
        st.text_area(
            label="Text Output",
            value=result_text,
            height=600,
            placeholder="ผลลัพธ์จากการ OCR จะปรากฏที่นี่...",
            label_visibility="collapsed"
        )
        
        # ปุ่ม Download (แก้ไขเป็น .docx)
        if result_text:
            docx_file = create_docx(result_text)
            st.download_button(
                label="💾 ดาวน์โหลดไฟล์ .docx",
                data=docx_file,
                file_name="ocr_result.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

# ส่วน else ถูกลบออกแล้ว
