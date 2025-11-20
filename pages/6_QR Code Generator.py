import streamlit as st
import requests
import json
from PIL import Image
from io import BytesIO
from docx import Document # ต้องติดตั้ง python-docx

# --- 1. ตั้งค่า Page Config ---
st.set_page_config(
    page_title="Typhoon OCR",
    page_icon="📄",
    layout="wide"
)

# --- 2. Custom CSS (Styles) ---
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

    /* ปรับแต่งปุ่ม Primary (Start OCR) */
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
        border: 1px solid #9dbdb9; 
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
    # สำคัญ: ต้อง seek(0) ก่อนส่งไฟล์เสมอเผื่อไฟล์ถูกอ่านไปแล้ว
    uploaded_file.seek(0)
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
            
            # แกะ JSON Response
            for page_result in result.get('results', []):
                if page_result.get('success') and page_result.get('message'):
                    content = page_result['message']['choices'][0]['message']['content']
                    try:
                        # พยายามแปลง String JSON กลับเป็น Object เพื่อดึงเฉพาะ natural_text
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
        # โหลดจาก st.secrets["api_key"]
        st.session_state['api_key'] = st.secrets.get("api_key", "")
    except Exception:
        st.session_state['api_key'] = "" 
# ------------------------------------------

# --- 4. Fixed Parameters ---
model = "typhoon-ocr"
task_type = "v1.5"

# --- 5. ส่วน Sidebar (Footer Only) ---
with st.sidebar:
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

# --- Input Selection (Radio) ---
st.write("")
input_method = st.radio(
    "เลือกวิธีการนำเข้าข้อมูล:",
    options=["📁 อัปโหลดไฟล์", "📸 ถ่ายภาพ (Camera)"],
    horizontal=True,
    label_visibility="collapsed"
)

uploaded_file = None

if input_method == "📁 อัปโหลดไฟล์":
    file_upload = st.file_uploader(
        "เลือกไฟล์ภาพ (JPG, PNG) หรือเอกสาร (PDF)", 
        type=['png', 'jpg', 'jpeg', 'webp', 'pdf'],
        key="file_uploader"
    )
    if file_upload:
        uploaded_file = file_upload

elif input_method == "📸 ถ่ายภาพ (Camera)":
    camera_image = st.camera_input("ถ่ายภาพเอกสาร")
    if camera_image:
        uploaded_file = camera_image
        # กำหนดค่าจำลองสำหรับไฟล์ภาพจากกล้อง
        if not hasattr(uploaded_file, 'name'):
            uploaded_file.name = "camera_capture.jpg"
        if not hasattr(uploaded_file, 'type'):
            uploaded_file.type = "image/jpeg"

# --- Logic Auto-Process ---
# สร้าง session state สำหรับเก็บ ID ไฟล์ล่าสุดที่ประมวลผลไปแล้ว
if 'last_processed_file_id' not in st.session_state:
    st.session_state['last_processed_file_id'] = None

should_process_auto = False

if uploaded_file:
    # สร้าง ID จำลองของไฟล์ (ใช้ชื่อ+ขนาด) เพื่อเช็คว่าเป็นไฟล์ใหม่หรือไม่
    current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    # ถ้าไฟล์ปัจจุบัน ไม่ตรงกับไฟล์ล่าสุดที่เคยทำ -> แปลว่าไฟล์ใหม่ -> สั่ง Process ทันที
    if current_file_id != st.session_state['last_processed_file_id']:
        should_process_auto = True
        st.session_state['last_processed_file_id'] = current_file_id # อัปเดต ID ล่าสุด

# --- Layout หลัก: 2 คอลัมน์ ---
if uploaded_file:
    col1, col2 = st.columns([1, 1], gap="large")

    # --- Column ซ้าย: Preview & Controls ---
    with col1:
        st.info("🖼️ **ไฟล์ต้นฉบับ**")
        
        if uploaded_file.type == "application/pdf":
            st.warning("⚠️ ไฟล์ PDF จะไม่แสดงตัวอย่าง แต่สามารถประมวลผลได้ปกติ")
        else:
            st.image(uploaded_file, use_column_width=True)
        
        # ย้าย pages_input มาไว้เป็นลำดับแรกสุด
        pages_input = st.text_input("ระบุหน้า (สำหรับ PDF)", placeholder="เช่น 1, 2 หรือ 1-5 (เว้นว่างเพื่อทำทั้งหมด)")
        
        st.markdown("---") 
        
        # --- Advanced Settings ---
        with st.expander("⚙️ การตั้งค่า (Advanced) | ปรับแต่งค่า Parameter", expanded=False):
            max_tokens = st.slider("Max Tokens", 1000, 16000, st.session_state.get("max_tokens", 16000), 100, key="max_tokens_slider")
            temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.get("temperature", 0.1), 0.1, key="temperature_slider")
            top_p = st.slider("Top P", 0.0, 1.0, st.session_state.get("top_p", 0.6), 0.1, key="top_p_slider")
            repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, st.session_state.get("repetition_penalty", 1.1), 0.1, key="repetition_penalty_slider")
        
        st.markdown("---") 

        # ปุ่ม Action (กดเองก็ได้ หรือรันออโต้ก็ได้)
        current_api_key = st.session_state.get("api_key", "")
        manual_start = st.button("🚀 เริ่มประมวลผล (Start OCR)", type="primary", use_container_width=True)

    # --- Column ขวา: Result ---
    with col2:
        st.info("📝 **ผลลัพธ์ข้อความ**")
        
        # Logic: รันถ้ากดปุ่ม หรือ เป็นไฟล์ใหม่ (Auto)
        if manual_start or should_process_auto:
             if not current_api_key:
                st.error("❌ กรุณาตั้งค่า API Key ใน Streamlit Secrets หรือติดต่อผู้ดูแลระบบเพื่อเข้าถึงฟังก์ชันนี้")
             else:
                # บันทึกค่าพารามิเตอร์
                st.session_state["max_tokens"] = max_tokens
                st.session_state["temperature"] = temperature
                st.session_state["top_p"] = top_p
                st.session_state["repetition_penalty"] = repetition_penalty

                with st.spinner("🌀 AI กำลังอ่านเอกสารอัตโนมัติ... โปรดรอสักครู่"):
                    result_text = extract_text_from_image(
                        uploaded_file, current_api_key, model, task_type, 
                        max_tokens, temperature, top_p, repetition_penalty, pages_input
                    )
                    st.session_state["ocr_result"] = result_text
                    
                    # ถ้ากดปุ่มเอง ให้แสดง success message (ถ้า auto อาจจะไม่ต้องแสดงเพื่อให้ดู smooth)
                    if manual_start:
                        st.success("✅ เสร็จสิ้น!")

        # แสดงผลลัพธ์จาก Session State
        result_text = st.session_state.get("ocr_result", "")
        
        # Text Area สำหรับแสดงผล
        st.text_area(
            label="Text Output",
            value=result_text,
            height=600,
            placeholder="ผลลัพธ์จากการ OCR จะปรากฏที่นี่...",
            label_visibility="collapsed"
        )
        
        # ปุ่ม Download
        if result_text:
            docx_file = create_docx(result_text)
            st.download_button(
                label="💾 ดาวน์โหลดไฟล์ .docx",
                data=docx_file,
                file_name="ocr_result.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
