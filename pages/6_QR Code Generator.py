import streamlit as st
import requests
import json
from PIL import Image
from io import BytesIO
from docx import Document

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
    html, body, [class*="css"] { font-family: 'Sarabun', sans-serif; }
    [data-testid="stAppViewContainer"] > .main { background-color: #e0f2f1; }
    h1, h2, h3 { color: #263238; font-weight: 700; }
    [data-testid="stSidebar"] { background-color: #e0f2f1; width: 250px !important; border-right: 1px solid #b2dfdb; }
    [data-testid="stSidebar"] > div:first-child { display: flex; flex-direction: column; height: 100%; }
    [data-testid="stSidebarNav"] { flex-grow: 1; margin-top: 20px; }
    .sidebar-footer { width: 100%; padding: 1rem; text-align: center; }
    .block-container { padding-top: 2rem; }
    .stButton > button { background-color: #2563EB; color: white; border: none; border-radius: 8px; padding: 0.5rem 1rem; font-weight: 500; transition: all 0.2s; }
    .stButton > button:hover { background-color: #1D4ED8; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .stTextArea textarea { background-color: #FFFFFF; border: 1px solid #9dbdb9; border-radius: 8px; font-family: 'Sarabun', sans-serif; line-height: 1.6; }
    [data-testid="stImage"] { border: 1px solid #9dbdb9; border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# --- 3. ฟังก์ชัน Logic การเรียก API ---
def extract_text_from_image(uploaded_file, api_key, model, task_type, max_tokens, temperature, top_p, repetition_penalty, pages=None):
    url = "https://api.opentyphoon.ai/v1/ocr"
    uploaded_file.seek(0)
    files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
    data = {
        'model': model, 'task_type': task_type,
        'max_tokens': str(max_tokens), 'temperature': str(temperature),
        'top_p': str(top_p), 'repetition_penalty': str(repetition_penalty)
    }
    if pages and pages.strip(): data['pages'] = pages.strip()
    headers = {'Authorization': f'Bearer {api_key}'}

    try:
        response = requests.post(url, files=files, data=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            extracted_texts = []
            for page_result in result.get('results', []):
                if page_result.get('success') and page_result.get('message'):
                    content = page_result['message']['choices'][0]['message']['content']
                    try:
                        parsed_content = json.loads(content)
                        text = parsed_content.get('natural_text', content)
                        if isinstance(text, (dict, list)): text = json.dumps(text, ensure_ascii=False)
                    except json.JSONDecodeError: text = content
                    extracted_texts.append(text)
                elif not page_result.get('success'):
                    error_msg = f"Error: {page_result.get('error', 'Unknown error')}"
                    extracted_texts.append(f"[{error_msg}]")
            return '\n\n---\n\n'.join(extracted_texts)
        else: return f"API Error: {response.status_code}\n{response.text}"
    except Exception as e: return f"Connection Error: {str(e)}"

def create_docx(text):
    doc = Document()
    for paragraph in text.split('\n'): doc.add_paragraph(paragraph)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# --- API Key & Config ---
if 'api_key' not in st.session_state:
    try: st.session_state['api_key'] = st.secrets.get("api_key", "")
    except Exception: st.session_state['api_key'] = "" 

model = "typhoon-ocr"
task_type = "v1.5"

# --- 4. Sidebar ---
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

# --- 5. Main Content ---
st.title("📄 ระบบแปลงภาพเป็นข้อความ (OCR)")
st.markdown("##### เครื่องมือช่วยดึงข้อความจากเอกสารภาษาไทยและอังกฤษด้วย AI")

# --- Input Selection ---
st.write("")
input_method = st.radio(
    "เลือกวิธีการนำเข้าข้อมูล:",
    options=["📁 อัปโหลดไฟล์", "📸 ถ่ายภาพ (Camera)"],
    horizontal=True,
    label_visibility="collapsed"
)

uploaded_file = None
if input_method == "📁 อัปโหลดไฟล์":
    file_upload = st.file_uploader("เลือกไฟล์ภาพ หรือเอกสาร PDF", type=['png', 'jpg', 'jpeg', 'webp', 'pdf'], key="file_uploader")
    if file_upload: uploaded_file = file_upload
elif input_method == "📸 ถ่ายภาพ (Camera)":
    camera_image = st.camera_input("ถ่ายภาพเอกสาร")
    if camera_image:
        uploaded_file = camera_image
        if not hasattr(uploaded_file, 'name'): uploaded_file.name = "camera_capture.jpg"
        if not hasattr(uploaded_file, 'type'): uploaded_file.type = "image/jpeg"

# --- Logic Auto-Process ---
# 1. Init Session State
if 'last_processed_id' not in st.session_state:
    st.session_state['last_processed_id'] = None
if 'ocr_result' not in st.session_state:
    st.session_state['ocr_result'] = ""

# 2. Check if we need to process
should_run_ocr = False
current_file_id = None

if uploaded_file:
    # สร้าง Unique ID สำหรับไฟล์นี้
    current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    # ถ้า ID ไม่ตรงกับที่เคยทำล่าสุด -> แปลว่าเป็นไฟล์ใหม่ -> สั่งรัน!
    if current_file_id != st.session_state['last_processed_id']:
        should_run_ocr = True

# --- Layout ---
if uploaded_file:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.info("🖼️ **ไฟล์ต้นฉบับ**")
        if uploaded_file.type == "application/pdf":
            st.warning("⚠️ ไฟล์ PDF จะไม่แสดงตัวอย่าง แต่สามารถประมวลผลได้ปกติ")
        else:
            st.image(uploaded_file, use_column_width=True)
        
        pages_input = st.text_input("ระบุหน้า (สำหรับ PDF)", placeholder="เช่น 1, 2 หรือ 1-5")
        st.markdown("---") 
        
        with st.expander("⚙️ การตั้งค่า (Advanced) | ปรับแต่งค่า Parameter", expanded=False):
            max_tokens = st.slider("Max Tokens", 1000, 16000, st.session_state.get("max_tokens", 16000), 100)
            temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.get("temperature", 0.1), 0.1)
            top_p = st.slider("Top P", 0.0, 1.0, st.session_state.get("top_p", 0.6), 0.1)
            repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, st.session_state.get("repetition_penalty", 1.1), 0.1)
        
        st.markdown("---") 
        current_api_key = st.session_state.get("api_key", "")
        # ปุ่ม Manual Start (เผื่อกดซ้ำ)
        manual_start = st.button("🚀 เริ่มประมวลผล (Start OCR)", type="primary", use_container_width=True)

    with col2:
        st.info("📝 **ผลลัพธ์ข้อความ**")
        
        # --- Execute OCR (Auto or Manual) ---
        # ทำงานเมื่อ: (เป็นไฟล์ใหม่) หรือ (กดปุ่ม Manual)
        if should_run_ocr or manual_start:
             if not current_api_key:
                st.error("❌ กรุณาตั้งค่า API Key ก่อน")
             else:
                # บันทึกค่า Parameter
                st.session_state["max_tokens"] = max_tokens
                st.session_state["temperature"] = temperature
                st.session_state["top_p"] = top_p
                st.session_state["repetition_penalty"] = repetition_penalty

                with st.spinner("🌀 กำลังประมวลผลอัตโนมัติ... โปรดรอสักครู่"):
                    result_text = extract_text_from_image(
                        uploaded_file, current_api_key, model, task_type, 
                        max_tokens, temperature, top_p, repetition_penalty, pages_input
                    )
                    # เก็บผลลัพธ์และ ID ลง State ทันทีที่เสร็จ
                    st.session_state["ocr_result"] = result_text
                    if current_file_id:
                        st.session_state['last_processed_id'] = current_file_id
                    
                # Force Rerun เพื่อให้ UI อัปเดตทันที (สำคัญสำหรับ Auto Process ในบางเคส)
                if should_run_ocr:
                    st.rerun()

        # แสดงผลลัพธ์
        result_text = st.session_state.get("ocr_result", "")
        st.text_area("Text Output", value=result_text, height=600, label_visibility="collapsed")
        
        if result_text:
            docx_file = create_docx(result_text)
            st.download_button("💾 ดาวน์โหลดไฟล์ .docx", data=docx_file, file_name="ocr_result.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
