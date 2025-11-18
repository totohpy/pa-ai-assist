import streamlit as st
import requests
import json
from PIL import Image

# --- 1. ตั้งค่า Page Config (ต้องอยู่บรรทัดแรกสุด) ---
st.set_page_config(
    page_title="Typhoon OCR",
    page_icon="📄",
    layout="wide"
)

# --- 2. Custom CSS (Styles) ---
# ใช้ฟอนต์ Sarabun และปรับแต่ง UI ให้ดูสะอาดตา (เหมือนหน้า Chat)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;700&display=swap');

    /* บังคับใช้ฟอนต์ Sarabun กับทุก Element */
    html, body, [class*="css"] {
        font-family: 'Sarabun', sans-serif;
    }
    
    /* ปรับหัวข้อ Header */
    h1, h2, h3 {
        color: #1E293B;
        font-weight: 700;
    }
    
    /* ปรับแต่ง Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
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
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        font-family: 'Sarabun', sans-serif;
        line-height: 1.6;
    }
    
    /* กรอบรูปภาพ */
    [data-testid="stImage"] {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. ฟังก์ชัน Logic การเรียก API ---
def extract_text_from_image(uploaded_file, api_key, model, task_type, max_tokens, temperature, top_p, repetition_penalty, pages=None):
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
        # ส่ง Request
        response = requests.post(url, files=files, data=data, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            extracted_texts = []
            
            # แกะ JSON Response ตามโครงสร้างของ Typhoon API
            for page_result in result.get('results', []):
                if page_result.get('success') and page_result.get('message'):
                    content = page_result['message']['choices'][0]['message']['content']
                    try:
                        # พยายามแปลง String JSON กลับเป็น Object เพื่อดึงเฉพาะ natural_text
                        parsed_content = json.loads(content)
                        text = parsed_content.get('natural_text', content)
                        # ถ้า text ยังเป็น dict/list ให้แปลงเป็น string
                        if isinstance(text, (dict, list)):
                            text = json.dumps(text, ensure_ascii=False)
                    except json.JSONDecodeError:
                        # ถ้าไม่ใช่ JSON ให้ใช้ content ดิบเลย
                        text = content
                    extracted_texts.append(text)
                elif not page_result.get('success'):
                    error_msg = f"Error: {page_result.get('error', 'Unknown error')}"
                    extracted_texts.append(f"[{error_msg}]")
            
            return '\n\n---\n\n'.join(extracted_texts)
        else:
            return f"API Error: {response.status_code}\n{response.text}"
            
    except Exception as e:
        return f"Connection Error: {str(e)}"

# --- 4. ส่วน Sidebar (Settings) ---
with st.sidebar:
    # Logo (อ้างอิงจากไฟล์ที่คุณเคยอัปโหลด ถ้าหาไม่เจอจะแสดงข้อความแทน)
    try:
        st.image("image_e05e9c.png", use_column_width=True)
    except:
        st.markdown("## 🌀 Typhoon OCR")

    st.markdown("---")
    
    # API Key Management
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        api_key = st.text_input("API Key", type="password", help="ใส่ API Key ของ Typhoon AI")
        if api_key:
            st.session_state["api_key"] = api_key
            st.rerun()
    else:
        st.success("✅ API Key เชื่อมต่อแล้ว")
        # ปุ่ม Logout เล็กๆ
        if st.button("Logout / Clear Key"):
            st.session_state["api_key"] = ""
            st.rerun()

    st.markdown("### ⚙️ การตั้งค่า (Advanced)")
    
    with st.expander("ปรับแต่งค่า Parameter", expanded=False):
        max_tokens = st.slider("Max Tokens", 1000, 16000, 16000, 100)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        top_p = st.slider("Top P", 0.0, 1.0, 0.6, 0.1)
        repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.1, 0.1)
        
        # Hidden fields (Fixed for this app)
        model = "typhoon-ocr"
        task_type = "v1.5"

    st.markdown("---")
    st.caption("Powered by **Typhoon AI**")

# --- 5. ส่วน Main Content ---
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
        
        # แสดง Preview
        if uploaded_file.type == "application/pdf":
            st.warning("⚠️ ไฟล์ PDF จะไม่แสดงตัวอย่าง แต่สามารถประมวลผลได้ปกติ")
        else:
            st.image(uploaded_file, use_column_width=True)
        
        st.markdown("---")
        
        # ตัวเลือกเสริม
        pages_input = st.text_input("ระบุหน้า (สำหรับ PDF)", placeholder="เช่น 1, 2 หรือ 1-5 (เว้นว่างเพื่อทำทั้งหมด)")
        
        # ปุ่ม Action
        if st.button("🚀 เริ่มประมวลผล (Start OCR)", type="primary", use_container_width=True):
            if not api_key:
                st.error("❌ กรุณากรอก API Key ที่แถบด้านซ้ายก่อน")
            else:
                with st.spinner("🌀 AI กำลังอ่านเอกสาร... โปรดรอสักครู่"):
                    # เรียกฟังก์ชันประมวลผล
                    result_text = extract_text_from_image(
                        uploaded_file, api_key, model, task_type, 
                        max_tokens, temperature, top_p, repetition_penalty, pages_input
                    )
                    # เก็บผลลัพธ์ลง Session State เพื่อไม่ให้หายเวลารีเฟรช
                    st.session_state["ocr_result"] = result_text
                    st.success("✅ เสร็จสิ้น!")

    # --- Column ขวา: Result ---
    with col2:
        st.info("📝 **ผลลัพธ์ข้อความ**")
        
        # ดึงผลลัพธ์จาก Session
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
            st.download_button(
                label="💾 ดาวน์โหลดไฟล์ .txt",
                data=result_text,
                file_name="ocr_result.txt",
                mime="text/plain"
            )

else:
    # หน้าจอว่างเปล่าเมื่อยังไม่ Upload
    st.container(border=True).markdown(
        """
        <div style="text-align: center; padding: 40px; color: #64748B;">
            <h3>👆 เริ่มต้นใช้งาน</h3>
            <p>กรุณาอัปโหลดไฟล์ภาพหรือ PDF ที่ด้านบนเพื่อเริ่มการ OCR</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
