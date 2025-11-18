import streamlit as st
import requests
import json

# ตั้งค่าหน้าเพจ
st.set_page_config(page_title="Typhoon OCR", layout="wide")

# ฟังก์ชันสำหรับเรียก API
def extract_text_from_image(uploaded_file, api_key, model, task_type, max_tokens, temperature, top_p, repetition_penalty, pages=None):
    url = "https://api.opentyphoon.ai/v1/ocr"
    
    # เตรียมไฟล์สำหรับส่ง (Streamlit file_uploader ส่งคืน BytesIO object)
    # เราต้องระบุชื่อไฟล์และ mime type ให้ชัดเจนเพื่อให้ requests ส่งถูกต้อง
    files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
    
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
        response = requests.post(url, files=files, data=data, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            extracted_texts = []
            
            # Extract text logic ตามต้นฉบับ
            for page_result in result.get('results', []):
                if page_result.get('success') and page_result.get('message'):
                    content = page_result['message']['choices'][0]['message']['content']
                    try:
                        # Try to parse as JSON if it's structured output
                        parsed_content = json.loads(content)
                        # พยายามดึง natural_text หรือใช้ทั้งก้อนถ้าไม่มี key นั้น
                        text = parsed_content.get('natural_text', content)
                        if isinstance(text, (dict, list)): # กรณี natural_text ยังเป็น struct
                            text = json.dumps(text, ensure_ascii=False)
                    except json.JSONDecodeError:
                        text = content
                    extracted_texts.append(text)
                elif not page_result.get('success'):
                    error_msg = f"Error processing {page_result.get('filename', 'unknown')}: {page_result.get('error', 'Unknown error')}"
                    extracted_texts.append(f"[{error_msg}]")
            
            return '\n\n---\n\n'.join(extracted_texts)
        else:
            return f"Error: {response.status_code}\n{response.text}"
            
    except Exception as e:
        return f"An error occurred: {str(e)}"

# --- UI Section ---

st.title("Typhoon OCR")

# ตรวจสอบ API Key (สมมติว่าเก็บใน session_state เหมือนหน้าอื่นๆ ตามที่คุณแจ้ง)
# ถ้าไม่มีใน session_state จะลองดึงจาก secrets หรือแสดงช่องกรอก
api_key = st.session_state.get("api_key")
if not api_key:
    # Fallback: เผื่อกรณียังไม่ได้ login หรือ set key ในหน้า Home
    api_key = st.text_input("Enter API Key", type="password")

col1, col2 = st.columns([1, 1])

with col1:
    # File Upload (ย้ายมาไว้บนสุด)
    uploaded_file = st.file_uploader("Upload Image or PDF", type=['png', 'jpg', 'jpeg', 'webp', 'pdf'])
    
    # Pages Input
    pages_input = st.text_input("Pages (optional)", placeholder="e.g., [1, 2] or 1-3")
    
    # ปุ่ม Start
    start_btn = st.button("Start OCR", type="primary", use_container_width=True, disabled=(not uploaded_file or not api_key))

    # Advanced Settings ใน Expander (ย้ายมาไว้ล่างสุด)
    with st.expander("Advanced Settings"):
        # Hidden fields logic (model, task_type) - เรากำหนดค่าตรงๆ ในโค้ดเรียก API ได้เลย
        # แต่ถ้าต้องการให้แก้ได้ ก็ใส่ input ไว้ตรงนี้
        model = "typhoon-ocr" 
        task_type = "v1.5"
        
        max_tokens = st.slider("Max Tokens", min_value=1000, max_value=16000, value=16000, step=100)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
        repetition_penalty = st.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.1, step=0.1)

with col2:
    st.markdown("### Extracted Text")
    output_area = st.empty()
    
    if start_btn:
        if not uploaded_file:
            st.error("Please upload a file first.")
        elif not api_key:
            st.error("API Key is missing.")
        else:
            with st.spinner("Processing..."):
                # เรียกฟังก์ชัน OCR
                extracted_text = extract_text_from_image(
                    uploaded_file, 
                    api_key, 
                    model, 
                    task_type, 
                    max_tokens, 
                    temperature, 
                    top_p, 
                    repetition_penalty, 
                    pages_input
                )
                
                # แสดงผลลัพธ์
                output_area.text_area("Result", value=extracted_text, height=600)
