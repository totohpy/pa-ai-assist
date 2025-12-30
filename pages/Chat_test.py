import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from PyPDF2 import PdfReader

# --- Page Configuration ---
st.set_page_config(page_title="PA Assistant Chat", page_icon="💬", layout="wide")

# --- Sidebar Configuration ---
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

# --- Apply Consistent Styling ---
st.markdown(
    """
     <style>
    [data-testid="stAppViewContainer"] > .main { background-color: #e0f2f1; }
    h1 { font-size: 36px !important; }
    [data-testid="stSidebar"] { background-color: #e0f2f1; width: 250px !important; }
    [data-testid="stSidebar"] > div:first-child { display: flex; flex-direction: column; height: 100%; }
    [data-testid="stSidebarNav"] { flex-grow: 1; margin-top: 20px; }
    .sidebar-footer { width: 100%; padding: 1rem; text-align: center; }
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💬 PA Assistant Chat (OpenRouter)")
st.markdown("ถาม-ตอบผู้ช่วยอัจฉริยะด้านการตรวจสอบ (Powered by GPT-4o)")

# ----------------- Functions for Chatbot -----------------
MAX_CHARS_LIMIT = 75000

@st.cache_data(show_spinner=False)
def load_local_documents(folder_path="Doc"):
    text = ""
    if not os.path.isdir(folder_path): return text 
    try:
        files_in_doc = os.listdir(folder_path)
        progress_placeholder = st.empty()
        for i, filename in enumerate(files_in_doc):
            if len(text) >= MAX_CHARS_LIMIT:
                st.warning(f"ถึงขีดจำกัดข้อมูลแล้ว ({MAX_CHARS_LIMIT:,} ตัวอักษร)")
                break
            file_path = os.path.join(folder_path, filename)
            try:
                if filename.endswith('.pdf'):
                    with open(file_path, 'rb') as f:
                        reader = PdfReader(f)
                        for page in reader.pages: text += page.extract_text() or ""
                elif filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: text += f.read()
                elif filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    text += df.head(15).to_string()
            except Exception as e: print(f"Error reading {filename}: {e}")
            progress_placeholder.progress((i + 1) / len(files_in_doc), text=f"กำลังโหลดเอกสาร... ({i+1}/{len(files_in_doc)})")
        progress_placeholder.empty()
    except Exception as e: st.error(f"Error loading docs: {e}")
    return text[:MAX_CHARS_LIMIT]

def process_documents(files, source_type, limit, current_len=0):
    text = ""
    for file in files:
        if current_len + len(text) >= limit:
            st.warning(f"Limit reached ({limit:,}). Some files ignored.")
            break
        try:
            if file.name.endswith('.pdf'):
                reader = PdfReader(file)
                for page in reader.pages: text += page.extract_text() or ""
            elif file.name.endswith('.txt'): text += file.getvalue().decode("utf-8")
            elif file.name.endswith('.csv'):
                df = pd.read_csv(file)
                text += df.head(15).to_string()
        except Exception as e: st.error(f"Error reading {file.name}: {e}")
    return text[:limit - current_len], [f.name for f in files]

# ----------------- Session Init -----------------
def init_chat_state():
    ss = st.session_state
    ss.setdefault('chatbot_messages', [{"role": "assistant", "content": "สวัสดีครับ ผมคือ PA Chat Assistant ผู้ช่วยอัจฉริยะด้านการตรวจสอบ"}])
    ss.setdefault('doc_context_uploaded', "")
    ss.setdefault('last_uploaded_files', set())
    
    # หมายเหตุ: ผมลบ logic การโหลด api_key_global ออกจากตรงนี้แล้ว 
    # เพื่อป้องกันการตีกันกับหน้าอื่น

    if 'doc_context_local' not in ss:
        with st.spinner("กำลังโหลดคลังข้อมูลตั้งต้น..."):
            ss.doc_context_local = load_local_documents()
        if ss.doc_context_local and os.path.isdir('Doc'):
             ss.chatbot_messages.append({"role": "assistant", "content": f"ผมได้โหลดเอกสารเป็นฐานความรู้แล้ว"})

init_chat_state()

# ----------------- Chatbot UI -----------------
with st.expander("อัปโหลดเอกสารเพิ่มเติม (PDF, TXT, CSV)"):
    st.info("ข้อมูลจากไฟล์ที่อัปโหลดจะถูกใช้ในการตอบคำถาม")
    uploaded_files = st.file_uploader("เลือกไฟล์...", type=['pdf', 'txt', 'csv'], accept_multiple_files=True, label_visibility="collapsed")

current_uploaded_file_names = {f.name for f in uploaded_files}
if uploaded_files and st.session_state.get('last_uploaded_files') != current_uploaded_file_names:
    with st.spinner("กำลังประมวลผลเอกสาร..."):
        st.session_state.doc_context_uploaded, _ = process_documents(uploaded_files, 'uploaded', MAX_CHARS_LIMIT, len(st.session_state.get('doc_context_local', '')))
        st.session_state.last_uploaded_files = current_uploaded_file_names
        st.session_state.chatbot_messages.append({"role": "assistant", "content": "อัปเดตเอกสารใหม่แล้ว"})
        st.rerun()
elif not uploaded_files and st.session_state.doc_context_uploaded:
    st.session_state.doc_context_uploaded = ""
    st.session_state.last_uploaded_files = set()
    st.session_state.chatbot_messages.append({"role": "assistant", "content": "ล้างเอกสารที่อัปโหลดแล้ว"})
    st.rerun()

local_len = len(st.session_state.get('doc_context_local', ''))
uploaded_len = len(st.session_state.get('doc_context_uploaded', ''))

with st.expander("ดูรายละเอียด Context"):
    if local_len > 0: st.info(f"💾 เนื้อหาจากคลังข้อมูล: {local_len:,} ตัวอักษร")
    if uploaded_len > 0: st.info(f"📤 เนื้อหาจากไฟล์ที่อัปโหลด: {uploaded_len:,} ตัวอักษร")
    st.success(f"✅ เนื้อหารวมทั้งหมด: {(local_len + uploaded_len):,} ตัวอักษร (สูงสุด: {MAX_CHARS_LIMIT:,})")

chat_container = st.container(height=320, border=True)
with chat_container:
    for message in st.session_state.chatbot_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("พิมพ์คำถามของคุณ...", key="chat_input_main"):
    st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
    
    with chat_container:
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # --- ตรงนี้ครับ: ดึง Key เฉพาะของ OpenRouter ---
            # จะไม่ไปยุ่งกับ st.session_state.api_key ของหน้าอื่นๆ
            try:
                openrouter_key = st.secrets["openrouter_api_key"]
            except (KeyError, FileNotFoundError):
                openrouter_key = None

            if not openrouter_key:
                error_message = "เกิดข้อผิดพลาด: ไม่พบ `openrouter_api_key` ใน Secrets.toml"
                message_placeholder.error(error_message)
                st.session_state.chatbot_messages.append({"role": "assistant", "content": error_message})
            else:
                try:
                    doc_context = st.session_state.get('doc_context_local', '') + st.session_state.get('doc_context_uploaded', '')
                    
                    system_prompt = f"""
คุณคือผู้ช่วย AI อัจฉริยะ หน้าที่ของคุณคือตอบคำถามของผู้ใช้ให้ถูกต้อง โดยใช้ข้อมูลจากสองแหล่ง:
1.  **เอกสารภายใน (Primary Source):** เนื้อหาจากไฟล์ในระบบ ให้ยึดข้อมูลนี้เป็นหลักเสมอ
2.  **ความรู้ทั่วไป (Secondary Source):** หากคำตอบไม่มีในเอกสาร ให้ใช้ความรู้ทั่วไป
**กฎการตอบ:**
- อ้างอิงเสมอว่าข้อมูลมาจากแหล่งใด
- หากข้อมูลขัดแย้งกัน ให้ยึดข้อมูลในเอกสารเป็นหลัก
- หากไม่พบคำตอบ ให้ตอบว่า "ขออภัยครับ ไม่พบข้อมูลที่เกี่ยวข้อง"
---
**บริบทจากเอกสารภายใน:**
{doc_context}
---
จากข้อมูลข้างต้นนี้ จงตอบคำถามล่าสุดของผู้ใช้
"""                   
                    messages_for_api = [
                        {"role": "system", "content": system_prompt}
                    ] + st.session_state.chatbot_messages[-10:]
                    
                    # 1. SETUP CLIENT เฉพาะหน้านี้
                    client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=openrouter_key, # ใช้ตัวแปร Local ที่ดึงมาตะกี้
                    )
                
                    full_response = ""
                    
                    # 2. CALL API
                    response_stream = client.chat.completions.create(
                        extra_headers={
                            "HTTP-Referer": "https://streamlit.io/", 
                            "X-Title": "PA Assistant Chat",
                        },
                        model="openai/gpt-4o",
                        messages=messages_for_api,
                        temperature=0.5,
                        stream=True 
                    )
                    
                    for chunk in response_stream:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    error_message = f"เกิดข้อผิดพลาดขณะประมวลผล: {e}"
                    message_placeholder.error(error_message)
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": error_message})
