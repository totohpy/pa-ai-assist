import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from PyPDF2 import PdfReader

# --- Page Configuration ---
st.set_page_config(page_title="PA Assistant Chat", page_icon="💬", layout="wide")

# --- Sidebar Configuration (empty to inherit from main page) ---
with st.sidebar:
    pass

# --- Apply Consistent Styling ---
st.markdown(
    """
     <style>
    /* --- Overall App Color Theme --- */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #e0f2f1;
    }
    [data-testid="stSidebar"] {
        background-color: #e0f2f1;
        width: 250px !important;
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

    /* --- Feature Box Styling (Main Page) --- */
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
        color: #263238 !important;     /* Darker text for inactive links */
        background-color: #b2dfdb;     /* Light teal for inactive links */
        border: 1px solid #9dbdb9;
        font-weight: 500;
    }
    
    /* Style the ACTIVE page link */
    div[data-testid="stSidebarNav"] a[aria-current="page"] {
        background-color: #80cbc4;     /* Dark teal for active link */
        color: #FFFFFF !important;     /* White text for active link */
        font-weight: 600;
        border: 1px solid #00796b;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("💬 PA Assistant Chat")
st.markdown("ถาม-ตอบผู้ช่วยอัจฉริยะด้านการตรวจสอบ")

# ----------------- Functions for Chatbot -----------------
MAX_CHARS_LIMIT = 300000

@st.cache_data(show_spinner=False)
def load_local_documents(folder_path="Doc"):
    """Reads all files from the local document library."""
    text = ""
    if not os.path.isdir(folder_path):
        return text 

    try:
        files_in_doc = os.listdir(folder_path)
        # Use a temporary placeholder for progress bar if sidebar is not always visible
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
                        for page in reader.pages:
                            text += page.extract_text() or ""
                elif filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text += f.read()
                elif filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    text += df.head(15).to_string()
            except Exception as e:
                print(f"Could not read file {filename}: {e}")
            
            progress_placeholder.progress((i + 1) / len(files_in_doc), text=f"กำลังโหลดเอกสาร... ({i+1}/{len(files_in_doc)})")
        progress_placeholder.empty()
                
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการเข้าถึงคลังข้อมูล: {e}")
    
    return text[:MAX_CHARS_LIMIT]

def process_documents(files, source_type, limit, current_len=0):
    """Function to read text from uploaded files."""
    text = ""
    for file in files:
        if current_len + len(text) >= limit:
            st.warning(f"ถึงขีดจำกัดตัวอักษรสูงสุด ({limit:,}) แล้ว บางไฟล์อาจไม่ถูกประมวลผล")
            break
        try:
            if file.name.endswith('.pdf'):
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            elif file.name.endswith('.txt'):
                text += file.getvalue().decode("utf-8")
            elif file.name.endswith('.csv'):
                df = pd.read_csv(file)
                text += df.head(15).to_string()
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ {file.name}: {e}")
    return text[:limit - current_len], [f.name for f in files]

# ----------------- Session Init (for Chatbot only) -----------------
def init_chat_state():
    ss = st.session_state
    ss.setdefault('chatbot_messages', [{"role": "assistant", "content": "สวัสดีครับ ผมคือ PA Chat Assistant ผู้ช่วยอัจฉริยะด้านการตรวจสอบ"}])
    ss.setdefault('doc_context_uploaded', "")
    ss.setdefault('last_uploaded_files', set())

    # Load API Key from secrets
    try:
        ss.setdefault('api_key_global', st.secrets["api_key"])
    except (KeyError, FileNotFoundError):
        ss.setdefault('api_key_global', "")
        # A warning will be shown in the main UI if key is missing

    # Load local documents only once
    if 'doc_context_local' not in ss:
        ss.doc_context_local = load_local_documents()
        if ss.doc_context_local and os.path.isdir('Doc'):
             ss.chatbot_messages.append({"role": "assistant", "content": f"ผมได้โหลดเอกสารเป็นฐานความรู้แล้ว"})

init_chat_state()

# ----------------- Chatbot UI -----------------
with st.expander("อัปโหลดเอกสารเพิ่มเติม (PDF, TXT, CSV)"):
    st.info("ข้อมูลจากไฟล์ที่อัปโหลดจะถูกใช้ในการตอบคำถาม")
    uploaded_files = st.file_uploader(
        "เลือกไฟล์...",
        type=['pdf', 'txt', 'csv'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

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
    if local_len > 0:
        st.info(f"💾 เนื้อหาจากคลังข้อมูล: {local_len:,} ตัวอักษร")
    if uploaded_len > 0:
        st.info(f"📤 เนื้อหาจากไฟล์ที่อัปโหลด: {uploaded_len:,} ตัวอักษร")
    st.success(f"✅ เนื้อหารวมทั้งหมด: {(local_len + uploaded_len):,} ตัวอักษร (สูงสุด: {MAX_CHARS_LIMIT:,})")

chat_container = st.container(height=500, border=True)
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
            
            api_key = st.session_state.api_key_global
            if not api_key:
                error_message = "เกิดข้อผิดพลาด: ไม่พบ API Key กรุณาติดต่อผู้ดูแลระบบ"
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
- อ้างอิงเสมอว่าข้อมูลมาจากแหล่งใด (เช่น "จากเอกสาร [ชื่อไฟล์]...", "จากข้อมูลที่ให้มา...")
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

                    client = OpenAI(api_key=api_key, base_url="https://api.opentyphoon.ai/v1")
                    response_stream = client.chat.completions.create(
                        model="typhoon-v2.1-12b-instruct", 
                        messages=messages_for_api, 
                        temperature=0.5, 
                        max_tokens=3072, 
                        stream=True
                    )
                    response = message_placeholder.write_stream(response_stream)
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_message = f"เกิดข้อผิดพลาดขณะประมวลผล: {e}"
                    message_placeholder.error(error_message)
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": error_message})
