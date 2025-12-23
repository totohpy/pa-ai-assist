import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from PyPDF2 import PdfReader

# --- Page Configuration ---
st.set_page_config(page_title="PA Assistant Chat", page_icon="💬", layout="wide")

# --- Sidebar Configuration (Updated to match Home.py) ---
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
    /* --- Overall App Color Theme --- */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #e0f2f1;
    }
    h1 { font-size: 36px !important; }
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
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize api_key from secrets or session state
api_key = st.secrets.get("api_key", st.session_state.get("api_key", ""))

# --- Helper Functions ---
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def extract_text_from_excel(file):
    df = pd.read_excel(file)
    return df.to_csv(index=False)

def init_chat_history():
    if "chatbot_messages" not in st.session_state:
        st.session_state.chatbot_messages = []
    if "file_context" not in st.session_state:
        st.session_state.file_context = "" # จากการอัปโหลด
    if "doc_folder_context" not in st.session_state:
        st.session_state.doc_folder_context = "" # จากโฟลเดอร์ Doc
    if "loaded_doc_files" not in st.session_state:
        st.session_state.loaded_doc_files = []

init_chat_history()

# --- Load Documents from 'Doc' Folder (ADDED) ---
def load_documents_from_folder(folder_path="Doc"):
    """อ่านเอกสารทั้งหมดจากโฟลเดอร์ที่ระบุ"""
    combined_text = ""
    loaded_files = []
    
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                text = ""
                if filename.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(file_path)
                elif filename.lower().endswith(('.xlsx', '.xls')):
                    text = extract_text_from_excel(file_path)
                
                if text:
                    combined_text += f"\n\n--- ข้อมูลจากไฟล์ในโฟลเดอร์ Doc: {filename} ---\n{text}"
                    loaded_files.append(filename)
            except Exception as e:
                # แสดง error เล็กๆ ใน sidebar ถ้าอ่านไฟล์ไหนไม่ได้
                st.sidebar.error(f"Error reading {filename}: {e}")
                
    return combined_text, loaded_files

# โหลดข้อมูลจากโฟลเดอร์ Doc (ถ้ายังไม่เคยโหลด)
if not st.session_state.doc_folder_context:
    folder_text, doc_files = load_documents_from_folder()
    if folder_text:
        st.session_state.doc_folder_context = folder_text
        st.session_state.loaded_doc_files = doc_files

# แสดงรายการไฟล์ที่โหลดจากโฟลเดอร์ Doc ใน Sidebar
if st.session_state.loaded_doc_files:
    with st.sidebar.expander("📂 เอกสารในโฟลเดอร์ Doc", expanded=True):
        for f in st.session_state.loaded_doc_files:
            st.caption(f"📄 {f}")

# --- Main Interface ---
st.title("💬 PA Assistant Chat")
st.caption("สอบถามข้อมูล หรือให้ช่วยวิเคราะห์เอกสาร (PDF/Excel)")

# File Uploader
uploaded_file = st.file_uploader("แนบเอกสารอ้างอิง (Optional)", type=["pdf", "xlsx", "xls"])

if uploaded_file:
    with st.spinner("กำลังอ่านไฟล์..."):
        try:
            if uploaded_file.name.endswith('.pdf'):
                file_text = extract_text_from_pdf(uploaded_file)
            else:
                file_text = extract_text_from_excel(uploaded_file)
            
            # Limit context size to prevent token overflow (approx limit)
            st.session_state.file_context = f"\n\n--- ข้อมูลจากไฟล์ที่แนบ: {uploaded_file.name} ---\n{file_text}"
            st.success(f"อ่านไฟล์ '{uploaded_file.name}' เรียบร้อยแล้ว")
            
            with st.expander("ดูเนื้อหาไฟล์ที่อ่านได้"):
                st.text(st.session_state.file_context[:1000] + "...")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")

# Clear Chat Button
if st.sidebar.button("Clear Chat"):
    st.session_state.chatbot_messages = []
    st.session_state.file_context = "" 
    # หมายเหตุ: เราไม่เคลียร์ doc_folder_context เพราะเป็นไฟล์ถาวร
    st.rerun()

# Display Chat History
for message in st.session_state.chatbot_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):
    if not api_key:
        st.error("กรุณากรอก API Key ก่อนเริ่มใช้งาน")
    else:
        # Add user message to state and display
        st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # รวม Context จากทั้ง Folder และ Uploaded File
            full_context = st.session_state.doc_folder_context + st.session_state.file_context
            
            # ตัด Context ถ้าเยอะเกินไป (ป้องกัน Token เต็ม)
            if len(full_context) > 30000:
                full_context = full_context[:30000] + "\n...(ตัดเนื้อหาบางส่วน)..."

            context_str = ""
            if full_context:
                context_str = f"\n\nข้อมูลอ้างอิงจากเอกสาร:\n{full_context}"
            
            system_prompt = (
                "คุณคือผู้ช่วยผู้ตรวจสอบภายใน (PA Assistant) ที่มีความเชี่ยวชาญ "
                "จงตอบคำถามอย่างมืออาชีพ กระชับ และตรวจสอบข้อมูลจากเอกสารที่แนบมาถ้ามี"
                f"{context_str}"
            )
            
            messages_for_api = [
                {"role": "system", "content": system_prompt}
            ] + st.session_state.chatbot_messages[-10:] # ส่งประวัติ 10 ข้อความล่าสุด

            try:
                client = OpenAI(api_key=api_key, base_url="https://api.opentyphoon.ai/v1")
                
                # Use a placeholder to display streaming output
                with message_placeholder:
                    full_response = ""
                    response_stream = client.chat.completions.create(
                        model="typhoon-v2.1-12b-instruct",
                        messages=messages_for_api,
                        temperature=0.5,
                        max_tokens=3072,
                        stream=True
                    )
                
                    # Accumulate and display each chunk
                    for chunk in response_stream:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")  # Cursor effect
                
                    # Final clean-up
                    message_placeholder.markdown(full_response)
            
                # Now safely append the complete response
                st.session_state.chatbot_messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                error_message = f"เกิดข้อผิดพลาดขณะประมวลผล: {e}"
                message_placeholder.error(error_message)
                st.session_state.chatbot_messages.append({"role": "assistant", "content": error_message})
