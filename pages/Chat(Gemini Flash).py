import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from PyPDF2 import PdfReader

# --- Page Configuration ---
st.set_page_config(page_title="PA Assistant Chat (Gemini Flash)", page_icon="⚡", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
        <div class="sidebar-footer">
            <p><span style="color: grey;">By PAO1 </span><br>
            <span style="font-size: 16px;"><b>A</b>udit <b>I</b>ntelligence <b>T</b>eam</span></p>
        </div>
    """, unsafe_allow_html=True)

st.markdown(
    """<style>
    [data-testid="stAppViewContainer"] > .main { background-color: #e0f2f1; }
    h1 { font-size: 36px !important; }
    .block-container { padding-top: 2rem; }
    </style>""", unsafe_allow_html=True
)

st.title("💬 PA Assistant (Gemini 2.0 Flash)")
st.markdown("ถาม-ตอบผู้ช่วยอัจฉริยะ (อ่านเอกสารทั้งเล่ม - ไม่ต้องย่อ - ฟรี)")

# ----------------- Functions -----------------

@st.cache_data(show_spinner=False)
def get_all_text(uploaded_files):
    text = ""
    # 1. อ่านไฟล์ Local
    if os.path.isdir("Doc"):
        for filename in os.listdir("Doc"):
            try:
                path = os.path.join("Doc", filename)
                if filename.endswith('.pdf'):
                    reader = PdfReader(path)
                    for page in reader.pages: text += page.extract_text() or ""
                elif filename.endswith('.txt'):
                    with open(path, 'r', encoding='utf-8') as f: text += f.read()
            except: pass
            
    # 2. อ่านไฟล์ Uploaded
    if uploaded_files:
        for file in uploaded_files:
            try:
                if file.name.endswith('.pdf'):
                    reader = PdfReader(file)
                    for page in reader.pages: text += page.extract_text() or ""
                elif file.name.endswith('.txt'):
                    text += file.getvalue().decode("utf-8")
            except: pass
    return text

# ----------------- Session Init -----------------
if "chatbot_messages" not in st.session_state:
    st.session_state.chatbot_messages = [{"role": "assistant", "content": "สวัสดีครับ ผมใช้โมเดล Gemini 2.0 Flash อ่านเอกสารได้ยาวมาก ถามมาได้เลยครับ"}]

# ----------------- UI -----------------
with st.expander("อัปโหลดเอกสาร (PDF, TXT)"):
    uploaded_files = st.file_uploader("เลือกไฟล์...", type=['pdf', 'txt'], accept_multiple_files=True)

# อ่านไฟล์ทั้งหมดรวดเดียว (ไม่ต้องทำ Index)
all_text = get_all_text(uploaded_files)

if all_text:
    st.success(f"✅ โหลดเนื้อหาแล้ว: {len(all_text):,} ตัวอักษร")
else:
    st.info("ยังไม่มีเอกสาร")

# Chat UI
chat_container = st.container(height=450, border=True)
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
            
            try:
                # ดึง API Key
                api_key = st.secrets.get("openrouter_api_key", st.secrets.get("api_key", ""))
                
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

                # Prompt แบบใส่เนื้อหาเข้าไปตรงๆ (Context Stuffing)
                system_prompt = f"""
                คุณคือผู้ช่วย PA Assistant ผู้เชี่ยวชาญ
                
                ข้อมูลอ้างอิงจากเอกสารแนบ:
                ---
                {all_text}
                ---
                
                คำสั่ง:
                1. ตอบคำถามโดยใช้ข้อมูลข้างบนนี้เป็นหลัก
                2. ถ้าข้อมูลมีระบุไว้ ให้ตอบตามจริง
                3. ถ้าข้อมูลไม่มีระบุไว้ ให้บอกว่า "ในเอกสารไม่ได้ระบุเรื่อง... ไว้ครับ" (แต่อย่าตอบปฏิเสธทันทีถ้ายังไม่ได้หาดีๆ)
                """             

                stream = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://streamlit.io/",
                        "X-Title": "PA Assistant RAG",
                    },
                    # --- แก้บรรทัดนี้ครับ ---
                    model="meta-llama/llama-3.3-70b-instruct:free", 
                    # ----------------------
                    messages=[
                        {"role": "system", "content": system_prompt},
                    ] + st.session_state.chatbot_messages[-6:], 
                    stream=True
                )




                
                full_response = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                st.session_state.chatbot_messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error: {e}")
