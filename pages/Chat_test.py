import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from PyPDF2 import PdfReader

# --- Import Libraries สำหรับ RAG ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- Page Configuration ---
st.set_page_config(page_title="PA Assistant Chat (RAG)", page_icon="💬", layout="wide")

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

st.markdown(
    """
     <style>
    [data-testid="stAppViewContainer"] > .main { background-color: #e0f2f1; }
    h1 { font-size: 36px !important; }
    [data-testid="stSidebar"] { background-color: #e0f2f1; width: 250px !important; }
    .sidebar-footer { width: 100%; padding: 1rem; text-align: center; }
    .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💬 PA Assistant Chat (RAG System)")
st.markdown("ถาม-ตอบผู้ช่วยอัจฉริยะ (ระบบค้นหาข้อมูลแม่นยำ - รองรับเอกสารไม่จำกัด)")

# ----------------- RAG Functions -----------------

@st.cache_resource(show_spinner=False)
def build_vector_store(text_content, api_key):
    """
    ฟังก์ชันสร้างฐานข้อมูลเวกเตอร์ (Vector Store) จากข้อความ
    1. แบ่งข้อความ (Split)
    2. แปลงเป็นตัวเลข (Embed)
    3. เก็บเข้า FAISS
    """
    if not text_content or not api_key:
        return None
    
    try:
        # 1. แบ่งข้อความออกเป็นชิ้นย่อยๆ (Chunks)
        # ภาษาไทยอาจต้องใช้ chunk_size ที่เหมาะสม ประมาณ 1000-2000 chars
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300, # ให้เนื้อหาเกยกันนิดหน่อยกันใจความขาด
            separators=["\n\n", "\n", " ", ""]
        )
        
        # สร้าง Document Objects
        docs = [Document(page_content=x) for x in text_splitter.split_text(text_content)]
        
        # 2. สร้าง Embeddings และ Vector Store
        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            base_url="https://openrouter.ai/api/v1", # ใช้ OpenRouter สำหรับ Embeddings (หรือใช้ OpenAI ตรงๆ ก็ได้)
            model="text-embedding-3-small", # โมเดล Embeddings มาตรฐาน
            check_embedding_ctx_length=False 
        )
        
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการสร้าง Index: {e}")
        return None

def get_relevant_context(vector_store, query):
    """ค้นหาเนื้อหาที่เกี่ยวข้องกับคำถามมากที่สุด 4-5 ชิ้น"""
    if not vector_store:
        return ""
    
    # Search หา 5 ชิ้นที่ใกล้เคียงที่สุด
    docs = vector_store.similarity_search(query, k=5)
    
    # รวมเนื้อหา
    context = "\n\n".join([f"[เนื้อหาที่ {i+1}]: {d.page_content}" for i, d in enumerate(docs)])
    return context

# ฟังก์ชันอ่านไฟล์ (เหมือนเดิมแต่ตัด limit ออก)
def extract_text_from_files(files, folder_path="Doc"):
    text = ""
    
    # 1. อ่านจาก Folder Local
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if filename.endswith('.pdf'):
                    with open(file_path, 'rb') as f:
                        reader = PdfReader(f)
                        for page in reader.pages: text += page.extract_text() or ""
                elif filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: text += f.read()
            except: pass

    # 2. อ่านจาก Uploaded Files
    if files:
        for file in files:
            try:
                if file.name.endswith('.pdf'):
                    reader = PdfReader(file)
                    for page in reader.pages: text += page.extract_text() or ""
                elif file.name.endswith('.txt'): text += file.getvalue().decode("utf-8")
                elif file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                    text += df.to_string()
            except: pass
            
    return text

# ----------------- Session Init -----------------
def init_chat_state():
    ss = st.session_state
    ss.setdefault('chatbot_messages', [{"role": "assistant", "content": "สวัสดีครับ ผมใช้ระบบ RAG ค้นหาข้อมูลเฉพาะจุด ทำให้ตอบคำถามจากเอกสารยาวๆ ได้แม่นยำครับ"}])
    ss.setdefault('vector_store', None)
    ss.setdefault('last_processed_files', set())

init_chat_state()

# ----------------- UI & Logic -----------------

with st.expander("อัปโหลดเอกสารเพิ่มเติม (PDF, TXT, CSV)"):
    uploaded_files = st.file_uploader("เลือกไฟล์...", type=['pdf', 'txt', 'csv'], accept_multiple_files=True)

# ตรวจสอบว่าต้องสร้าง Vector Store ใหม่หรือไม่ (เมื่อมีการอัปโหลดไฟล์เพิ่ม หรือเพิ่งเริ่ม)
current_files_set = {f.name for f in uploaded_files} if uploaded_files else set()
is_files_changed = current_files_set != st.session_state.last_processed_files
is_first_load = st.session_state.vector_store is None

if is_files_changed or is_first_load:
    # ดึง API Key
    try:
        api_key = st.secrets["openrouter_api_key"]
    except:
        api_key = ""

    if api_key:
        with st.spinner("กำลังสร้างดัชนีข้อมูล (RAG Indexing)..."):
            # 1. อ่านข้อความทั้งหมด
            raw_text = extract_text_from_files(uploaded_files)
            
            if raw_text:
                # 2. สร้าง Vector Store
                st.session_state.vector_store = build_vector_store(raw_text, api_key)
                st.session_state.last_processed_files = current_files_set
                st.success(f"✅ สร้างฐานข้อมูลเรียบร้อย! (จากเนื้อหา {len(raw_text):,} ตัวอักษร)")
            else:
                st.warning("ยังไม่มีข้อมูลเอกสาร กรุณาอัปโหลดไฟล์หรือใส่ไฟล์ในโฟลเดอร์ Doc")

# Chat UI
chat_container = st.container(height=400, border=True)
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
                api_key = st.secrets["openrouter_api_key"]
                vector_store = st.session_state.vector_store
                
                # --- RAG Step 1: ค้นหาเนื้อหาที่เกี่ยวข้อง (Retrieval) ---
                if vector_store:
                    context_text = get_relevant_context(vector_store, prompt)
                else:
                    context_text = "ไม่มีเอกสารให้อ้างอิง ตอบตามความรู้ทั่วไป"

                # --- RAG Step 2: สร้าง Prompt ---
                system_prompt = f"""
คุณคือผู้ช่วย AI ผู้เชี่ยวชาญด้านการตรวจสอบ (PA Assistant)
หน้าที่: ตอบคำถามโดยใช้ข้อมูลจาก "บริบทที่ค้นพบ" ด้านล่างนี้เป็นหลัก
กฎ:
- อ้างอิงข้อมูลจากบริบทที่ให้มาเท่านั้น
- ถ้าข้อมูลในบริบทไม่เพียงพอ ให้ตอบว่า "ขออภัย ข้อมูลในเอกสารไม่เพียงพอต่อการตอบคำถามนี้"
- ห้ามมั่วข้อมูลขึ้นมาเอง

--- บริบทที่ค้นพบ (Context) ---
{context_text}
-----------------------------
"""
                messages_for_api = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]

                # --- RAG Step 3: ส่งให้ LLM ตอบ (Generation) ---
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
                
                stream = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://streamlit.io/",
                        "X-Title": "PA Assistant RAG",
                    },
                    model="openai/gpt-4o",
                    messages=messages_for_api,
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
