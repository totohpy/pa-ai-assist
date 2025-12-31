import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from PyPDF2 import PdfReader

# --- Import Libraries สำหรับ RAG ---
# ตรวจสอบว่าใน requirements.txt มี langchain-text-splitters แล้ว
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

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

st.title("💬 PA Assistant Chat")
st.markdown("ถาม-ตอบผู้ช่วยอัจฉริยะ (ตอบแบบมืออาชีพอ้างอิงคู่มือการปฎิบัติงานและผลการตรวจสอบที่ผ่านมา)")

# ----------------- RAG Functions -----------------

@st.cache_resource(show_spinner=False)
def build_vector_store(text_content, api_key):
    """
    ฟังก์ชันสร้างฐานข้อมูลเวกเตอร์ (Vector Store) จากข้อความ
    """
    if not text_content or not api_key:
        return None
    
    try:
        # 1. แบ่งข้อความออกเป็นชิ้นย่อยๆ (Chunks)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # สร้าง Document Objects
        docs = [Document(page_content=x) for x in text_splitter.split_text(text_content)]
        
        # 2. สร้าง Embeddings และ Vector Store
        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            model="text-embedding-3-small",
            check_embedding_ctx_length=False 
        )
        
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการสร้าง Index: {e}")
        return None

def get_relevant_context(vector_store, query):
    """ค้นหาเนื้อหาที่เกี่ยวข้องกับคำถามมากที่สุด 5 ชิ้น"""
    if not vector_store:
        return ""
    
    # Search หา 5 ชิ้นที่ใกล้เคียงที่สุด
    docs = vector_store.similarity_search(query, k=5)
    
    # รวมเนื้อหา
    context = "\n\n".join([f"[เนื้อหาที่ {i+1}]: {d.page_content}" for i, d in enumerate(docs)])
    return context

def rewrite_query(user_question, chat_history, client):
    """
    ฟังก์ชันสำหรับแปลงคำถามสั้นๆ ให้เป็นคำถามที่สมบูรณ์ โดยดูบริบทจากประวัติการแชท
    """
    # แปลง History 6 ข้อความล่าสุดให้เป็น String
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-6:]])
    
    system_prompt_rewrite = f"""
    คุณคือ AI ที่ทำหน้าที่ "เรียบเรียงประโยคคำถามใหม่" (Query Rewriter)
    หน้าที่ของคุณ:
    1. อ่านประวัติการสนทนา (Chat History) และคำถามล่าสุดของผู้ใช้
    2. ถ้าคำถามล่าสุดเชื่อมโยงกับบริบทก่อนหน้า ให้เขียนคำถามใหม่ให้สมบูรณ์และชัดเจนขึ้น (ระบุประธาน/กรรม ให้ครบ)
    3. ถ้าคำถามล่าสุดเป็นเรื่องใหม่ ไม่เกี่ยวกับบริบทเดิม ให้คืนค่าคำถามเดิมกลับมา
    4. **ไม่ต้องตอบคำถาม** แค่เรียบเรียงประโยคคำถามใหม่เท่านั้น
    5. ผลลัพธ์ต้องเป็น "ภาษาไทย" เท่านั้น

    ตัวอย่าง:
    History: User: การแจ้งผลตรวจสอบทำอย่างไร?, AI: ต้องทำหนังสือแจ้ง...
    Current Question: ต้องทำถึงใครบ้าง?
    Rewritten Question: ในการแจ้งผลการตรวจสอบ ต้องทำหนังสือแจ้งถึงใครบ้าง?

    --- Chat History ---
    {history_text}
    --------------------
    Current Question: {user_question}
    """
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o", 
            messages=[
                {"role": "system", "content": system_prompt_rewrite},
                {"role": "user", "content": "Rewritten Question:"}
            ],
            temperature=0.3,
            max_tokens=200
        )
        new_question = response.choices[0].message.content.strip()
        return new_question
    except Exception as e:
        print(f"Error rewriting: {e}")
        return user_question # ถ้า Error ให้ใช้คำถามเดิม

# ฟังก์ชันอ่านไฟล์
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
    ss.setdefault('chatbot_messages', [{"role": "assistant", "content": "สวัสดีครับ ผมคือ PA Assistant ระบบ RAG ที่ปรับปรุงใหม่ พร้อมให้บริการตรวจสอบข้อมูลครับ"}])
    ss.setdefault('vector_store', None)
    ss.setdefault('last_processed_files', set())

init_chat_state()

# ----------------- UI & Logic -----------------

with st.expander("อัปโหลดเอกสารเพิ่มเติม (PDF, TXT, CSV)"):
    uploaded_files = st.file_uploader("เลือกไฟล์...", type=['pdf', 'txt', 'csv'], accept_multiple_files=True)

# ตรวจสอบว่าต้องสร้าง Vector Store ใหม่หรือไม่
current_files_set = {f.name for f in uploaded_files} if uploaded_files else set()
is_files_changed = current_files_set != st.session_state.last_processed_files
is_first_load = st.session_state.vector_store is None

if is_files_changed or is_first_load:
    try:
        api_key = st.secrets["openrouter_api_key"]
    except:
        api_key = ""

    if api_key:
        with st.spinner("กำลังสร้างดัชนีข้อมูล (RAG Indexing)..."):
            raw_text = extract_text_from_files(uploaded_files)
            
            if raw_text:
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
    # แสดงคำถาม User
    st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
    
    with chat_container:
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                api_key = st.secrets["openrouter_api_key"]
                
                # สร้าง Client
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )

                # --- STEP 1: Query Rewriting (หัวใจสำคัญของการจำบริบท) ---
                has_history = len(st.session_state.chatbot_messages) > 1
                
                if has_history:
                    # ส่งประวัติเก่า (ไม่รวมข้อล่าสุด) ไปให้ AI ช่วยเกลาคำถาม
                    with st.spinner("กำลังทบทวนบริบท..."): 
                        search_query = rewrite_query(prompt, st.session_state.chatbot_messages[:-1], client)
                else:
                    search_query = prompt

                # --- STEP 2: Retrieval (ค้นหาด้วยคำถามที่เกลาแล้ว) ---
                vector_store = st.session_state.vector_store
                if vector_store:
                    context_text = get_relevant_context(vector_store, search_query)
                else:
                    context_text = "ไม่มีเอกสารให้อ้างอิง ตอบตามความรู้ทั่วไป"

                # --- STEP 3: Generation (ตอบคำถามด้วย Prompt ใหม่ที่ฉลาดขึ้น) ---
                system_prompt = f"""
คุณคือผู้ช่วย AI ผู้เชี่ยวชาญด้านการตรวจสอบ (PA Assistant)
หน้าที่: ตอบคำถามโดยใช้ข้อมูลจาก "บริบทที่ค้นพบ" ด้านล่างนี้เป็นหลัก

กฎการตอบ (สำคัญ):
1. อ้างอิงข้อมูลจากบริบทที่ให้มาเท่านั้น
2. **กรณีคำถามแบบ "ใช่หรือไม่" หรือถามถึงสิ่งที่ "ไม่มีในเอกสาร":**
   - ห้ามตอบว่า "ไม่ทราบ" หรือ "ข้อมูลไม่เพียงพอ" ทันที
   - ให้ตอบโดยระบุ **"สิ่งที่มีอยู่จริงในเอกสาร"** แทน เพื่อให้ผู้ใช้เปรียบเทียบเอง
   - ตัวอย่าง: ถ้าถามว่า "ต้องส่ง นาย A ไหม" แต่เอกสารบอกแค่ส่ง นาย B -> ให้ตอบว่า "จากเอกสารระบุให้ส่งถึง นาย B เท่านั้น ไม่ปรากฏข้อมูลเกี่ยวกับการส่งถึง นาย A"
3. ห้ามมั่วข้อมูลขึ้นมาเอง
4. **คำถามของผู้ใช้คือ:** "{prompt}" (ฉันค้นหาข้อมูลเรื่อง "{search_query}" มาให้คุณประกอบการตอบ)

--- บริบทที่ค้นพบ (Context) ---
{context_text}
-----------------------------
"""
                messages_for_api = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]

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
