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

# ----------------- Helper Functions -----------------

def rewrite_query(user_question, chat_history, client):
    """แปลงคำถามสั้นๆ ให้เป็นคำถามที่สมบูรณ์"""
    try:
        # ใช้ History แค่ 2 อันล่าสุดพอ เพื่อประหยัด Token
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-2:]])
        
        system_prompt_rewrite = f"""
        คุณคือ AI Query Rewriter
        หน้าที่: อ่านประวัติการคุย แล้วเขียนคำถามล่าสุดใหม่ให้สมบูรณ์ชัดเจน (ภาษาไทย)
        บริบท: {history_text}
        คำถามปัจจุบัน: {user_question}
        คำถามใหม่:
        """
        
        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:free", 
            messages=[{"role": "user", "content": system_prompt_rewrite}],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return user_question 

def filter_relevant_content(full_text, query, max_chars=100000):
    """
    ฟังก์ชันกรองเนื้อหา (Manual RAG)
    ปรับปรุง: ลด max_chars ลงเหลือ 100,000 เพื่อความปลอดภัยของ Token Limit
    """
    if not full_text:
        return ""
        
    # ถ้าข้อความสั้นอยู่แล้ว ไม่ต้องทำอะไร
    if len(full_text) < max_chars:
        return full_text

    # 1. แบ่งเป็นชิ้นๆ (Chunks) ขนาดประมาณ 2000 ตัวอักษร
    chunk_size = 2000
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

    # 2. ให้คะแนนแต่ละชิ้น
    query_words = set(query.replace("?", "").split())
    scored_chunks = []
    
    for i, chunk in enumerate(chunks):
        score = 0
        for word in query_words:
            if word in chunk:
                score += chunk.count(word) # ยิ่งมีคำค้นหาเยอะ ยิ่งคะแนนสูง
        
        # เพิ่มคะแนนให้ส่วนหัวของเอกสาร (มักเป็นบทสรุป)
        if i < 3: score += 1 
            
        scored_chunks.append((score, chunk))
    
    # 3. เรียงลำดับคะแนนมากไปน้อย
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # 4. เลือกชิ้นเนื้อหาจนกว่าจะครบโควต้า
    final_context = ""
    current_chars = 0
    
    for score, chunk in scored_chunks:
        if current_chars + len(chunk) < max_chars:
            final_context += f"\n...[เนื้อหาที่เกี่ยวข้อง]...\n{chunk}"
            current_chars += len(chunk)
        else:
            break
            
    return final_context

def extract_text_from_files(files, folder_path="Doc"):
    text = ""
    # อ่านจาก Folder
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
                elif filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    text += df.to_string()
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    # อ่านจาก Upload
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
    ss.setdefault('chatbot_messages', [{"role": "assistant", "content": "สวัสดีครับ PA Assistant พร้อมให้บริการครับ"}])
    ss.setdefault('file_context', "") 
    ss.setdefault('last_processed_files', set())

init_chat_state()

# ----------------- UI & Logic -----------------

with st.expander("อัปโหลดเอกสารเพิ่มเติม (PDF, TXT, CSV)"):
    uploaded_files = st.file_uploader("เลือกไฟล์...", type=['pdf', 'txt', 'csv'], accept_multiple_files=True)

# Process Files
current_files_set = {f.name for f in uploaded_files} if uploaded_files else set()
is_files_changed = current_files_set != st.session_state.last_processed_files
is_first_load = not st.session_state.file_context and (uploaded_files or os.path.isdir("Doc"))

if is_files_changed or (is_first_load and not st.session_state.file_context):
    with st.spinner("กำลังประมวลผลเอกสาร..."):
        raw_text = extract_text_from_files(uploaded_files)
        if raw_text:
            st.session_state.file_context = raw_text
            st.session_state.last_processed_files = current_files_set
            st.success(f"✅ อ่านข้อมูลเรียบร้อย! ({len(raw_text):,} ตัวอักษร)")
        else:
            st.warning("ยังไม่มีข้อมูลเอกสาร")

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
                try: api_key = st.secrets["openrouter_api_key"]
                except: api_key = "" 
                
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

                # 1. Rewrite Query
                has_history = len(st.session_state.chatbot_messages) > 1
                search_query = rewrite_query(prompt, st.session_state.chatbot_messages, client) if has_history else prompt

                # 2. Filter Context (CRITICAL FIX: Max 100k chars)
                raw_context = st.session_state.file_context
                final_context = filter_relevant_content(raw_context, search_query, max_chars=100000) 
                
                if not final_context: final_context = "ไม่พบข้อมูลในเอกสาร ตอบตามความรู้ทั่วไป"

                # 3. Generate Answer
                system_prompt = f"""
คุณคือ PA Assistant ผู้เชี่ยวชาญการตรวจสอบ
ตอบคำถามโดยอ้างอิง "เนื้อหาที่คัดเลือกมา" ด้านล่างนี้:

กฎ:
1. ตอบเฉพาะที่มีในเนื้อหา
2. ถ้าเนื้อหาไม่พอ ให้บอกว่า "ไม่พบข้อมูลในเอกสารที่เกี่ยวข้อง"
3. คำถาม: "{prompt}" (บริบท: "{search_query}")

--- เนื้อหาที่คัดเลือกมา ---
{final_context}
-------------------------
"""
                # Safety Truncate: ถ้า System Prompt ยาวเกินไปจริงๆ ให้ตัดทิ้งอีกรอบ
                if len(system_prompt) > 120000:
                    system_prompt = system_prompt[:120000] + "\n... (ข้อมูลถูกตัดเนื่องจากยาวเกินไป)"

                messages_for_api = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]

                # Model Selection
                primary_model = "google/gemini-2.0-pro-exp-02-05:free"
                backup_model = "meta-llama/llama-3.3-70b-instruct:free"

                try:
                    stream = client.chat.completions.create(
                        extra_headers={"HTTP-Referer": "https://streamlit.io/", "X-Title": "PA Chat"},
                        model=primary_model,
                        messages=messages_for_api,
                        stream=True
                    )
                except Exception:
                    stream = client.chat.completions.create(
                        extra_headers={"HTTP-Referer": "https://streamlit.io/", "X-Title": "PA Chat"},
                        model=backup_model,
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
