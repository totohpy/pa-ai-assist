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
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]]) # ลด History เหลือ 4 เพื่อประหยัด Token
        
        system_prompt_rewrite = f"""
        คุณคือ AI ที่ทำหน้าที่ "เรียบเรียงประโยคคำถามใหม่" (Query Rewriter)
        หน้าที่ของคุณ:
        1. อ่านประวัติการสนทนาและคำถามล่าสุด
        2. ถ้าคำถามล่าสุดเชื่อมโยงกับบริบทก่อนหน้า ให้เขียนคำถามใหม่ให้สมบูรณ์ (เช่น "มันคืออะไร" -> "การตรวจสอบภายในคืออะไร")
        3. ถ้าไม่เกี่ยวข้องกัน ให้ใช้คำถามเดิม
        4. ตอบเฉพาะคำถามใหม่เท่านั้น ห้ามมีคำอธิบายอื่น

        --- Chat History ---
        {history_text}
        --------------------
        Current Question: {user_question}
        """
        
        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:free", 
            messages=[
                {"role": "system", "content": system_prompt_rewrite},
                {"role": "user", "content": "Rewritten Question:"}
            ],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return user_question 

def filter_relevant_content(full_text, query, max_chars=350000):
    """
    ฟังก์ชันกรองเนื้อหาแบบง่าย (Keyword Matching) เพื่อลดขนาดข้อความก่อนส่ง AI
    โดยไม่ต้องใช้ RAG Library
    """
    if not full_text or len(full_text) < max_chars:
        return full_text
        
    # 1. แบ่งข้อความเป็นย่อหน้า (Chunks)
    chunks = full_text.split('\n\n')
    if len(chunks) < 5: # ถ้าแบ่งย่อหน้าไม่ได้ ให้แบ่งตามบรรทัด
        chunks = full_text.split('\n')

    # 2. ให้คะแนนแต่ละย่อหน้าตามคำค้นหา (Query)
    query_words = set(query.replace("?", "").split())
    scored_chunks = []
    
    for chunk in chunks:
        # นับคำที่ตรงกัน
        score = sum(1 for word in query_words if word in chunk)
        # ให้คะแนนพิเศษถ้าย่อหน้านั้นยาวพอสมควร (มีเนื้อหา)
        if len(chunk) > 100: 
            score += 0.5 
        scored_chunks.append((score, chunk))
    
    # 3. เรียงลำดับตามคะแนน (มากไปน้อย)
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # 4. เลือกเฉพาะเนื้อหา Top จนกว่าจะเต็มโควต้า (max_chars)
    final_context = ""
    current_chars = 0
    
    for score, chunk in scored_chunks:
        if current_chars + len(chunk) < max_chars:
            final_context += chunk + "\n\n...[ตัดตอน]...\n\n"
            current_chars += len(chunk)
        else:
            break
            
    return final_context

def extract_text_from_files(files, folder_path="Doc"):
    text = ""
    # อ่านจาก Folder Local
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

    # อ่านจาก Uploaded Files
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
    ss.setdefault('chatbot_messages', [{"role": "assistant", "content": "สวัสดีครับ ผมคือ PA Assistant Chat พร้อมให้บริการครับ"}])
    ss.setdefault('file_context', "") 
    ss.setdefault('last_processed_files', set())

init_chat_state()

# ----------------- UI & Logic -----------------

with st.expander("อัปโหลดเอกสารเพิ่มเติม (PDF, TXT, CSV)"):
    uploaded_files = st.file_uploader("เลือกไฟล์...", type=['pdf', 'txt', 'csv'], accept_multiple_files=True)

# ตรวจสอบไฟล์และโหลด
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
                try:
                    api_key = st.secrets["openrouter_api_key"]
                except:
                    api_key = "" 
                
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )

                # --- STEP 1: Query Rewriting ---
                has_history = len(st.session_state.chatbot_messages) > 1
                if has_history:
                    with st.spinner("..."): 
                        search_query = rewrite_query(prompt, st.session_state.chatbot_messages[:-1], client)
                else:
                    search_query = prompt

                # --- STEP 2: Filter Context (แก้ปัญหา Token เกิน) ---
                # กรองเอาเฉพาะส่วนที่เกี่ยวกับคำถาม + จำกัดขนาดไม่ให้เกิน ~100k tokens
                raw_context = st.session_state.file_context
                
                # ถ้าไม่มีข้อมูล
                if not raw_context:
                    final_context = "ไม่มีเอกสารให้อ้างอิง ตอบตามความรู้ทั่วไป"
                else:
                    # เรียกใช้ฟังก์ชันกรองที่เขียนเพิ่ม
                    final_context = filter_relevant_content(raw_context, search_query, max_chars=300000) 

                # --- STEP 3: Generation ---
                system_prompt = f"""
คุณคือผู้ช่วย AI ผู้เชี่ยวชาญด้านการตรวจสอบ (PA Assistant)
หน้าที่: ตอบคำถามโดยใช้ข้อมูลจาก "เนื้อหาเอกสารแนบ" ที่คัดเลือกมาแล้วด้านล่างนี้

กฎการตอบ:
1. อ้างอิงข้อมูลจากเนื้อหาเอกสารที่ให้มาเท่านั้น
2. ถ้าข้อมูลถูกตัดทอน (มีคำว่า ...[ตัดตอน]...) ให้พยายามปะติดปะต่อเท่าที่ทำได้
3. หากไม่พบข้อมูลในส่วนที่คัดมา ให้แจ้งผู้ใช้ว่า "จากเอกสารที่เกี่ยวข้อง ไม่พบข้อมูลดังกล่าว"
4. **คำถามของผู้ใช้คือ:** "{prompt}" (บริบทค้นหา: "{search_query}")

--- เนื้อหาเอกสารแนบ (คัดเลือกมาบางส่วน) ---
{final_context}
-----------------------------
"""
                messages_for_api = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]

                # ใช้ Model
                primary_model = "google/gemini-2.0-pro-exp-02-05:free"
                backup_model = "meta-llama/llama-3.3-70b-instruct:free"

                try:
                    stream = client.chat.completions.create(
                        extra_headers={
                            "HTTP-Referer": "https://streamlit.io/",
                            "X-Title": "PA Assistant",
                        },
                        model=primary_model,
                        messages=messages_for_api,
                        stream=True
                    )
                except Exception:
                    stream = client.chat.completions.create(
                        extra_headers={
                            "HTTP-Referer": "https://streamlit.io/",
                            "X-Title": "PA Assistant",
                        },
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
