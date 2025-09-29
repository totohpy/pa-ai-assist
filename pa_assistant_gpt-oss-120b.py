# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from groq import Groq # --- 1. เพิ่มการ Import ---
import os
import io
from PyPDF2 import PdfReader

# ตั้งค่าหน้าเพจ
st.set_page_config(page_title="Planning Studio (+ Findings Suggestions)", page_icon="🧭", layout="wide")

# ----------------- ⚙️ การตั้งค่ากลาง -----------------
with st.sidebar:
    st.title("⚙️ การตั้งค่ากลาง")
    # --- 2. แก้ไข Sidebar ---
    st.info("ใส่ API Key ของผู้ให้บริการ AI ที่ต้องการใช้งาน")
    st.session_state.api_key_global = st.text_input(
        "กรุณากรอก API Key",
        type="password",
        key="api_key_global_input_sidebar",
        help="สำหรับ Groq, ไปที่ https://console.groq.com/keys"
    )
    st.markdown("---")
    st.markdown("PA Planning Studio By PAO1 Audit Intelligence Nexus")

# ----------------- ฟังก์ชันต่างๆ (ไม่มีการเปลี่ยนแปลง) -----------------
MAX_CHARS_LIMIT = 200000

@st.cache_data(show_spinner=False)
def load_local_documents(folder_path="Doc"):
    text = ""
    if not os.path.isdir(folder_path): return text
    try:
        files_in_doc = os.listdir(folder_path)
        progress_bar = st.sidebar.progress(0, text=f"กำลังโหลดเอกสาร... (0/{len(files_in_doc)})")
        for i, filename in enumerate(files_in_doc):
            if len(text) >= MAX_CHARS_LIMIT:
                st.warning(f"ถึงขีดจำกัดข้อมูลแล้ว ({MAX_CHARS_LIMIT:,} ตัวอักษร)"); break
            file_path = os.path.join(folder_path, filename)
            try:
                if filename.endswith('.pdf'):
                    with open(file_path, 'rb') as f:
                        reader = PdfReader(f)
                        for page in reader.pages: text += page.extract_text() or ""
                elif filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: text += f.read()
                elif filename.endswith('.csv'):
                    df = pd.read_csv(file_path); text += df.head(15).to_string()
            except Exception as e: print(f"Could not read file {filename}: {e}")
            progress_bar.progress((i + 1) / len(files_in_doc), text=f"กำลังโหลดเอกสาร... ({i+1}/{len(files_in_doc)})")
        progress_bar.empty()
    except Exception as e: st.error(f"เกิดข้อผิดพลาดในการเข้าถึงคลังข้อมูล: {e}")
    return text[:MAX_CHARS_LIMIT]

def process_documents(files, source_type, limit, current_len=0):
    text = "";
    for file in files:
        if current_len + len(text) >= limit:
            st.warning(f"ถึงขีดจำกัดตัวอักษรสูงสุดแล้ว"); break
        try:
            if file.name.endswith('.pdf'):
                reader = PdfReader(file)
                for page in reader.pages: text += page.extract_text() or ""
            elif file.name.endswith('.txt'): text += file.getvalue().decode("utf-8")
            elif file.name.endswith('.csv'): df = pd.read_csv(file); text += df.head(15).to_string()
        except Exception as e: st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ {file.name}: {e}")
    return text[:limit - current_len], [f.name for f in files]

def init_state():
    ss = st.session_state
    ss.setdefault("plan", {"plan_id": "PLN-" + datetime.now().strftime("%y%m%d-%H%M%S"),"plan_title": "","program_name": "","who": "", "what": "", "where": "", "when": "", "why": "", "how": "", "how_much": "", "whom": "","objectives": "", "scope": "", "assumptions": "", "status": "Draft"})
    ss.setdefault("logic_items", pd.DataFrame(columns=["item_id","plan_id","type","description","metric","unit","target","source"]))
    ss.setdefault("methods", pd.DataFrame(columns=["method_id","plan_id","type","tool_ref","sampling","questions","linked_issue","data_source","frequency"]))
    ss.setdefault("kpis", pd.DataFrame(columns=["kpi_id","plan_id","level","name","formula","numerator","denominator","unit","baseline","target","frequency","data_source","quality_requirements"]))
    ss.setdefault("risks", pd.DataFrame(columns=["risk_id","plan_id","description","category","likelihood","impact","mitigation","hypothesis"]))
    ss.setdefault("audit_issues", pd.DataFrame(columns=["issue_id","plan_id","title","rationale","linked_kpi","proposed_methods","source_finding_id","issue_detail", "recommendation"]))
    ss.setdefault("gen_issues", ""); ss.setdefault("gen_findings", ""); ss.setdefault("gen_report", "")
    ss.setdefault("issue_results", pd.DataFrame()); ss.setdefault("ref_seed", ""); ss.setdefault("issue_query_text", "")
    ss.setdefault('api_key_global', '')
    ss.setdefault('chatbot_messages', [{"role": "assistant", "content": "สวัสดีครับ ผมคือ PA Chat ผู้ช่วยอัจฉริยะด้านการตรวจสอบ"}])
    ss.setdefault('doc_context_uploaded', ""); ss.setdefault('last_uploaded_files', set())
    if 'doc_context_local' not in ss:
        ss.doc_context_local = load_local_documents()
        if ss.doc_context_local and os.path.isdir('Doc'):
             ss.chatbot_messages.append({"role": "assistant", "content": f"ผมได้โหลดเอกสาร {len(os.listdir('Doc'))} ฉบับ เป็นฐานความรู้เรียบร้อยแล้วครับ"})

def next_id(prefix, df, col):
    if df.empty: return f"{prefix}-001"
    nums = [int(str(x).split("-")[-1]) for x in df[col] if str(x).split("-")[-1].isdigit()]
    n = max(nums) + 1 if nums else 1
    return f"{prefix}-{n:03d}"

def df_download_link(df: pd.DataFrame, filename: str, label: str):
    buf = BytesIO(); df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="text/csv")

@st.cache_data(show_spinner=False)
def load_findings(uploaded=None):
    findings_df = pd.DataFrame()
    # ... (เนื้อหาฟังก์ชันเหมือนเดิม) ...
    return findings_df

@st.cache_resource(show_spinner=False)
def build_tfidf_index(findings_df: pd.DataFrame):
    texts = (findings_df["issue_title"].fillna("") + " " + findings_df["issue_detail"].fillna("") + " " + findings_df["cause_detail"].fillna("") + " " + findings_df["recommendation"].fillna(""))
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2)); X = vec.fit_transform(texts)
    return vec, X

def search_candidates(query_text, findings_df, vec, X, top_k=8):
    # ... (เนื้อหาฟังก์ชันเหมือนเดิม) ...
    return out.sort_values("score", ascending=False).head(top_k)[[c for c in cols if c in out.columns]]
    
def create_excel_template():
    df = pd.DataFrame(columns=["finding_id", "issue_title", "unit", "program", "year", "cause_category", "cause_detail", "issue_detail", "recommendation", "outcomes_impact", "severity"])
    output = io.BytesIO();
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df.to_excel(writer, index=False, sheet_name='FindingsLibrary')
    return output.getvalue()

# ----------------- ส่วนแสดงผลหลัก (ไม่มีการเปลี่ยนแปลง) -----------------
init_state()
plan = st.session_state["plan"]
logic_df = st.session_state["logic_items"]
methods_df = st.session_state["methods"]
kpis_df = st.session_state["kpis"]
risks_df = st.session_state["risks"]
audit_issues_df = st.session_state["audit_issues"]

st.title("🧭 Planning Studio – Performance Audit")

with st.expander("💡 คำแนะนำการใช้งาน"):
    st.info("กรุณาระบุข้อมูล อย่างน้อย **ระบุ แผน & 6W2H** ส่วนใดส่วนหนึ่ง เพื่อค้นหาข้อตรวจพบที่ผ่านมาและให้ PA Assistant แนะนำ ได้แม่นยำที่สุด")

st.markdown("""<style> ... </style>""", unsafe_allow_html=True) # CSS เหมือนเดิม

tab_plan, tab_logic, tab_method, tab_kpi, tab_risk, tab_issue, tab_preview, tab_assist, tab_chatbot = st.tabs(["1. ระบุ แผน & 6W2H", "2. ระบุ Logic Model", ...]) # Tabs เหมือนเดิม

with tab_plan:
    # ... (เนื้อหาแท็บ 1-7 เหมือนเดิมทั้งหมด) ...
    pass
# ... (เนื้อหาแท็บ 2-7 เหมือนเดิมทั้งหมด) ...

# --- 3. แก้ไข tab_assist ---
with tab_assist:
    st.subheader("💡 PA Assistant (ขับเคลื่อนโดย Groq)")
    st.write("🤖 สร้างคำแนะนำประเด็นการตรวจสอบจาก AI")

    if st.button("🚀 สร้างคำแนะนำจาก AI", type="primary", key="llm_assist_button"):
        if not st.session_state.api_key_global:
            st.error("กรุณากรอก API Key ที่ Sidebar ด้านซ้ายก่อนใช้งาน")
        else:
            with st.spinner("กำลังสร้างคำแนะนำด้วย Groq..."):
                try:
                    issues_for_llm = st.session_state['audit_issues'][['title', 'rationale']]
                    plan_summary = f"""...""" # คงเดิม
                    user_prompt = f"""...""" # คงเดิม

                    # --- เปลี่ยนมาใช้ Groq client ---
                    client = Groq(
                        api_key=st.session_state.api_key_global,
                    )
                    
                    messages = [
                        {"role": "system", "content": "คุณคือผู้เชี่ยวชาญด้านการตรวจสอบผลสัมฤทธิ์และประสิทธิภาพการดำเนินงาน (Performance Audit)"},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    response = client.chat.completions.create(
                        # --- เปลี่ยนชื่อโมเดลเป็นของ Groq ---
                        model="llama3-70b-8192",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2048,
                    )

                    full_response = response.choices[0].message.content
                    # ... (ส่วน parsing เหมือนเดิม) ...
                    
                    st.success("สร้างคำแนะนำจาก AI (Groq) เรียบร้อยแล้ว ✅")

                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการเรียกใช้ Groq AI: {e}")
                    # ... (ส่วนจัดการ error เหมือนเดิม) ...
    
    st.subheader("ผลลัพธ์จาก AI")
    with st.expander("1. ...", expanded=True):
        st.write(st.session_state.get('gen_issues', "..."))
    with st.expander("2. ..."):
        st.write(st.session_state.get('gen_findings', "..."))
    with st.expander("3. ..."):
        st.write(st.session_state.get('gen_report', "..."))

# --- 4. แก้ไข tab_chatbot ---
with tab_chatbot:
    st.subheader("💬 PA Chat - ผู้ช่วยอัจฉริยะ (ขับเคลื่อนโดย Groq)")
    
    with st.expander("อัปโหลดเอกสารเพิ่มเติม (PDF, TXT, CSV)"):
        # ... (ส่วนนี้เหมือนเดิม) ...
        uploaded_files = st.file_uploader(...)
    
    # ... (โค้ดประมวลผลไฟล์อัปโหลดเหมือนเดิม) ...
    
    if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่...", key="chat_input_main"):
        st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                api_key = st.session_state.api_key_global
                if not api_key:
                    error_message = "เกิดข้อผิดพลาด: ไม่พบ API Key กรุณากรอก API Key ที่ Sidebar"
                    # ... (ส่วนจัดการ error เหมือนเดิม) ...
                else:
                    try:
                        doc_context = st.session_state.get('doc_context_local', '') + st.session_state.get('doc_context_uploaded', '')
                        system_prompt = f"""...""" # คงเดิม
                        
                        messages_for_api = [
                            {"role": "system", "content": system_prompt}
                        ] + st.session_state.chatbot_messages[-10:]

                        # --- เปลี่ยนมาใช้ Groq client ---
                        client = Groq(api_key=api_key)

                        response_stream = client.chat.completions.create(
                            # --- เปลี่ยนชื่อโมเดลเป็นของ Groq ---
                            model="llama3-70b-8192", 
                            messages=messages_for_api, 
                            temperature=0.5, 
                            max_tokens=3072, 
                            stream=True
                        )
                        response = message_placeholder.write_stream(response_stream)
                        st.session_state.chatbot_messages.append({"role": "assistant", "content": response})

                    except Exception as e:
                        # ... (ส่วนจัดการ error เหมือนเดิม) ...
                        pass
