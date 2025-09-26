# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
import io
# *** เพิ่ม Imports สำหรับ RAG Chatbot ***
from PyPDF2 import PdfReader
import glob
from io import StringIO

# กำหนดโฟลเดอร์เอกสารและข้อจำกัดสำหรับ RAG
DOC_FOLDER = "Doc" 
MAX_CHARS_LIMIT = 100000 # กำหนดการจำกัดที่ 100,000 ตัวอักษร

# ตั้งค่าหน้าเพจ
st.set_page_config(page_title="Planning Studio (+ PA Assistant)", page_icon="🧭", layout="wide")

# ----------------- Utility Functions สำหรับ RAG Chatbot -----------------

def read_pdf_text_from_path(file_path):
    """Extracts text content from a PDF file using a local file path."""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception:
        return ""

def read_pdf_text_from_uploaded(uploaded_file):
    """Extracts text content from an uploaded PDF file."""
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
            return text
    except Exception:
        return ""
    
def get_text_from_file(file, source_type):
    """Handles text extraction based on file source (local path or uploaded object)."""
    file_name = file if source_type == 'local' else file.name
    file_extension = os.path.splitext(file_name)[1].lower()
    current_text = ""
    
    try:
        if file_extension == '.pdf':
            if source_type == 'local':
                current_text = read_pdf_text_from_path(file)
            else:
                current_text = read_pdf_text_from_uploaded(file)
        elif file_extension == '.txt':
            if source_type == 'local':
                with open(file, 'r', encoding='utf-8') as f:
                    current_text = f.read()
            else:
                current_text = StringIO(file.getvalue().decode("utf-8")).read()
        elif file_extension == '.csv':
            if source_type == 'local':
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file, encoding='TIS-620')
            else:
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file, encoding='TIS-620')

            current_text = df.to_string()
    
    except Exception as e:
        return "", ""

    return current_text, file_name


def process_documents(files, source_type, max_chars_limit, existing_context_len):
    """
    Processes a list of files (either local paths or uploaded objects) and returns context.
    *** ฟังก์ชันนี้จะทำการตัดข้อความ (TRUNCATION) หากเกิน MAX_CHARS_LIMIT ***
    """
    context = ""
    total_chars = existing_context_len
    
    for file in files:
        current_text, file_name = get_text_from_file(file, source_type)
        
        if current_text:
            if total_chars + len(current_text) > max_chars_limit:
                remaining_chars = max_chars_limit - total_chars
                
                # ถ้ามีพื้นที่เหลืออยู่บ้าง (remaining_chars > 0) ให้ตัดมาแค่ส่วนที่เหลือ
                if remaining_chars > 0:
                    context += f"\n--- Start of Document: {file_name} (TRUNCATED) ---\n"
                    context += current_text[:remaining_chars] + f"\n... [ข้อความถูกตัดทอนจาก {file_name}]"
                    total_chars += remaining_chars
                
                # บรรทัด st.warning ถูกลบออกตามคำขอของผู้ใช้ เพื่อลดความรกตา
                # st.warning(f"จำกัดจำนวนข้อความที่นำเข้าที่ {max_chars_limit:,} ตัวอักษร ข้อความที่เกินจะถูกตัดออก")
                
                break # หยุดการประมวลผลไฟล์ถัดไปทันที
            
            else:
                # ถ้าไม่เกิน ให้เพิ่มเอกสารทั้งฉบับ
                context += f"\n--- Start of Document: {file_name} ---\n"
                context += current_text
                context += f"\n--- End of Document: {file_name} ---\n"
                total_chars += len(current_text)
                if source_type == 'local':
                    st.toast(f"โหลด: {file_name} ({len(current_text):,} ตัวอักษร)")
                else:
                    st.toast(f"อัปโหลด: {file_name} ({len(current_text):,} ตัวอักษร)")

    return context, total_chars - existing_context_len

def load_local_documents_on_init():
    """Initial load function for Doc folder (run once)."""
    supported_files = []
    for ext in ['*.pdf', '*.txt', '*.csv']:
        supported_files.extend(glob.glob(os.path.join(DOC_FOLDER, ext)))
        
    if not supported_files:
        return ""

    context, total_chars_added = process_documents(supported_files, 'local', MAX_CHARS_LIMIT, 0)
    return context

# ----------------- Session Init (รวม RAG State & Global API Key) -----------------
def init_state():
    ss = st.session_state
    ss.setdefault("plan", {
        "plan_id": "PLN-" + datetime.now().strftime("%y%m%d-%H%M%S"),
        "plan_title": "",
        "program_name": "",
        "who": "", "what": "", "where": "", "when": "", "why": "", "how": "", "how_much": "", "whom": "",
        "objectives": "", "scope": "", "assumptions": "", "status": "Draft"
    })
    ss.setdefault("logic_items", pd.DataFrame(columns=["item_id","plan_id","type","description","metric","unit","target","source"]))
    ss.setdefault("methods", pd.DataFrame(columns=["method_id","plan_id","type","tool_ref","sampling","questions","linked_issue","data_source","frequency"]))
    ss.setdefault("kpis", pd.DataFrame(columns=["kpi_id","plan_id","level","name","formula","numerator","denominator","unit","baseline","target","frequency","data_source","quality_requirements"]))
    ss.setdefault("risks", pd.DataFrame(columns=["risk_id","plan_id","description","category","likelihood","impact","mitigation","hypothesis"]))
    ss.setdefault("audit_issues", pd.DataFrame(columns=["issue_id","plan_id","title","rationale","linked_kpi","proposed_methods","source_finding_id","issue_detail", "recommendation"]))
    ss.setdefault("gen_issues", "")
    ss.setdefault("gen_findings", "")
    ss.setdefault("gen_report", "")
    ss.setdefault("issue_results", pd.DataFrame())
    ss.setdefault("ref_seed", "") 
    ss.setdefault("issue_query_text", "")
    
    # *** GLOBAL API Key for all LLM features ***
    # โหลดจาก st.secrets ก่อน ถ้าไม่มีให้เป็นค่าว่าง
    ss.setdefault("api_key_global", st.secrets.get('OPENAI_API_KEY', '')) 
    
    # *** RAG Chatbot specific states ***
    ss.setdefault("doc_context_local", "") 
    ss.setdefault("doc_context_uploaded", "") 
    ss.setdefault("chatbot_messages", [
        {"role": "assistant", "content": "สวัสดีครับ ผมคือผู้ช่วย AI อัจฉริยะ (PA Assistant) ผมพร้อมตอบคำถามโดยอ้างอิงจากเอกสารภายในที่เกี่ยวกับการตรวจสอบครับ"}
    ])
    ss.setdefault("local_load_attempted", False)
    # เพิ่ม state สำหรับบอกว่าแท็บไหนถูกเลือกอยู่
    ss.setdefault("current_tab", "1. ระบุ แผน & 6W2H")


def next_id(prefix, df, col):
    if df.empty: return f"{prefix}-001"
    nums = []
    for x in df[col]:
        try:
            nums.append(int(str(x).split("-")[-1]))
        except:
            pass
    n = max(nums) + 1 if nums else 1
    return f"{prefix}-{n:03d}"

def df_download_link(df: pd.DataFrame, filename: str, label: str):
    buf = BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="text/csv")

# ----------------- Findings Loader & Search -----------------
@st.cache_data(show_spinner=False)
def load_findings(uploaded=None):
    findings_df = pd.DataFrame()

    findings_db_path = "FindingsLibrary.csv"
    if os.path.exists(findings_db_path):
        try:
            findings_df = pd.read_csv(findings_db_path)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ FindingsLibrary.csv: {e}")
            findings_df = pd.DataFrame()

    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'):
                uploaded_df = pd.read_csv(uploaded)
            elif uploaded.name.endswith(('.xlsx', '.xls')):
                xls = pd.ExcelFile(uploaded)
                if "Data" in xls.sheet_names:
                    uploaded_df = pd.read_excel(xls, sheet_name="Data")
                    st.success("อ่านข้อมูลจากชีต 'Data' เรียบร้อยแล้ว")
                else:
                    st.warning("ไม่พบชีตชื่อ 'Data' ในไฟล์ที่อัปโหลด จะอ่านจากชีตแรกแทน")
                    uploaded_df = pd.read_excel(xls, sheet_name=0)

            if not uploaded_df.empty:
                findings_df = pd.concat([findings_df, uploaded_df], ignore_index=True)
                st.success(f"อัปโหลดไฟล์ '{uploaded.name}' และรวมกับฐานข้อมูลเดิมแล้ว")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ที่อัปโหลด: {e}")

    if not findings_df.empty:
        for c in ["issue_title","issue_detail","cause_detail","recommendation","program","unit"]:
            if c in findings_df.columns:
                findings_df[c] = findings_df[c].fillna("")
        if "year" in findings_df.columns:
            findings_df["year"] = pd.to_numeric(findings_df["year"], errors="coerce").fillna(0).astype(int)
        if "severity" in findings_df.columns:
            findings_df["severity"] = pd.to_numeric(findings_df["severity"], errors="coerce").fillna(3).clip(1,5).astype(int)

    return findings_df

@st.cache_resource(show_spinner=False)
def build_tfidf_index(findings_df: pd.DataFrame):
    texts = (findings_df["issue_title"].fillna("") + " " +
             findings_df["issue_detail"].fillna("") + " " +
             findings_df["cause_detail"].fillna("") + " " +
             findings_df["recommendation"].fillna(""))
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    return vec, X

def search_candidates(query_text, findings_df, vec, X, top_k=8):
    qv = vec.transform([query_text])
    sims = cosine_similarity(qv, X)[0]
    out = findings_df.copy()
    out["sim_score"] = sims
    if "year" in out.columns and out["year"].max() != out["year"].min():
        out["year_norm"] = (out["year"] - out["year"].min()) / (out["year"].max() - out["year"].min())
    else:
        out["year_norm"] = 0.0
    out["sev_norm"] = out.get("severity", 3) / 5
    out["score"] = out["sim_score"]*0.65 + out["sev_norm"]*0.25 + out["year_norm"]*0.10
    cols = [
        "finding_id","year","unit","program","issue_title","issue_detail",
        "cause_category","cause_detail","recommendation","outcomes_impact","severity","score"
    ]
    cols = [c for c in cols if c in out.columns] + ["sim_score"]
    return out.sort_values("score", ascending=False).head(top_k)[cols]
    
def create_excel_template():
    df = pd.DataFrame(columns=[
        "finding_id", "issue_title", "unit", "program", "year", 
        "cause_category", "cause_detail", "issue_detail", "recommendation", 
        "outcomes_impact", "severity"
    ])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='FindingsLibrary')
    processed_data = output.getvalue()
    return processed_data

# ----------------- App UI -----------------
init_state()

# *** Initial Load RAG Data ***
if not st.session_state.doc_context_local and not st.session_state.get('local_load_attempted', False):
    st.session_state.doc_context_local = load_local_documents_on_init()
    st.session_state.local_load_attempted = True
# *****************************

plan = st.session_state["plan"]
logic_df = st.session_state["logic_items"]
methods_df = st.session_state["methods"]
kpis_df = st.session_state["kpis"]
risks_df = st.session_state["risks"]
audit_issues_df = st.session_state["audit_issues"]

st.title("🧭 Planning Studio – Performance Audit")

# ----------------- START: Custom CSS (User's preferred multi-color tabs) -----------------
st.markdown("""
<style>
/* ... (Custom CSS ยังคงอยู่เหมือนเดิม) ... */
/* ---- Global Font ---- */
body { font-family: 'Kanit', sans-serif; }

/* ---- Base Style for Tabs (Button Look) ---- */
button[data-baseweb="tab"] {
    border: 1px solid #007bff;
    border-radius: 8px;
    padding: 10px 15px;
    margin: 5px 5px 5px 0px;
    transition: background-color 0.3s, color 0.3s;
    font-weight: bold;
    color: #007bff !important;
    background-color: #ffffff;
    box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
    /* Remove default Streamlit tab line/highlight */
    border-bottom: none !important; 
    &::after { content: none !important; }
}
button[data-baseweb="tab"][aria-selected="true"] {
    box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
}

/* ---- Group 1: 1-5 (Blue - Planning) ---- */
div[data-baseweb="tab-list"] button:nth-of-type(1),
div[data-baseweb="tab-list"] button:nth-of-type(2),
div[data-baseweb="tab-list"] button:nth-of-type(3),
div[data-baseweb="tab-list"] button:nth-of-type(4),
div[data-baseweb="tab-list"] button:nth-of-type(5) {
    border-color: #007bff;
    color: #007bff !important;
}
div[data-baseweb="tab-list"] button:nth-of-type(1)[aria-selected="true"],
div[data-baseweb="tab-list"] button:nth-of-type(2)[aria-selected="true"],
div[data-baseweb="tab-list"] button:nth-of-type(3)[aria-selected="true"],
div[data-baseweb="tab-list"] button:nth-of-type(4)[aria-selected="true"],
div[data-baseweb="tab-list"] button:nth-of-type(5)[aria-selected="true"] {
    background-color: #007bff;
    color: white !important;
}

/* ---- Group 2: 6-7 (Purple - Analysis/Review) ---- */
div[data-baseweb="tab-list"] button:nth-of-type(6),
div[data-baseweb="tab-list"] button:nth-of-type(7) {
    border-color: #6f42c1;
    color: #6f42c1 !important;
}
div[data-baseweb="tab-list"] button:nth-of-type(6)[aria-selected="true"],
div[data-baseweb="tab-list"] button:nth-of-type(7)[aria-selected="true"] {
    background-color: #6f42c1;
    color: white !important;
}

/* ---- Group 3: 8 (Gold - AI/Assist) ---- */
div[data-baseweb="tab-list"] button:nth-of-type(8) {
    border-color: #ffc107;
    color: #cc9900 !important;
    box-shadow: 0 0 5px rgba(255, 193, 7, 0.5);
}
div[data-baseweb="tab-list"] button:nth-of-type(8)[aria-selected="true"] {
    background-color: #ffc107;
    border-color: #ffc107;
    color: #333333 !important;
}

/* ---- Group 4: 9 (Green - Chatbot) ---- */
div[data-baseweb="tab-list"] button:nth-of-type(9) {
    border-color: #28a745;
    color: #28a745 !important;
    box-shadow: 0 0 5px rgba(40, 167, 69, 0.5);
}
div[data-baseweb="tab-list"] button:nth-of-type(9)[aria-selected="true"] {
    background-color: #28a745;
    border-color: #28a745;
    color: white !important;
}

/* ---- Container/Layout/Responsiveness ---- */
div[data-baseweb="tab-list"] { border-bottom: none !important; margin-bottom: 15px; flex-wrap: wrap; gap: 10px; }
@media (max-width: 768px) {
    .st-emotion-cache-18ni2cb, .st-emotion-cache-1jm69l4, [data-testid="stColumn"] {
        width: 100% !important;
        margin-bottom: 1rem;
    }
}

/* ---- Headers ---- */
h4 { color: #007bff !important; border-bottom: 2px solid #e0e0e0; padding-bottom: 5px; }
</style>
""", unsafe_allow_html=True)
# ----------------- END: Custom CSS -----------------

# ----------------- Tab Definitions (แก้ไขชื่อแท็บ) -----------------
# เราสร้างฟังก์ชัน on_click เพื่อบอกว่าตอนนี้เราอยู่แท็บไหน
def set_current_tab(tab_name):
    st.session_state.current_tab = tab_name

tab_plan, tab_logic, tab_method, tab_kpi, tab_risk, tab_issue, tab_preview, tab_assist, tab_chatbot = st.tabs([
    "1. ระบุ แผน & 6W2H", 
    "2. ระบุ Logic Model", 
    "3. ระบุ Methods", 
    "4. ระบุ KPIs", 
    "5. ระบุ Risks", 
    "6. ค้นหาข้อตรวจพบที่ผ่านมา", 
    "7. สรุปข้อมูล (Preview)", 
    "💡 Assistant แนะนำประเด็นการตรวจสอบ", # เปลี่ยนจาก "💡 PA Audit Assistant (AI/ LLM)"
    "💬 PA Chat (ถาม-ตอบเ)" # เปลี่ยนจาก "💬 PA Chat Assistant (ถาม-ตอบเอกสาร)"
]) 

# ----------------- Tab 1: ระบุ แผน & 6W2H -----------------
with tab_plan:
    set_current_tab("1. ระบุ แผน & 6W2H")
    # ... (เนื้อหาของ Tab 1 ยังคงอยู่เหมือนเดิม)
    st.subheader("ข้อมูลแผน (Plan) - กรุณาระบุข้อมูล")
    with st.container(border=True):
        c1, c2, c3 = st.columns([2,2,1])
        with c1:
            plan["plan_title"] = st.text_input("ชื่อแผน/เรื่องที่จะตรวจ", plan["plan_title"])
            plan["program_name"] = st.text_input("ชื่อโครงการ/แผนงาน", plan["program_name"])
            plan["objectives"] = st.text_area("วัตถุประสงค์การตรวจ", plan["objectives"])
        with c2:
            plan["scope"] = st.text_area("ขอบเขตการตรวจ", plan["scope"])
            plan["assumptions"] = st.text_area("สมมุติฐาน/ข้อจำกัดข้อมูล", plan["assumptions"])
        with c3:
            st.text_input("Plan ID", plan["plan_id"], disabled=True)
            plan["status"] = st.selectbox("สถานะ", ["Draft","Published"], index=0)

    st.divider()
    st.subheader("สรุปเรื่องที่ตรวจสอบ (6W2H)")

    # *** START: ลดความรกตา - ซ่อนส่วน AI Assist ด้วย expander ***
    with st.expander("🚀 สร้าง 6W2H อัตโนมัติด้วย AI (คลิกเพื่อเปิด)"): 
        with st.container(border=True):
            st.write("คัดลอกข้อความจากไฟล์ของคุณแล้วนำมาวางในช่องด้านล่างนี้")
            uploaded_text = st.text_area("ระบุข้อความเกี่ยวกับเรื่องที่จะตรวจสอบ ที่ต้องการให้ AI ช่วยสรุป 6W2H", height=200, key="uploaded_text")
            
            # *** ใช้ API Key Global และกำหนด Callback เพื่อบันทึก Key ***
            def save_api_key_global():
                st.session_state.api_key_global = st.session_state.api_key_input_6w2h

            st.markdown("💡 **ยังไม่มี API Key?** คลิก [ที่นี่](https://playground.opentyphoon.ai/settings/api-key) เพื่อรับ key ฟรี!")
            api_key_6w2h = st.text_input(
                "กรุณากรอก API Key เพื่อใช้บริการ AI:", 
                type="password", 
                value=st.session_state.api_key_global,
                key="api_key_input_6w2h",
                on_change=save_api_key_global
            )
            # ตรวจสอบว่า Key ถูกบันทึกแล้วหรือไม่
            api_key_6w2h = st.session_state.api_key_global
            
            if st.button("🚀 สร้าง 6W2H จากข้อความ", type="primary", key="6w2h_button"):
                if not uploaded_text:
                    st.error("กรุณาวางข้อความในช่องก่อน")
                elif not api_key_6w2h:
                    st.error("กรุณากรอก API Key ก่อนใช้งาน")
                else:
                    with st.spinner("กำลังประมวลผล..."):
                        try:
                            user_prompt = f"""
จากข้อความด้านล่างนี้ กรุณาสรุปและแยกแยะข้อมูลให้เป็น 6W2H ได้แก่ Who, Whom, What, Where, When, Why, How, และ How much โดยให้อยู่ในรูปแบบ key-value ที่ชัดเจน
ข้อความ:
---
{uploaded_text}
---
รูปแบบที่ต้องการ:
Who: [ข้อความ]
Whom: [ข้อความ]
What: [ข้อความ]
Where: [ข้อความ]
When: [ข้อความ]
Why: [ข้อความ]
How: [ข้อความ]
How Much: [ข้อความ]
"""
                            client = OpenAI(
                                api_key=api_key_6w2h,
                                base_url="https://api.opentyphoon.ai/v1"
                            )
                            response = client.chat.completions.create(
                                model="typhoon-v2.1-12b-instruct",
                                messages=[{"role": "user", "content": user_prompt}],
                                temperature=0.7,
                                max_tokens=1024,
                                top_p=0.9,
                            )
                            llm_output = response.choices[0].message.content
                            
                            with st.expander("แสดงผลลัพธ์จาก AI"):
                                st.write(llm_output)

                            lines = llm_output.strip().split('\n')
                            for line in lines:
                                if ':' in line:
                                    key, value = line.split(':', 1)
                                    normalized_key = key.strip().lower().replace(' ', '_')
                                    value = value.strip()
                                    if normalized_key == 'how_much': st.session_state.plan['how_much'] = value
                                    elif normalized_key == 'whom': st.session_state.plan['whom'] = value
                                    elif normalized_key == 'who': st.session_state.plan['who'] = value
                                    elif normalized_key == 'what': st.session_state.plan['what'] = value
                                    elif normalized_key == 'where': st.session_state.plan['where'] = value
                                    elif normalized_key == 'when': st.session_state.plan['when'] = value
                                    elif normalized_key == 'why': st.session_state.plan['why'] = value
                                    elif normalized_key == 'how': st.session_state.plan['how'] = value

                            st.success("สร้าง 6W2H เรียบร้อยแล้ว! กรุณาตรวจสอบข้อมูลแล้วคัดลอกไปวางตามรายละเอียดด้านล่าง")
                            st.balloons()
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาดในการเรียกใช้ AI: {e}")
    # *** END: ลดความรกตา - ซ่อนส่วน AI Assist ด้วย expander ***
        
    st.markdown("##### ⭐กรุณาระบุข้อมูล เพื่อนำไปใช้ประมวลผล")
    with st.container(border=True):
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.session_state.plan["who"] = st.text_input("Who (ใคร)", value=st.session_state.plan["who"], key="who_input")
            st.session_state.plan["whom"] = st.text_input("Whom (เพื่อใคร)", value=st.session_state.plan["whom"], key="whom_input")
            st.session_state.plan["what"] = st.text_input("What (ทำอะไร)", value=st.session_state.plan["what"], key="what_input")
            st.session_state.plan["where"] = st.text_input("Where (ที่ไหน)", value=st.session_state.plan["where"], key="where_input")
        with cc2:
            st.session_state.plan["when"] = st.text_input("When (เมื่อใด)", value=st.session_state.plan["when"], key="when_input")
            st.session_state.plan["why"] = st.text_area("Why (ทำไม)", value=st.session_state.plan["why"], key="why_input")
        with cc3:
            st.session_state.plan["how"] = st.text_area("How (อย่างไร)", value=st.session_state.plan["how"], key="how_input")
            st.session_state.plan["how_much"] = st.text_input("How much (เท่าไร)", value=st.session_state.plan["how_much"], key="how_much_input")

# ----------------- Tab 2: Logic Model -----------------
with tab_logic:
    set_current_tab("2. ระบุ Logic Model")
    # ... (เนื้อหาของ Tab 2 ยังคงอยู่เหมือนเดิม)
    st.subheader("ระบุข้อมูล Logic Model: Input → Activities → Output → Outcome → Impact")
    st.dataframe(logic_df, use_container_width=True, hide_index=True)
    with st.expander("➕ เพิ่มรายการใน Logic Model"):
        with st.container(border=True):
            colA, colB, colC = st.columns(3)
            with colA:
                typ = st.selectbox("ประเภท", ["Input","Activity","Output","Outcome","Impact"])
                desc = st.text_input("คำอธิบาย/รายละเอียด")
                metric = st.text_input("ตัวชี้วัด/metric (เช่น จำนวน, สัดส่วน)")
            with colB:
                unit = st.text_input("หน่วย", value="หน่วย", key="logic_unit")
                target = st.text_input("เป้าหมาย", value="", key="logic_target")
            with colC:
                source = st.text_input("แหล่งข้อมูล", value="", key="logic_source")
                if st.button("เพิ่ม Logic Item", type="primary"):
                    new_row = pd.DataFrame([{
                        "item_id": next_id("LG", logic_df, "item_id"),
                        "plan_id": plan["plan_id"],
                        "type": typ, "description": desc, "metric": metric,
                        "unit": unit, "target": target, "source": source
                    }])
                    st.session_state["logic_items"] = pd.concat([logic_df, new_row], ignore_index=True)
                    st.rerun()

# ----------------- Tab 3: Methods -----------------
with tab_method:
    set_current_tab("3. ระบุ Methods")
    # ... (เนื้อหาของ Tab 3 ยังคงอยู่เหมือนเดิม)
    st.subheader("ระบุวิธีการเก็บข้อมูล (Methods)")
    st.dataframe(methods_df, use_container_width=True, hide_index=True)
    with st.expander("➕ เพิ่ม Method"):
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                mtype = st.selectbox("ชนิด", ["observe","interview","questionnaire","document"])
                tool_ref = st.text_input("รหัส/อ้างอิงเครื่องมือ", value="")
                sampling = st.text_input("วิธีคัดเลือกตัวอย่าง", value="")
            with c2:
                questions = st.text_area("คำถาม/ประเด็นหลัก")
                linked_issue = st.text_input("โยงประเด็นตรวจ", value="")
            with c3:
                data_source = st.text_input("แหล่งข้อมูล", value="", key="method_data_source")
                frequency = st.text_input("ความถี่", value="ครั้งเดียว", key="method_frequency")
                if st.button("เพิ่ม Method", type="primary", key="add_method_btn"):
                    new_row = pd.DataFrame([{
                        "method_id": next_id("MT", methods_df, "method_id"),
                        "plan_id": plan["plan_id"],
                        "type": mtype, "tool_ref": tool_ref, "sampling": sampling,
                        "questions": questions, "linked_issue": linked_issue,
                        "data_source": data_source, "frequency": frequency
                    }])
                    st.session_state["methods"] = pd.concat([methods_df, new_row], ignore_index=True)
                    st.rerun()

# ----------------- Tab 4: KPIs -----------------
with tab_kpi:
    set_current_tab("4. ระบุ KPIs")
    # ... (เนื้อหาของ Tab 4 ยังคงอยู่เหมือนเดิม)
    st.subheader("ระบุตัวชี้วัด (KPIs)")
    st.dataframe(kpis_df, use_container_width=True, hide_index=True)
    with st.expander("➕ เพิ่ม KPI เอง"):
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                level = st.selectbox("ระดับ", ["output","outcome"])
                name = st.text_input("ชื่อ KPI")
                formula = st.text_input("สูตร/นิยาม")
            with col2:
                numerator = st.text_input("ตัวตั้ง (numerator)")
                denominator = st.text_input("ตัวหาร (denominator)")
                unit = st.text_input("หน่วย", value="%", key="kpi_unit")
            with col3:
                baseline = st.text_input("Baseline", value="")
                target = st.text_input("Target", value="")
                freq = st.text_input("ความถี่", value="รายไตรมาส")
                data_src = st.text_input("แหล่งข้อมูล", value="", key="kpi_data_source")
                quality = st.text_input("ข้อกำหนดคุณภาพข้อมูล", value="ถูกต้อง/ทันเวลา", key="kpi_quality")
                if st.button("เพิ่ม KPI", type="primary", key="add_kpi_btn"):
                    new_row = pd.DataFrame([{
                        "kpi_id": next_id("KPI", kpis_df, "kpi_id"),
                        "plan_id": plan["plan_id"],
                        "level": level, "name": name, "formula": formula,
                        "numerator": numerator, "denominator": denominator, "unit": unit,
                        "baseline": baseline, "target": target, "frequency": freq,
                        "data_source": data_src, "quality_requirements": quality
                    }])
                    st.session_state["kpis"] = pd.concat([kpis_df, new_row], ignore_index=True)
                    st.rerun()

# ----------------- Tab 5: Risks -----------------
with tab_risk:
    set_current_tab("5. ระบุ Risks")
    # ... (เนื้อหาของ Tab 5 ยังคงอยู่เหมือนเดิม)
    st.subheader("ระบุความเสี่ยง (Risks)")
    st.dataframe(risks_df, use_container_width=True, hide_index=True)
    with st.expander("➕ เพิ่ม Risk"):
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                desc = st.text_area("คำอธิบายความเสี่ยง")
                category = st.selectbox("หมวด", ["policy","org","data","process","people"])
            with c2:
                likelihood = st.select_slider("โอกาสเกิด (1-5)", options=[1,2,3,4,5], value=3)
                impact = st.select_slider("ผลกระทบ (1-5)", options=[1,2,3,4,5], value=3)
            with c3:
                mitigation = st.text_area("มาตรการลดความเสี่ยง")
                hypothesis = st.text_input("สมมุติฐานที่ต้องทดสอบ")
                if st.button("เพิ่ม Risk", type="primary", key="add_risk_btn"):
                    new_row = pd.DataFrame([{
                        "risk_id": next_id("RSK", risks_df, "risk_id"),
                        "plan_id": plan["plan_id"],
                        "description": desc, "category": category,
                        "likelihood": likelihood, "impact": impact,
                        "mitigation": mitigation, "hypothesis": hypothesis
                    }])
                    st.session_state["risks"] = pd.concat([risks_df, new_row], ignore_index=True)
                    st.rerun()

# ----------------- Tab 6: ค้นหาข้อตรวจพบที่ผ่านมา -----------------
with tab_issue:
    set_current_tab("6. ค้นหาข้อตรวจพบที่ผ่านมา")
    # ... (เนื้อหาของ Tab 6 ยังคงอยู่เหมือนเดิม)
    st.subheader("🔎 แนะนำประเด็นตรวจจากรายงานเก่า (Issue Suggestions)")
    st.write("***กรุณาอัพโหลดฐานข้อมูล (ถ้าไม่มีจะใช้ฐานข้อมูลในระบบ)***")

    # *** START: ลดความรกตา - ซ่อนส่วนอัปโหลดไฟล์ด้วย expander ***
    with st.expander("⬆️ อัปโหลด/จัดการฐานข้อมูล Findings (คลิกเพื่อเปิด)"):
        with st.container(border=True):
            st.download_button(
                label="⬇️ ดาวน์โหลดไฟล์แม่แบบ FindingsLibrary.xlsx",
                data=create_excel_template(),
                file_name="FindingsLibrary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            uploaded = st.file_uploader("อัปโหลด FindingsLibrary.csv หรือ .xlsx", type=["csv", "xlsx", "xls"])
    # *** END: ลดความรกตา - ซ่อนส่วนอัปโหลดไฟล์ด้วย expander ***
    
    findings_df = load_findings(uploaded=uploaded)
    
    if findings_df.empty:
        st.info("ไม่พบข้อมูล Findings ที่จะนำมาใช้ โปรดอัปโหลดไฟล์ หรือตรวจสอบว่ามีไฟล์ FindingsLibrary.csv อยู่ในโฟลเดอร์เดียวกัน")
    else:
        st.success(f"พบข้อมูล Findings ทั้งหมด {len(findings_df)} รายการ")
        vec, X = build_tfidf_index(findings_df)
        
        seed = f"""
Who:{plan.get('who','')} What:{plan.get('what','')} Where:{plan.get('where','')}
When:{plan.get('when','')} Why:{plan.get('why','')} How:{plan.get('how','')}
Outputs:{' | '.join(logic_df[logic_df['type']=='Output']['description'].tolist())}
Outcomes:{' | '.join(logic_df[logic_df['type']=='Outcome']['description'].tolist())}
"""
        
        def refresh_query_text(new_seed):
            st.session_state["issue_query_text"] = new_seed
            st.session_state["ref_seed"] = new_seed 

        if "issue_query_text" not in st.session_state or st.session_state["issue_query_text"] == "":
            st.session_state["issue_query_text"] = seed
            st.session_state["ref_seed"] = seed
            
        elif st.session_state.get("ref_seed") != seed and st.session_state.get("issue_query_text") == st.session_state.get("ref_seed"):
            st.session_state["issue_query_text"] = seed
            st.session_state["ref_seed"] = seed 

        c_query_area, c_refresh_btn = st.columns([6, 1])

        with c_query_area:
            query_text = st.text_area(
                "**สรุปบริบทที่ใช้ค้นหา (แก้ไขได้):**", 
                st.session_state["issue_query_text"], 
                height=140, 
                key="issue_query_text"
            )
        
        with c_refresh_btn:
            st.markdown("<br>", unsafe_allow_html=True) 
            st.button(
                "🔄 ดึงข้อมูลจากหน้าก่อนหน้า", 
                on_click=refresh_query_text,
                args=(seed,), 
                help="คลิกเพื่ออัปเดตช่องค้นหาด้วยข้อมูลล่าสุดจากแท็บ 'ระบุ แผน & 6W2H' และ 'ระบุ Logic Model' (จะล้างข้อมูลที่คุณเคยแก้ไข)",
                type="secondary"
            )
        
        if st.button("ค้นหาประเด็นที่ใกล้เคียง", type="primary", key="search_button_fix"):
            search_value = st.session_state.get("issue_query_text", seed)
            results = search_candidates(search_value, findings_df, vec, X, top_k=8)
            st.session_state["issue_results"] = results
            st.success(f"พบประเด็นที่เกี่ยวข้อง {len(results)} รายการ")
            
        results = st.session_state.get("issue_results", pd.DataFrame())
        
        if not results.empty:
            st.divider()
            st.subheader("ผลลัพธ์การค้นหา")
            for i, row in results.reset_index(drop=True).iterrows():
                with st.container(border=True):
                    title_txt = row.get("issue_title", "(ไม่มีชื่อประเด็น)")
                    unit_txt = row.get("unit", "-")
                    prog_txt = row.get("program", "-")
                    year_txt = int(row["year"]) if "year" in row and str(row["year"]).isdigit() else row.get("year", "-")
                    st.markdown(f"**{title_txt}** \nหน่วย: {unit_txt} • โครงการ: {prog_txt} • ปี {year_txt}")
                    cause_cat = row.get("cause_category", "-")
                    cause_detail = row.get("cause_detail", "-")
                    st.caption(f"สาเหตุ: *{cause_cat}* — {cause_detail}")
    
                    with st.expander("รายละเอียด/ข้อเสนอแนะ (เดิม)"):
                        st.write(row.get("issue_detail", "-"))
                        st.caption("ข้อเสนอแนะเดิม: " + (row.get("recommendation", "") or "-"))
                        impact = row["outcomes_impact"] if "outcomes_impact" in row else "-"
                        sim = row["sim_score"] if "sim_score" in row else 0
                        score = row["score"] if "score" in row else 0
                        
                        st.markdown(f"**ผลกระทบที่อาจเกิดขึ้น:** {impact}  •  <span style='color:red;'>**คะแนนความเกี่ยวข้อง**</span>: {score:.3f} (<span style='color:blue;'>**Similarity Score**</span>={sim:.3f})", unsafe_allow_html=True)

                    if st.button(f"➕ เพิ่มเป็นประเด็นตรวจ (อ้างอิง ID: {row['finding_id']})", key=f"add_issue_from_finding_{i}"):
                        new_row = pd.DataFrame([{
                            "issue_id": next_id("ISS", audit_issues_df, "issue_id"),
                            "plan_id": plan["plan_id"],
                            "title": title_txt,
                            "rationale": f"อ้างอิงจากข้อตรวจพบเก่า: {row['finding_id']} (Score: {score:.3f})",
                            "linked_kpi": "",
                            "proposed_methods": "",
                            "source_finding_id": row['finding_id'],
                            "issue_detail": row.get("issue_detail", "") or "-",
                            "recommendation": row.get("recommendation", "") or "-"
                        }])
                        st.session_state["audit_issues"] = pd.concat([audit_issues_df, new_row], ignore_index=True)
                        st.rerun()

    st.divider()
    st.subheader("จัดการประเด็นการตรวจสอบ (Audit Issues)")
    # ... (เนื้อหาจัดการ Audit Issues ยังคงอยู่เหมือนเดิม)

# ----------------- Tab 7: สรุปข้อมูล (Preview) -----------------
with tab_preview:
    set_current_tab("7. สรุปข้อมูล (Preview)")
    # ... (เนื้อหาของ Tab 7 ยังคงอยู่เหมือนเดิม)
    st.subheader("ภาพรวมแผนการตรวจสอบ")
    
    st.markdown("#### ข้อมูลพื้นฐาน")
    c1, c2 = st.columns(2)
    c1.markdown(f"**Plan ID:** {plan['plan_id']}")
    c1.markdown(f"**ชื่อแผน:** {plan['plan_title']}")
    c1.markdown(f"**โครงการ/แผนงาน:** {plan['program_name']}")
    c1.markdown(f"**สถานะ:** {plan['status']}")
    c2.markdown(f"**วัตถุประสงค์:** {plan['objectives']}")
    c2.markdown(f"**ขอบเขต:** {plan['scope']}")
    c2.markdown(f"**สมมุติฐาน/ข้อจำกัด:** {plan['assumptions']}")

    st.markdown("#### 6W2H Summary")
    c_w1, c_w2, c_w3 = st.columns(3)
    c_w1.markdown(f"**Who (ใคร):** {plan['who']}")
    c_w1.markdown(f"**Whom (เพื่อใคร):** {plan['whom']}")
    c_w1.markdown(f"**What (ทำอะไร):** {plan['what']}")
    c_w1.markdown(f"**Where (ที่ไหน):** {plan['where']}")
    c_w2.markdown(f"**When (เมื่อใด):** {plan['when']}")
    c_w2.markdown(f"**Why (ทำไม):** {plan['why']}")
    c_w3.markdown(f"**How (อย่างไร):** {plan['how']}")
    c_w3.markdown(f"**How much (เท่าไร):** {plan['how_much']}")

    st.markdown("#### Logic Model")
    st.dataframe(logic_df, use_container_width=True, hide_index=True)

    st.markdown("#### Methods")
    st.dataframe(methods_df, use_container_width=True, hide_index=True)
    
    st.markdown("#### KPIs")
    st.dataframe(kpis_df, use_container_width=True, hide_index=True)
    
    st.markdown("#### Risks")
    st.dataframe(risks_df, use_container_width=True, hide_index=True)
    
    st.markdown("#### Audit Issues")
    st.dataframe(audit_issues_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("ดาวน์โหลดข้อมูลทั้งหมด")
    
    col_dl1, col_dl2, col_dl3, col_dl4, col_dl5 = st.columns(5)
    df_download_link(logic_df, f"Logic_Model_{plan['plan_id']}.csv", "⬇️ Logic Model")
    df_download_link(methods_df, f"Methods_{plan['plan_id']}.csv", "⬇️ Methods")
    df_download_link(kpis_df, f"KPIs_{plan['plan_id']}.csv", "⬇️ KPIs")
    df_download_link(risks_df, f"Risks_{plan['plan_id']}.csv", "⬇️ Risks")
    df_download_link(audit_issues_df, f"Audit_Issues_{plan['plan_id']}.csv", "⬇️ Audit Issues")

# ----------------- Tab 8: 💡 Assistant แนะนำประเด็นการตรวจสอบ -----------------
with tab_assist:
    set_current_tab("💡 Assistant แนะนำประเด็นการตรวจสอบ")
    # ... (เนื้อหาของ Tab 8 ยังคงอยู่เหมือนเดิม)
    st.subheader("💡 AI Assistant - สรุปและวิเคราะห์เบื้องต้น")

    # *** ใช้ API Key Global และกำหนด Callback เพื่อบันทึก Key ***
    def save_api_key_global_assist():
        st.session_state.api_key_global = st.session_state.api_key_input_assist
        st.toast("บันทึก API Key แล้ว!")
    
    st.markdown("💡 **ยังไม่มี API Key?** คลิก [ที่นี่](https://playground.opentyphoon.ai/settings/api-key) เพื่อรับ key ฟรี!")
    api_key_assist = st.text_input(
        "กรุณากรอก API Key เพื่อใช้บริการ AI:", 
        type="password", 
        value=st.session_state.api_key_global,
        key="api_key_input_assist",
        on_change=save_api_key_global_assist
    )
    api_key_assist = st.session_state.api_key_global

    # ... (ส่วนการเตรียมข้อมูลสำหรับ LLM)
    context_data = f"""
[ข้อมูลพื้นฐาน]
Plan ID: {plan['plan_id']}
ชื่อแผน/เรื่องที่จะตรวจ: {plan['plan_title']}
โครงการ/แผนงาน: {plan['program_name']}
วัตถุประสงค์การตรวจ: {plan['objectives']}
ขอบเขตการตรวจ: {plan['scope']}
สมมุติฐาน/ข้อจำกัด: {plan['assumptions']}
6W2H: Who={plan['who']}, Whom={plan['whom']}, What={plan['what']}, Where={plan['where']}, When={plan['when']}, Why={plan['why']}, How={plan['how']}, How much={plan['how_much']}

[Logic Model]
{logic_df.to_string(index=False)}

[KPIs]
{kpis_df.to_string(index=False)}

[Risks]
{risks_df.to_string(index=False)}

[Audit Issues (ที่เพิ่มแล้ว)]
{audit_issues_df.to_string(index=False)}
"""

    if st.button("ประมวลผลและสร้างคำแนะนำจาก AI", type="primary"):
        if not api_key_assist:
            st.error("กรุณากรอก API Key ก่อนใช้งาน")
            st.session_state["gen_issues"] = ""
            st.session_state["gen_findings"] = ""
            st.session_state["gen_report"] = ""
        elif not plan["plan_title"]:
            st.error("กรุณากรอก 'ชื่อแผน/เรื่องที่จะตรวจ' ก่อน")
        else:
            with st.spinner("กำลังวิเคราะห์ข้อมูลและสร้างคำแนะนำ..."):
                try:
                    client = OpenAI(
                        api_key=api_key_assist,
                        base_url="https://api.opentyphoon.ai/v1"
                    )

                    # 1. Generate Issue Suggestions
                    prompt_issue = f"""
คุณเป็นผู้เชี่ยวชาญด้านการตรวจสอบผลการดำเนินงาน (Performance Audit) หน้าที่ของคุณคือวิเคราะห์ข้อมูลแผนการตรวจสอบด้านล่างและระบุประเด็นการตรวจสอบที่สำคัญและมีความเสี่ยงสูง (High-Risk Audit Issues) ที่ควรตรวจสอบเพิ่มเติม โดยอ้างอิงจาก Logic Model, KPIs และ Risks ที่ระบุไว้

เงื่อนไข:
1. ให้เน้นประเด็นที่มีความเสี่ยงสูง (จาก Risks) หรือประเด็นที่โยงกับ Output/Outcome/Impact ที่สำคัญ (จาก Logic Model และ KPIs)
2. **ห้าม** สร้างประเด็นที่ซ้ำซ้อนกับ [Audit Issues (ที่เพิ่มแล้ว)] ที่มีอยู่แล้ว
3. สรุปประเด็นการตรวจสอบเพิ่มเติมแต่ละรายการให้ชัดเจนในรูปของคำถามการตรวจสอบ (Audit Question) และระบุเหตุผลที่มา (Rationale) สั้นๆ ว่าโยงกับส่วนใดในแผน

ข้อมูลแผน:
---
{context_data}
---

รูปแบบผลลัพธ์:
1. Audit Question: [คำถามการตรวจสอบ] (Rationale: [เหตุผลสั้นๆ])
2. Audit Question: [คำถามการตรวจสอบ] (Rationale: [เหตุผลสั้นๆ])
... (ให้มาอย่างน้อย 3-5 ประเด็น)
"""
                    response_issue = client.chat.completions.create(
                        model="typhoon-v2.1-12b-instruct",
                        messages=[{"role": "user", "content": prompt_issue}],
                        temperature=0.7, max_tokens=1024
                    )
                    st.session_state["gen_issues"] = response_issue.choices[0].message.content

                    # 2. Generate Findings Hypothesis
                    prompt_finding = f"""
คุณเป็นผู้เชี่ยวชาญด้านการตรวจสอบผลการดำเนินงาน (Performance Audit) หน้าที่ของคุณคือวิเคราะห์ข้อมูลแผนการตรวจสอบด้านล่างและประเด็นการตรวจสอบที่ AI ได้แนะนำมา **(โดยไม่ต้องย้อนไปดู Findings Library เก่า)** และสร้างสมมุติฐาน 'ข้อตรวจพบที่คาดว่าจะพบ' (Hypothesized Findings) ที่เป็นไปได้ พร้อมระบุระดับโอกาสที่จะพบ (Likelihood: ต่ำ/ปานกลาง/สูง)

ข้อมูลแผน:
---
{context_data}
---

ประเด็นการตรวจสอบที่แนะนำ:
---
{st.session_state["gen_issues"]}
---

รูปแบบผลลัพธ์:
- ข้อตรวจพบที่คาดว่าจะพบ 1 (Likelihood: [ระดับโอกาส]): [รายละเอียดข้อตรวจพบสั้นๆ]
- ข้อตรวจพบที่คาดว่าจะพบ 2 (Likelihood: [ระดับโอกาส]): [รายละเอียดข้อตรวจพบสั้นๆ]
... (ให้มาอย่างน้อย 3-5 ข้อตรวจพบ)
"""
                    response_finding = client.chat.completions.create(
                        model="typhoon-v2.1-12b-instruct",
                        messages=[{"role": "user", "content": prompt_finding}],
                        temperature=0.7, max_tokens=1024
                    )
                    st.session_state["gen_findings"] = response_finding.choices[0].message.content

                    # 3. Generate High-Level Report Structure
                    prompt_report = f"""
คุณเป็นผู้เชี่ยวชาญด้านการตรวจสอบผลการดำเนินงาน (Performance Audit) หน้าที่ของคุณคือวิเคราะห์ข้อมูลทั้งหมดในแผนการตรวจสอบ และโครงสร้าง Logic Model, Risks, และ Issues ที่กำหนด เพื่อสร้างโครงร่างสรุปรายงาน (High-Level Report Outline) สำหรับการตรวจสอบนี้

ข้อมูลแผน:
---
{context_data}
---

ประเด็นการตรวจสอบที่แนะนำ:
---
{st.session_state["gen_issues"]}
---

รูปแบบผลลัพธ์:
[หัวข้อสรุปรายงานระดับสูง 1]
    - ประเด็นย่อย 1.1
    - ประเด็นย่อย 1.2
[หัวข้อสรุปรายงานระดับสูง 2]
    - ประเด็นย่อย 2.1
    - ประเด็นย่อย 2.2
...
"""
                    response_report = client.chat.completions.create(
                        model="typhoon-v2.1-12b-instruct",
                        messages=[{"role": "user", "content": prompt_report}],
                        temperature=0.7, max_tokens=1024
                    )
                    st.session_state["gen_report"] = response_report.choices[0].message.content
                    
                    st.success("ประมวลผลเสร็จสิ้น! กรุณาดูผลลัพธ์ด้านล่าง")
                    st.balloons()
                    
                except Exception as e:
                    # ตรวจสอบประเภทข้อผิดพลาดจาก OpenAI
                    error_type = type(e).__name__
                    if "AuthenticationError" in str(e):
                        error_message = f"เกิดข้อผิดพลาดในการเชื่อมต่อ API: (AuthenticationError) โปรดตรวจสอบ **API Key** ของคุณ"
                    elif "RateLimitError" in str(e):
                        error_message = f"เกิดข้อผิดพลาดในการเชื่อมต่อ API: (RateLimitError) โปรดตรวจสอบ **ขีดจำกัดการใช้งาน (Rate Limit)** ของคุณ"
                    elif "APIError" in str(e):
                        error_message = f"เกิดข้อผิดพลาดในการเชื่อมต่อ API: ({error_type}) โปรดตรวจสอบ API Key หรือขีดจำกัดการใช้งาน (Rate Limit) ของคุณ\nรายละเอียด: {e}"
                    else:
                         error_message = f"เกิดข้อผิดพลาดขณะทำงาน: ({error_type}) โปรดลองอีกครั้ง\nรายละเอียด: {e}"
                         
                    st.error(error_message)
                    st.session_state["gen_issues"] = ""
                    st.session_state["gen_findings"] = ""
                    st.session_state["gen_report"] = ""

    st.markdown("<h4 style='color:blue;'>ประเด็นการตรวจสอบที่ควรให้ความสำคัญ</h4>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color: #f0f2f6; border: 1px solid #ccc; padding: 10px; border-radius: 5px; height: 200px; overflow-y: scroll;'>{st.session_state.get('gen_issues', '')}</div>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='color:blue;'>ข้อตรวจพบที่คาดว่าจะพบ (พร้อมระดับโอกาส)</h4>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color: #f0f2f6; border: 1px solid #ccc; padding: 10px; border-radius: 5px; height: 200px; overflow-y: scroll;'>{st.session_state.get('gen_findings', '')}</div>", unsafe_allow_html=True)

    st.markdown("<h4 style='color:blue;'>โครงร่างรายงานเบื้องต้น (High-Level Report Outline)</h4>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color: #f0f2f6; border: 1px solid #ccc; padding: 10px; border-radius: 5px; height: 300px; overflow-y: scroll;'>{st.session_state.get('gen_report', '')}</div>", unsafe_allow_html=True)

# ----------------- Tab 9: 💬 PA Chat (ถาม-ตอบเ) -----------------
with tab_chatbot:
    set_current_tab("💬 PA Chat (ถาม-ตอบเ)")
    # ... (เนื้อหาของ Tab 9 ยังคงอยู่เหมือนเดิม)
    st.subheader("💬 PA Chat Assistant - ถาม-ตอบเอกสารอ้างอิงภายใน")
    
    # ----------------- Configuration Section -----------------
    with st.expander("⬆️ จัดการเอกสารและ API Key (คลิกเพื่อเปิด)"):
        
        # *** ใช้ API Key Global และกำหนด Callback เพื่อบันทึก Key ***
        def save_api_key_global_chat():
            st.session_state.api_key_global = st.session_state.api_key_input_chat
            st.toast("บันทึก API Key แล้ว!")
            
        st.markdown("💡 **ยังไม่มี API Key?** คลิก [ที่นี่](https://playground.opentyphoon.ai/settings/api-key) เพื่อรับ key ฟรี!")
        api_key_chat = st.text_input(
            "กรุณากรอก API Key เพื่อใช้บริการ AI:", 
            type="password", 
            value=st.session_state.api_key_global,
            key="api_key_input_chat",
            on_change=save_api_key_global_chat
        )
        api_key_chat = st.session_state.api_key_global
        
        st.markdown("---")
        
        st.markdown("##### 📄 เอกสารอ้างอิง")
        st.markdown(f"**จำกัดข้อความรวม:** {MAX_CHARS_LIMIT:,} ตัวอักษร")
        
        # 1. เอกสารที่โหลดจากโฟลเดอร์ Doc
        local_context_len = len(st.session_state.doc_context_local)
        st.markdown(f"**เอกสารในโฟลเดอร์ `Doc`:** {local_context_len:,} ตัวอักษร (โหลดอัตโนมัติ)")
        
        # 2. เอกสารที่ผู้ใช้อัปโหลด
        uploaded_files = st.file_uploader(
            "อัปโหลดเอกสารเพิ่มเติม (PDF/TXT/CSV)", 
            type=["pdf", "txt", "csv"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("ประมวลผลเอกสารที่อัปโหลด", key="process_docs_btn"):
                # Clear previous uploaded context before processing new ones
                st.session_state.doc_context_uploaded = ""
                # Calculate remaining space based on local context
                remaining_limit = MAX_CHARS_LIMIT - local_context_len
                
                if remaining_limit <= 0:
                    st.error("ไม่สามารถอัปโหลดได้ เนื่องจากเอกสารในโฟลเดอร์ `Doc` มีจำนวนถึงขีดจำกัดสูงสุดแล้ว")
                else:
                    st.session_state.doc_context_uploaded, chars_added = process_documents(
                        uploaded_files, 
                        'uploaded', 
                        remaining_limit, 
                        0
                    )
                    st.success(f"โหลดเอกสารที่อัปโหลดเพิ่ม {chars_added:,} ตัวอักษร. (รวมทั้งหมด: {local_context_len + chars_added:,} ตัวอักษร)")
                    
            if st.session_state.doc_context_uploaded:
                 st.info(f"เอกสารที่อัปโหลดล่าสุด: {len(st.session_state.doc_context_uploaded):,} ตัวอักษร")

    # ----------------- Chat Interface -----------------
    
    # รวมบริบทเอกสารทั้งหมด (Local + Uploaded)
    doc_context = st.session_state.doc_context_local + st.session_state.doc_context_uploaded
    total_context_len = len(doc_context)
    
    if total_context_len > 0:
        st.markdown(f"**บริบทเอกสารรวม:** <span style='color:green;'>{total_context_len:,} ตัวอักษร</span> / {MAX_CHARS_LIMIT:,} ตัวอักษร", unsafe_allow_html=True)
    else:
        st.warning("ไม่มีเอกสารอ้างอิง โปรดอัปโหลดไฟล์ หรือสร้างโฟลเดอร์ 'Doc' พร้อมไฟล์ภายใน")

    # แสดงข้อความแชท
    for message in st.session_state.chatbot_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # การรับ Input จากผู้ใช้
    if prompt := st.chat_input("กรุณาป้อนคำถามของคุณ..."):
        
        # 1. เพิ่มข้อความผู้ใช้ในประวัติ
        st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not api_key_chat:
            response = "ขออภัยครับ กรุณาใส่ API Key ในส่วน 'จัดการเอกสารและ API Key' ก่อนใช้งาน"
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.chatbot_messages.append({"role": "assistant", "content": response})
        elif total_context_len == 0:
            response = "ขออภัยครับ ไม่พบเอกสารอ้างอิง (RAG Context) โปรดตรวจสอบว่าคุณได้อัปโหลดเอกสาร หรือมีไฟล์ในโฟลเดอร์ 'Doc' และโหลดเรียบร้อยแล้ว"
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.chatbot_messages.append({"role": "assistant", "content": response})
        else:
            # 2. เรียกใช้ AI
            with st.chat_message("assistant"):
                with st.spinner("กำลังค้นหาและประมวลผลคำตอบ..."):
                    try:
                        client = OpenAI(
                            api_key=api_key_chat,
                            base_url="https://api.opentyphoon.ai/v1"
                        )
                        
                        system_prompt = f"""
คุณคือผู้ช่วย AI อัจฉริยะ (PA Assistant) ที่เชี่ยวชาญด้านการตรวจสอบผลการดำเนินงาน (Performance Audit)
หน้าที่ของคุณคือตอบคำถามของผู้ใช้โดยยึดบริบท (Context) จากเอกสารภายในที่ให้มาในส่วน "บริบทจากเอกสารภายใน" เป็นหลัก

หลักการตอบ:
- **ต้อง** อ้างอิงคำตอบจาก "บริบทจากเอกสารภายใน" เท่านั้น หากข้อมูลในเอกสารและข้อมูลทั่วไปขัดแย้งกัน ให้ยึดข้อมูลในเอกสารเป็นหลักและอาจกล่าวถึงความขัดแย้งนั้น
- หากไม่พบคำตอบทั้งในเอกสารและความรู้ทั่วไป ให้ตอบว่า "ขออภัยครับ ไม่พบข้อมูลที่เกี่ยวข้องทั้งในเอกสารและฐานข้อมูลของผม"

---
**บริบทจากเอกสารภายใน:**
{doc_context}
---

จากข้อมูลข้างต้นนี้ จงตอบคำถามล่าสุดของผู้ใช้
"""
                        
                        messages_for_api = [
                            {"role": "system", "content": system_prompt}
                        ]
                        # Add chat history, but keep it concise
                        for msg in st.session_state.chatbot_messages[-10:]:
                            messages_for_api.append(msg)
                        
                        response_stream = client.chat.completions.create(
                            model="typhoon-v2.1-12b-instruct",
                            messages=messages_for_api,
                            temperature=0.5,
                            max_tokens=3072,
                            stream=True
                        )
                        
                        response = st.write_stream(response_stream)
                        st.session_state.chatbot_messages.append({"role": "assistant", "content": response})

                    except Exception as e:
                        error_message = f"เกิดข้อผิดพลาด: {e}"
                        st.error(error_message)
                        st.session_state.chatbot_messages.append({"role": "assistant", "content": error_message})
