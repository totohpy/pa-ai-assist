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
from PyPDF2 import PdfReader
import streamlit.components.v1 as components

# ตั้งค่าหน้าเพจ
st.set_page_config(page_title="Planning Studio (+ Findings Suggestions)", page_icon="🧭", layout="wide")

# ----------------- ⚙️ การตั้งค่ากลาง -----------------
with st.sidebar:
    st.title("⚙️ การตั้งค่ากลาง")
    st.info("API Key ที่กรอกด้านล่างนี้จะถูกใช้กับทุกฟีเจอร์ AI, ดูรายละเอียด ? ")
    st.session_state.api_key_global = st.text_input(
        "กรุณากรอก API Key (OpenTyphoon)",
        type="password",
        key="api_key_global_input_sidebar",
        help="คลิกที่นี่เพื่อรับ Key ฟรี: https://playground.opentyphoon.ai/settings/api-key"
    )
    st.markdown("---")
    st.markdown("PA Planning Studio Web App By PAO1 DataCenter")

# ----------------- ฟังก์ชันสำหรับ Chatbot -----------------
MAX_CHARS_LIMIT = 200000

@st.cache_data(show_spinner=False)
def load_local_documents(folder_path="Doc"):
    text = ""
    if not os.path.isdir(folder_path): return text
    try:
        files_in_doc = os.listdir(folder_path)
        progress_bar = st.sidebar.progress(0, text=f"กำลังโหลดเอกสารจากคลังข้อมูล... (0/{len(files_in_doc)})")
        for i, filename in enumerate(files_in_doc):
            if len(text) >= MAX_CHARS_LIMIT:
                st.warning(f"ถึงขีดจำกัดข้อมูลจากคลังข้อมูลแล้ว ({MAX_CHARS_LIMIT:,} ตัวอักษร)"); break
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
            st.warning(f"ถึงขีดจำกัดจำนวนตัวอักษรสูงสุด ({limit:,}) แล้ว ไฟล์บางส่วนอาจไม่ถูกประมวลผล"); break
        try:
            if file.name.endswith('.pdf'):
                reader = PdfReader(file)
                for page in reader.pages: text += page.extract_text() or ""
            elif file.name.endswith('.txt'): text += file.getvalue().decode("utf-8")
            elif file.name.endswith('.csv'): df = pd.read_csv(file); text += df.head(15).to_string()
        except Exception as e: st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ {file.name}: {e}")
    return text[:limit - current_len], [f.name for f in files]

# ----------------- Session Init -----------------
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
    ss.setdefault("active_tab", 0)
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

def create_next_tab_button(next_tab_index, label="แท็บถัดไป »"):
    def switch_tab():
        st.session_state.active_tab = next_tab_index
    st.divider()
    _, col2 = st.columns([8, 2])
    col2.button(label, on_click=switch_tab, use_container_width=True, key=f"next_tab_btn_{next_tab_index}")

@st.cache_data(show_spinner=False)
def load_findings(uploaded=None):
    findings_df = pd.DataFrame()
    findings_db_path = "FindingsLibrary.csv"
    if os.path.exists(findings_db_path):
        try: findings_df = pd.read_csv(findings_db_path)
        except Exception as e: st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ FindingsLibrary.csv: {e}")
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'): uploaded_df = pd.read_csv(uploaded)
            elif uploaded.name.endswith(('.xlsx', '.xls')):
                xls = pd.ExcelFile(uploaded)
                sheet_name = "Data" if "Data" in xls.sheet_names else 0
                uploaded_df = pd.read_excel(xls, sheet_name=sheet_name)
            if not uploaded_df.empty:
                findings_df = pd.concat([findings_df, uploaded_df], ignore_index=True)
                st.success(f"อัปโหลดไฟล์ '{uploaded.name}' และรวมกับฐานข้อมูลเดิมแล้ว")
        except Exception as e: st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ที่อัปโหลด: {e}")
    if not findings_df.empty:
        for c in ["issue_title","issue_detail","cause_detail","recommendation","program","unit"]:
            if c in findings_df.columns: findings_df[c] = findings_df[c].fillna("")
        if "year" in findings_df.columns: findings_df["year"] = pd.to_numeric(findings_df["year"], errors="coerce").fillna(0).astype(int)
        if "severity" in findings_df.columns: findings_df["severity"] = pd.to_numeric(findings_df["severity"], errors="coerce").fillna(3).clip(1,5).astype(int)
    return findings_df

@st.cache_resource(show_spinner=False)
def build_tfidf_index(findings_df: pd.DataFrame):
    texts = (findings_df["issue_title"].fillna("") + " " + findings_df["issue_detail"].fillna("") + " " + findings_df["cause_detail"].fillna("") + " " + findings_df["recommendation"].fillna(""))
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2)); X = vec.fit_transform(texts)
    return vec, X

def search_candidates(query_text, findings_df, vec, X, top_k=8):
    qv = vec.transform([query_text]); sims = cosine_similarity(qv, X)[0]
    out = findings_df.copy(); out["sim_score"] = sims
    out["year_norm"] = (out["year"] - out["year"].min()) / (out["year"].max() - out["year"].min()) if "year" in out.columns and out["year"].max() != out["year"].min() else 0.0
    out["sev_norm"] = out.get("severity", 3) / 5
    out["score"] = out["sim_score"]*0.65 + out["sev_norm"]*0.25 + out["year_norm"]*0.10
    cols = ["finding_id","year","unit","program","issue_title","issue_detail","cause_category","cause_detail","recommendation","outcomes_impact","severity","score", "sim_score"]
    return out.sort_values("score", ascending=False).head(top_k)[[c for c in cols if c in out.columns]]
    
def create_excel_template():
    df = pd.DataFrame(columns=["finding_id", "issue_title", "unit", "program", "year", "cause_category", "cause_detail", "issue_detail", "recommendation", "outcomes_impact", "severity"])
    output = io.BytesIO();
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df.to_excel(writer, index=False, sheet_name='FindingsLibrary')
    return output.getvalue()

init_state()
plan = st.session_state["plan"]
logic_df = st.session_state["logic_items"]
methods_df = st.session_state["methods"]
kpis_df = st.session_state["kpis"]
risks_df = st.session_state["risks"]
audit_issues_df = st.session_state["audit_issues"]

st.title("🧭 Planning Studio – Performance Audit")

st.markdown("""<style> body { font-family: 'Kanit', sans-serif; } button[data-baseweb="tab"] { border: 1px solid #007bff; border-radius: 8px; padding: 10px 15px; margin: 5px 5px 5px 0px; font-weight: bold; color: #007bff !important; background-color: #ffffff; box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1); border-bottom: none !important; &::after { content: none !important; } } button[data-baseweb="tab"][aria-selected="true"] { box-shadow: 2px 2px 5px rgba(0,0,0,0.2); } div[data-baseweb="tab-list"] button:nth-of-type(-n+5) { border-color: #007bff; color: #007bff !important; } div[data-baseweb="tab-list"] button:nth-of-type(-n+5)[aria-selected="true"] { background-color: #007bff; color: white !important; } div[data-baseweb="tab-list"] button:nth-of-type(6), div[data-baseweb="tab-list"] button:nth-of-type(7) { border-color: #6f42c1; color: #6f42c1 !important; } div[data-baseweb="tab-list"] button:nth-of-type(6)[aria-selected="true"], div[data-baseweb="tab-list"] button:nth-of-type(7)[aria-selected="true"] { background-color: #6f42c1; color: white !important; } div[data-baseweb="tab-list"] button:nth-of-type(8) { border-color: #ffc107; color: #cc9900 !important; } div[data-baseweb="tab-list"] button:nth-of-type(8)[aria-selected="true"] { background-color: #ffc107; color: #333333 !important; } div[data-baseweb="tab-list"] button:nth-of-type(9) { border-color: #28a745; color: #28a745 !important; } div[data-baseweb="tab-list"] button:nth-of-type(9)[aria-selected="true"] { background-color: #28a745; color: white !important; } div[data-baseweb="tab-list"] { border-bottom: none !important; margin-bottom: 15px; flex-wrap: wrap; gap: 10px; } h4 { color: #007bff !important; border-bottom: 2px solid #e0e0e0; padding-bottom: 5px; } </style>""", unsafe_allow_html=True)

tab_names = ["1. ระบุ แผน & 6W2H", "2. ระบุ Logic Model", "3. ระบุ Methods", "4. ระบุ KPIs", "5. ระบุ Risks", "6. ค้นหาข้อตรวจพบที่ผ่านมา", "7. สรุปข้อมูล (Preview)", "🤖 ให้ PA Assistant แนะนำประเด็นการตรวจสอบ ✨✨", "💬 PA Chat (ถาม-ตอบ)"]
tabs = st.tabs(tab_names) 

if "active_tab" in st.session_state:
    active_tab_js = f"""
    <script>
        var tab_buttons = parent.document.querySelectorAll('button[data-baseweb="tab"]');
        var tab_index = {st.session_state.active_tab};
        if (tab_buttons.length > tab_index && tab_buttons[tab_index].getAttribute('aria-selected') === 'false') {{
            tab_buttons[tab_index].click();
        }}
    </script>
    """
    components.html(active_tab_js, height=0, width=0)

with tabs[0]:
    st.subheader("ข้อมูลแผน (Plan) - กรุณาระบุข้อมูล")
    # ... content ...
    create_next_tab_button(1)

with tabs[1]:
    st.subheader("ระบุข้อมูล Logic Model: Input → Activities → Output → Outcome → Impact")
    st.dataframe(logic_df, use_container_width=True, hide_index=True)
    with st.expander("➕ เพิ่มรายการใน Logic Model"):
        with st.container(border=True):
            colA, colB, colC = st.columns(3)
            typ = colA.selectbox("ประเภท", ["Input","Activity","Output","Outcome","Impact"], key="logic_type")
            desc = colA.text_input("คำอธิบาย/รายละเอียด", key="logic_desc")
            metric = colA.text_input("ตัวชี้วัด/metric", key="logic_metric")
            unit = colB.text_input("หน่วย", value="หน่วย", key="logic_unit")
            target = colB.text_input("เป้าหมาย", value="", key="logic_target")
            source = colC.text_input("แหล่งข้อมูล", value="", key="logic_source")
            # --- FIXED ---: เพิ่ม key ที่ไม่ซ้ำกันให้กับปุ่ม
            if st.button("เพิ่ม Logic Item", type="primary", key="add_logic_item_btn"):
                new_row = pd.DataFrame([{"item_id": next_id("LG", logic_df, "item_id"),"plan_id": plan["plan_id"],"type": typ, "description": desc, "metric": metric,"unit": unit, "target": target, "source": source}])
                st.session_state["logic_items"] = pd.concat([logic_df, new_row], ignore_index=True)
                st.rerun()
    create_next_tab_button(2)

with tabs[2]:
    st.subheader("ระบุวิธีการเก็บข้อมูล (Methods)")
    # ... content ...
    create_next_tab_button(3)

with tabs[3]:
    st.subheader("ระบุตัวชี้วัด (KPIs)")
    # ... content ...
    create_next_tab_button(4)

with tabs[4]:
    st.subheader("ระบุความเสี่ยง (Risks)")
    # ... content ...
    create_next_tab_button(5)
    
with tabs[5]:
    st.subheader("🔎 แนะนำประเด็นตรวจจากรายงานเก่า (Audit Findings Suggestions)")
    # ... content ...
    create_next_tab_button(6)

with tabs[6]:
    st.subheader("สรุปแผน (Preview)")
    # ... content ...
    create_next_tab_button(7)

with tabs[7]:
    st.subheader("💡 PA Audit Assistant (AI/LLM)")
    # ... content ...
    create_next_tab_button(8)

with tabs[8]:
    st.subheader("💬 PA Chat - ผู้ช่วยอัจฉริยะ (Typhoon AI)")
    # ... content ...
    # No next button on the last tab
