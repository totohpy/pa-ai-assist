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
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(page_title="Design Assistant", page_icon="✨", layout="wide")

# (โค้ดส่วนที่เหลือทั้งหมดจากไฟล์ pa_assistant.py เดิมจะอยู่ที่นี่...
# ...ยกเว้นส่วนของ Chatbot ซึ่งจะถูกย้ายไปอีกไฟล์)

# ----------------- ⚙️ การตั้งค่ากลาง -----------------
with st.sidebar:
    st.title("⚙️ PA Planning Studio")
    st.info("ระบบมีฟีเจอร์ AI อาจทำผิดพลาดได้ ดังนั้น โปรดตรวจสอบคำตอบอีกครั้ง")
    try:
        st.session_state.api_key_global = st.secrets["api_key"]
    except KeyError:
        st.session_state.api_key_global = ""
        st.warning("ฟีเจอร์ AI ยังไม่พร้อมใช้งาน กรุณาติดต่อผู้ดูแลระบบ")
    except Exception as e:
        st.session_state.api_key_global = ""
        st.error(f"เกิดข้อผิดพลาดในการโหลด API Key: {e}")
    st.markdown("---")
    st.markdown(
        '<p style="font-family: \'Kanit\', sans-serif;"><span style="color: grey;">By PAO1 </span><br><span style="font-size: 16px; letter-spacing: 0.5px;"><span style="color: red; font-weight: bold;">A</span>udit <span style="color: red; font-weight: bold;">I</span>ntelligence Team</span></p>',
        unsafe_allow_html=True
    )

# ----------------- ฟังก์ชันต่างๆ (เหมือนเดิม) -----------------
# (วางฟังก์ชันทั้งหมดจาก pa_assistant.py ที่นี่: init_state, next_id, df_download_link, load_findings, etc.)
def init_state():
    ss = st.session_state
    ss.setdefault("plan", {"plan_id": "PLN-" + datetime.now().strftime("%y%m%d-%H%M%S"),"plan_title": "","program_name": "","who": "", "what": "", "where": "", "when": "", "why": "", "how": "", "how_much": "", "whom": "","objectives": "", "scope": "", "assumptions": "", "status": "Draft"})
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
    ss.setdefault('api_key_global', '')
    ss.setdefault("6w2h_output", "") 

def next_id(prefix, df, col):
    if df.empty: return f"{prefix}-001"
    nums = []
    for x in df[col]:
        try: nums.append(int(str(x).split("-")[-1]))
        except: pass
    n = max(nums) + 1 if nums else 1
    return f"{prefix}-{n:03d}"

def df_download_link(df: pd.DataFrame, filename: str, label: str):
    buf = BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="text/csv")

@st.cache_data(show_spinner=False)
def load_findings(uploaded=None):
    findings_df = pd.DataFrame()
    findings_db_path = "FindingsLibrary.csv"
    if os.path.exists(findings_db_path):
        try: findings_df = pd.read_csv(findings_db_path)
        except Exception as e: st.error(f"เกิดข้อผิดพลาดในการอ่าน FindingsLibrary.csv: {e}")
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'): uploaded_df = pd.read_csv(uploaded)
            elif uploaded.name.endswith(('.xlsx', '.xls')):
                xls = pd.ExcelFile(uploaded)
                sheet_name = "Data" if "Data" in xls.sheet_names else 0
                uploaded_df = pd.read_excel(xls, sheet_name=sheet_name)
            if not uploaded_df.empty:
                findings_df = pd.concat([findings_df, uploaded_df], ignore_index=True)
                st.success(f"อัปโหลด '{uploaded.name}' และรวมกับฐานข้อมูลเดิมแล้ว")
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
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    return vec, X

def search_candidates(query_text, findings_df, vec, X, top_k=8):
    qv = vec.transform([query_text])
    sims = cosine_similarity(qv, X)[0]
    out = findings_df.copy()
    out["sim_score"] = sims
    out["year_norm"] = (out["year"] - out["year"].min()) / (out["year"].max() - out["year"].min()) if "year" in out.columns and out["year"].max() != out["year"].min() else 0.0
    out["sev_norm"] = out.get("severity", 3) / 5
    out["score"] = out["sim_score"]*0.65 + out["sev_norm"]*0.25 + out["year_norm"]*0.10
    cols = ["finding_id","year","unit","program","issue_title","issue_detail","cause_category","cause_detail","recommendation","outcomes_impact","severity","score", "sim_score"]
    return out.sort_values("score", ascending=False).head(top_k)[[c for c in cols if c in out.columns]]
    
def create_excel_template():
    df = pd.DataFrame(columns=["finding_id", "issue_title", "unit", "program", "year", "cause_category", "cause_detail", "issue_detail", "recommendation", "outcomes_impact", "severity"])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df.to_excel(writer, index=False, sheet_name='FindingsLibrary')
    return output.getvalue()
    
def create_interactive_flowchart(df: pd.DataFrame):
    nodes = []
    edges = []
    styles = { "Objective": "#E6E6FA", "Input": "#a9def9", "Activity": "#e4c1f9", "Output": "#fcf6bd", "Outcome": "#d0f4de", "Impact": "#ff99c8"}
    sequence = ["Objective", "Input", "Activity", "Output", "Outcome", "Impact"]
    nodes_exist = []
    for i, item_type in enumerate(sequence):
        items_df = df[df['type'] == item_type]
        if not items_df.empty:
            header = item_type
            description_lines = []
            for _, row in items_df.iterrows():
                desc = str(row.get('description', '') or ''); metric = str(row.get('metric', '') or '')
                target = str(row.get('target', '') or ''); unit = str(row.get('unit', '') or '')
                number = target if target else metric
                line_parts = [f"• {desc}"]
                if number: line_parts.append(number)
                if unit: line_parts.append(unit)
                description_lines.append(" ".join(part for part in line_parts if part))
            label = f"{header}\n\n" + "\n".join(description_lines)
            nodes.append(Node(id=item_type, label=label, color=styles.get(item_type), shape="box", font={'face': 'Kanit', 'align': 'left'}, level=i))
            nodes_exist.append(item_type)
    if len(nodes_exist) > 1:
        for i in range(len(nodes_exist)-1):
            edges.append(Edge(source=nodes_exist[i], target=nodes_exist[i+1], dashes=False, color="#000000"))
    config = Config(width='100%', height=600, directed=True, physics=False, hierarchical={"enabled": True, "direction": "LR", "sortMethod": "directed"})
    return nodes, edges, config

init_state()
plan = st.session_state["plan"]
logic_df = st.session_state["logic_items"]
methods_df = st.session_state["methods"]
kpis_df = st.session_state["kpis"]
risks_df = st.session_state["risks"]
audit_issues_df = st.session_state["audit_issues"]

st.title("✨ Design Assistant")

with st.expander("💡 คำแนะนำการใช้งาน"):
    st.info("กรุณาระบุข้อมูล อย่างน้อย **ระบุ แผน & 6W2H** ส่วนใดส่วนหนึ่ง เพื่อค้นหาข้อตรวจพบที่ผ่านมาและให้ PA Assistant แนะนำ ได้แม่นยำที่สุด")

st.markdown("""
<style> 
    /* ... (CSS เหมือนเดิม) ... */
</style>
""", unsafe_allow_html=True)

tab_plan, tab_logic, tab_method, tab_kpi, tab_risk, tab_issue, tab_preview, tab_assist = st.tabs([
    "2.1) &nbsp;ระบุ แผน & 6W2H", 
    "2.2) &nbsp;ระบุ Logic Model", 
    "2.3) &nbsp;ระบุ Methods", 
    "2.4) &nbsp;ระบุ KPIs", 
    "2.5) &nbsp;ระบุ Risks", 
    "2.6) 🔍 ค้นหาข้อตรวจพบที่ผ่านมา", 
    "2.7) 📋 สรุปข้อมูล (Preview)", 
    "2.8) ✨ PA Assistant แนะนำประเด็น"
]) 

# (วางโค้ดสำหรับแต่ละ tab จากไฟล์ pa_assistant.py ที่นี่)
with tab_plan:
    # ... โค้ดของ tab_plan ...
    st.subheader("ข้อมูลแผน - กรุณาระบุข้อมูล")
    # ... (เนื้อหาเต็มๆ ของ tab_plan จากไฟล์ pa_assistant.py)
with tab_logic:
    # ... โค้ดของ tab_logic ...
with tab_method:
    # ... โค้ดของ tab_method ...
with tab_kpi:
    # ... โค้ดของ tab_kpi ...
with tab_risk:
    # ... โค้ดของ tab_risk ...
with tab_issue:
    # ... โค้ดของ tab_issue ...
with tab_preview:
    # ... โค้ดของ tab_preview ...
with tab_assist:
    # ... โค้ดของ tab_assist ...
