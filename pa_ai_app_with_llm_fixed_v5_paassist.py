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

st.set_page_config(page_title="Planning Studio (+ Issue Suggestions)", page_icon="🧭", layout="wide")

# ----------------- Session Init -----------------
def init_state():
    ss = st.session_state
    ss.setdefault("plan", {
        "plan_id": "PLN-" + datetime.now().strftime("%y%m%d-%H%M%S"),
        "plan_title": "",
        "program_name": "",
        "who": "", "what": "", "where": "", "when": "", "why": "", "how": "", "how_much": "", "whom": "",
        "objectives": "", "scope": "", "assumptions": "", "status": "Draft"
    })
    ss.setdefault("logic_items", pd.DataFrame(columns=[
        "item_id","plan_id","type","description","metric","unit","target","source"
    ]))
    ss.setdefault("methods", pd.DataFrame(columns=[
        "method_id","plan_id","type","tool_ref","sampling","questions","linked_issue","data_source","frequency"
    ]))
    ss.setdefault("kpis", pd.DataFrame(columns=[
        "kpi_id","plan_id","level","name","formula","numerator","denominator","unit","baseline","target","frequency","data_source","quality_requirements"
    ]))
    ss.setdefault("risks", pd.DataFrame(columns=[
        "risk_id","plan_id","description","category","likelihood","impact","mitigation","hypothesis"
    ]))
    ss.setdefault("audit_issues", pd.DataFrame(columns=[
        "issue_id","plan_id","title","rationale","linked_kpi","proposed_methods","source_finding_id","issue_detail","recommendation"
    ]))
    ss.setdefault("gen_issues", "")
    ss.setdefault("gen_findings", "")
    ss.setdefault("gen_report", "")
    ss.setdefault("issue_results", pd.DataFrame())
    ss.setdefault("ref_seed", "") 
    ss.setdefault("issue_query_text", "")
    ss.setdefault("chatbot_messages", [
        {"role": "assistant", "content": "สวัสดีครับ ผมคือผู้ช่วยตรวจสอบ (PA Chatbot) ผมพร้อมตอบคำถามจากคู่มือการตรวจสอบ PA และข้อมูลบนอินเทอร์เน็ตแล้วครับ"}
    ])
    ss.setdefault("doc_context", "")

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
                    st.warning("ไม่พบชีตชื่อ 'Data' จะอ่านจากชีตแรกแทน")
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
    return output.getvalue()

# ----------------- App UI -----------------
init_state()
plan = st.session_state["plan"]
logic_df = st.session_state["logic_items"]
methods_df = st.session_state["methods"]
kpis_df = st.session_state["kpis"]
risks_df = st.session_state["risks"]
audit_issues_df = st.session_state["audit_issues"]

st.title("🧭 Planning Studio – Performance Audit")

# ----------------- START: Custom CSS (grouped tabs via nth-of-type) -----------------
st.markdown("""
<style>
/* ---- Global Font ---- */
body { font-family: 'Kanit', sans-serif; }

/* ---- Base Style for Tabs ---- */
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
}
button[data-baseweb="tab"][aria-selected="true"] {
    box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
}

/* ---- Group 1: 1-5 (Blue) ---- */
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

/* ---- Group 2: 6-7 (Purple) ---- */
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

/* ---- Group 3: 8-9 (Gold) ---- */
div[data-baseweb="tab-list"] button:nth-of-type(8),
div[data-baseweb="tab-list"] button:nth-of-type(9) {
    border-color: #ffc107;
    color: #cc9900 !important;
    box-shadow: 0 0 5px rgba(255, 193, 7, 0.5);
}
div[data-baseweb="tab-list"] button:nth-of-type(8)[aria-selected="true"],
div[data-baseweb="tab-list"] button:nth-of-type(9)[aria-selected="true"] {
    background-color: #ffc107;
    border-color: #ffc107;
    color: #333333 !important;
}

/* ---- Responsive ---- */
div[data-baseweb="tab-list"] { border-bottom: none !important; margin-bottom: 15px; flex-wrap: wrap; }
@media (max-width: 768px) {
    .st-emotion-cache-18ni2cb, .st-emotion-cache-1jm69l4 {
        width: 100% !important;
        margin-bottom: 1rem;
    }
}

/* ---- Headers ---- */
h4 { color: #007bff !important; border-bottom: 2px solid #e0e0e0; padding-bottom: 5px; }
</style>
""", unsafe_allow_html=True)
# ----------------- END: Custom CSS -----------------

# ----------------- Tabs -----------------
tab_plan, tab_logic, tab_method, tab_kpi, tab_risk, tab_issue, tab_preview, tab_assist, tab_chatbot = st.tabs([
    "1. ระบุ แผน & 6W2H", 
    "2. ระบุ Logic Model", 
    "3. ระบุ Methods", 
    "4. ระบุ KPIs", 
    "5. ระบุ Risks", 
    "6. ค้นหาข้อตรวจพบที่ผ่านมา", 
    "7. สรุปข้อมูล (Preview)", 
    "✨ ให้ PA Assist ช่วย", 
    "🤖 คุยกับ PA Chatbot"
])

# ----------------- Tab 1: แผน & 6W2H -----------------
with tab_plan:
    st.subheader("ข้อมูลแผน (Plan) - กรุณาระบุข้อมูล")
    with st.container(border=True):
        c1, c2 = st.columns([2,1])
        with c1:
            plan["plan_title"] = st.text_input("ชื่อแผน/งานตรวจ", value=plan.get("plan_title",""))
            plan["program_name"] = st.text_input("หน่วยรับตรวจ/โครงการ/โปรแกรม", value=plan.get("program_name",""))
        with c2:
            st.write("รหัสแผน:", plan["plan_id"])
            plan["status"] = st.selectbox("สถานะ", ["Draft","Planned","In Progress","Completed"], index=["Draft","Planned","In Progress","Completed"].index(plan.get("status","Draft")))
    st.markdown("**6W2H**")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        plan["who"] = st.text_input("Who (ใคร)", value=plan.get("who",""))
        plan["whom"] = st.text_input("Whom (กับใคร/ผู้รับผล)", value=plan.get("whom",""))
    with c2:
        plan["what"] = st.text_input("What (ทำอะไร)", value=plan.get("what",""))
        plan["when"] = st.text_input("When (เมื่อไร)", value=plan.get("when",""))
    with c3:
        plan["where"] = st.text_input("Where (ที่ไหน)", value=plan.get("where",""))
        plan["why"] = st.text_input("Why (ทำไม)", value=plan.get("why",""))
    with c4:
        plan["how"] = st.text_input("How (อย่างไร)", value=plan.get("how",""))
        plan["how_much"] = st.text_input("How much (ทรัพยากร/งบ)", value=plan.get("how_much",""))
    st.session_state["plan"] = plan
    st.success("บันทึกข้อมูล 6W2H ในหน่วยความจำแอปแล้ว")

# ----------------- Tab 2: Logic Model -----------------
with tab_logic:
    st.subheader("Logic Model")
    with st.expander("เพิ่มรายการ", expanded=True):
        c1,c2,c3 = st.columns([2,2,1])
        with c1:
            t = st.selectbox("ประเภท", ["Input","Activity","Output","Outcome","Impact"])
        with c2:
            desc = st.text_area("คำอธิบาย", height=80)
        with c3:
            add = st.button("➕ เพิ่ม")
        if add and desc.strip():
            rid = next_id("LM", logic_df, "item_id")
            new = pd.DataFrame([{
                "item_id": rid, "plan_id": plan["plan_id"], "type": t,
                "description": desc, "metric": "", "unit": "", "target": "", "source":""
            }])
            st.session_state["logic_items"] = pd.concat([logic_df, new], ignore_index=True)
            st.rerun()
    if not st.session_state["logic_items"].empty:
        st.dataframe(st.session_state["logic_items"], use_container_width=True)
        df_download_link(st.session_state["logic_items"], f"{plan['plan_id']}_logic_items.csv", "⬇️ ดาวน์โหลด Logic Items")

# ----------------- Tab 3: Methods -----------------
with tab_method:
    st.subheader("Methods")
    with st.expander("เพิ่มวิธีการตรวจ", expanded=True):
        c1,c2 = st.columns([1,3])
        with c1:
            mtype = st.selectbox("ประเภท", ["Document Review","Interview","Survey","Observation","Data Analytics","GIS/Remote Sensing","Experiment/Test"])
        with c2:
            q = st.text_area("ประเด็น/คำถามหลัก", height=80)
        c3,c4,c5 = st.columns([1,1,1])
        with c3:
            sampling = st.text_input("Sampling/กลุ่มตัวอย่าง", "")
        with c4:
            tool_ref = st.text_input("เครื่องมือ/แบบฟอร์มอ้างอิง", "")
        with c5:
            freq = st.text_input("ความถี่/รอบการเก็บ", "")
        add_m = st.button("➕ เพิ่มวิธีการ")
        if add_m and q.strip():
            mid = next_id("MTH", methods_df, "method_id")
            new = pd.DataFrame([{
                "method_id": mid, "plan_id": plan["plan_id"], "type": mtype, "tool_ref": tool_ref,
                "sampling": sampling, "questions": q, "linked_issue": "", "data_source": "", "frequency": freq
            }])
            st.session_state["methods"] = pd.concat([methods_df, new], ignore_index=True)
            st.rerun()
    if not st.session_state["methods"].empty:
        st.dataframe(st.session_state["methods"], use_container_width=True)
        df_download_link(st.session_state["methods"], f"{plan['plan_id']}_methods.csv", "⬇️ ดาวน์โหลด Methods")

# ----------------- Tab 4: KPIs -----------------
with tab_kpi:
    st.subheader("KPIs")
    with st.expander("เพิ่ม KPI", expanded=True):
        c1,c2 = st.columns(2)
        with c1:
            level = st.selectbox("ระดับ", ["Output","Outcome","Impact"])
            name = st.text_input("ชื่อ KPI")
            unit = st.text_input("หน่วย")
        with c2:
            formula = st.text_input("สูตรคำนวณ")
            baseline = st.text_input("ค่า Base/ปีฐาน")
            target = st.text_input("ค่าเป้าหมาย")
        c3,c4 = st.columns(2)
        with c3:
            numerator = st.text_input("ตัวตั้ง (ถ้ามี)")
            denominator = st.text_input("ตัวหาร (ถ้ามี)")
        with c4:
            freq = st.text_input("ความถี่/รอบการวัด")
            source = st.text_input("แหล่งข้อมูล")
        add_k = st.button("➕ เพิ่ม KPI")
        if add_k and name.strip():
            kid = next_id("KPI", kpis_df, "kpi_id")
            new = pd.DataFrame([{
                "kpi_id": kid, "plan_id": plan["plan_id"], "level": level, "name": name, "formula": formula,
                "numerator": numerator, "denominator": denominator, "unit": unit, "baseline": baseline,
                "target": target, "frequency": freq, "data_source": source, "quality_requirements": ""
            }])
            st.session_state["kpis"] = pd.concat([kpis_df, new], ignore_index=True)
            st.rerun()
    if not st.session_state["kpis"].empty:
        st.dataframe(st.session_state["kpis"], use_container_width=True)
        df_download_link(st.session_state["kpis"], f"{plan['plan_id']}_kpis.csv", "⬇️ ดาวน์โหลด KPIs")

# ----------------- Tab 5: Risks -----------------
with tab_risk:
    st.subheader("Risks")
    with st.expander("เพิ่มความเสี่ยง", expanded=True):
        c1,c2 = st.columns(2)
        with c1:
            desc = st.text_area("คำอธิบายความเสี่ยง", height=80)
            cat = st.selectbox("หมวดหมู่", ["Strategic","Operational","Financial","Compliance","Reputation","Technology","Climate/Environment"])
        with c2:
            like = st.slider("โอกาสเกิด (1-5)", 1, 5, 3)
            impact = st.slider("ผลกระทบ (1-5)", 1, 5, 3)
            miti = st.text_area("แนวทางลดความเสี่ยง", height=80)
        add_r = st.button("➕ เพิ่มความเสี่ยง")
        if add_r and desc.strip():
            rid = next_id("RSK", risks_df, "risk_id")
            new = pd.DataFrame([{
                "risk_id": rid, "plan_id": plan["plan_id"], "description": desc, "category": cat,
                "likelihood": like, "impact": impact, "mitigation": miti, "hypothesis": ""
            }])
            st.session_state["risks"] = pd.concat([risks_df, new], ignore_index=True)
            st.rerun()
    if not st.session_state["risks"].empty:
        st.dataframe(st.session_state["risks"], use_container_width=True)
        df_download_link(st.session_state["risks"], f"{plan['plan_id']}_risks.csv", "⬇️ ดาวน์โหลด Risks")

# ----------------- Tab 6: ค้นหาข้อตรวจพบที่ผ่านมา -----------------
with tab_issue:
    st.subheader("ค้นหาข้อตรวจพบที่ผ่านมา (TF-IDF + Cosine)")
    uploaded = st.file_uploader("อัปโหลด FindingsLibrary (.csv/.xlsx)", type=["csv","xlsx","xls"])
    findings_df = load_findings(uploaded)
    if not findings_df.empty:
        st.caption(f"รายการทั้งหมด: {len(findings_df):,} แถว")
        # สร้าง Index
        vec, X = build_tfidf_index(findings_df)
        query = st.text_area("ใส่คำอธิบายงาน/บริบท ที่ต้องการค้นหา", value=st.session_state.get("issue_query_text",""), height=120)
        colk, colbtn = st.columns([1,1])
        with colk:
            k = st.slider("จำนวนผลลัพธ์", 3, 15, 8)
        with colbtn:
            do = st.button("🔎 ค้นหา")
        if do and query.strip():
            st.session_state["issue_query_text"] = query
            results = search_candidates(query, findings_df, vec, X, top_k=k)
            st.session_state["issue_results"] = results
            st.dataframe(results, use_container_width=True)
            df_download_link(results, "issue_candidates.csv", "⬇️ ดาวน์โหลดผลลัพธ์")
    else:
        st.info("ยังไม่มีฐานข้อมูล FindingsLibrary.csv ในโฟลเดอร์ หรือยังไม่ได้อัปโหลดไฟล์")

# ----------------- Tab 7: สรุปข้อมูล (Preview) -----------------
with tab_preview:
    st.subheader("สรุปข้อมูลแผน (Preview)")
    st.markdown(f"**รหัสแผน:** {plan['plan_id']}  \n**ชื่อแผน:** {plan.get('plan_title','')}")
    st.markdown("### 6W2H")
    st.json({k:plan.get(k,"") for k in ["who","whom","what","where","when","why","how","how_much"]})
    st.markdown("### Logic Model")
    st.dataframe(st.session_state["logic_items"], use_container_width=True)
    st.markdown("### Methods")
    st.dataframe(st.session_state["methods"], use_container_width=True)
    st.markdown("### KPIs")
    st.dataframe(st.session_state["kpis"], use_container_width=True)
    st.markdown("### Risks")
    st.dataframe(st.session_state["risks"], use_container_width=True)
    st.markdown("### ข้อเสนอประเด็นตรวจ (ถ้ามี)")
    st.dataframe(st.session_state.get("issue_results", pd.DataFrame()), use_container_width=True)

# ----------------- Tab 8: ให้ PA Assist ช่วย -----------------
with tab_assist:
    st.subheader("✨ ให้ PA Assist ช่วย (สังเคราะห์/สรุป/ข้อเสนอ)")
    st.info("ปลั๊กอิน LLM ยังไม่เชื่อม API ในไฟล์นี้ ตัวอย่างโครงสร้าง UI เท่านั้น")
    st.text_area("Seed / สมมติฐานเบื้องต้น", key="ref_seed", height=100)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧩 สร้างประเด็นตรวจ (Issues)"):
            st.session_state["gen_issues"] = "ตัวอย่าง: สร้างประเด็นตรวจจาก 6W2H, KPIs, Risks"
    with col2:
        if st.button("📚 สกัดบทเรียน/แนวปฏิบัติ (Findings/Lessons)"):
            st.session_state["gen_findings"] = "ตัวอย่าง: ดึงประเด็นจากฐาน FindingsLibrary"
    with col3:
        if st.button("📝 ร่างรายงานย่อ (Mini Report)"):
            st.session_state["gen_report"] = "ตัวอย่าง: โครงร่างรายงานฉบับย่อ"
    st.text_area("ผลลัพธ์ Issues", value=st.session_state.get("gen_issues",""), height=140)
    st.text_area("ผลลัพธ์ Findings", value=st.session_state.get("gen_findings",""), height=140)
    st.text_area("ผลลัพธ์ Report", value=st.session_state.get("gen_report",""), height=140)

# ----------------- Tab 9: คุยกับ PA Chatbot -----------------
with tab_chatbot:
    st.subheader("🤖 คุยกับ PA Chatbot")
    st.info("ตัวอย่าง UI แชตบอท (ยังไม่เชื่อมต่อโมเดล)")
    with st.form("chat_form", clear_on_submit=True):
        q = st.text_input("พิมพ์คำถามเกี่ยวกับแนวทาง/คู่มือการตรวจ PA")
        submitted = st.form_submit_button("ส่ง")
    if submitted and q.strip():
        st.session_state["chatbot_messages"].append({"role":"user","content":q})
        # ตอบตัวอย่าง echo
        st.session_state["chatbot_messages"].append({"role":"assistant","content":f"คุณถามว่า: {q}\n(เดโม) ระบบจะค้นเอกสาร PDF ในโฟลเดอร์ Doc/ แล้วตอบให้"})
    for m in st.session_state["chatbot_messages"]:
        if m["role"]=="user":
            st.markdown(f"**คุณ:** {m['content']}")
        else:
            st.markdown(f"**ผู้ช่วย:** {m['content']}")
