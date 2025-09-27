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
# Re-imported PyPDF2 for the new chatbot functionality
from PyPDF2 import PdfReader

# ตั้งค่าหน้าเพจ
st.set_page_config(page_title="Planning Studio (+ Issue Suggestions)", page_icon="🧭", layout="wide")

# ----------------- Global Settings in Sidebar -----------------
with st.sidebar:
    st.title("⚙️ การตั้งค่ากลาง")
    st.info("API Key ที่กรอกด้านล่างนี้จะถูกใช้กับทุกฟีเจอร์ AI ในแอปพลิเคชัน")
    st.session_state.api_key_global = st.text_input(
        "กรุณากรอก API Key (OpenTyphoon)",
        type="password",
        key="api_key_global_input",
        help="คลิกที่นี่เพื่อรับ Key ฟรี: https://playground.opentyphoon.ai/settings/api-key"
    )
    st.markdown("---")
    st.markdown("พัฒนาโดย [Planning Studio]")


# ----------------- Chatbot Helper Functions (NEW) -----------------
MAX_CHARS_LIMIT = 200000 # Set a character limit for RAG context

def process_documents(files, source_type, limit, current_len=0):
    """Extracts text from uploaded files (PDF, TXT, CSV)."""
    text = ""
    filenames = []
    limit_reached = False
    for file in files:
        if current_len + len(text) >= limit:
            st.warning(f"ถึงขีดจำกัดจำนวนตัวอักษรสูงสุด ({limit:,}) แล้ว ไฟล์บางส่วนอาจไม่ถูกประมวลผล")
            limit_reached = True
            break
        try:
            if file.name.endswith('.pdf'):
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            elif file.name.endswith('.txt'):
                text += file.getvalue().decode("utf-8")
            elif file.name.endswith('.csv'):
                 # Read only a sample to avoid overwhelming the context
                df = pd.read_csv(file)
                text += df.head(15).to_string() # Convert head of dataframe to string
            filenames.append(file.name)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ {file.name}: {e}")
    return text[:limit - current_len], filenames


# ----------------- Session Init -----------------
def init_state():
    ss = st.session_state
    # --- Existing States ---
    ss.setdefault("plan", {
        "plan_id": "PLN-" + datetime.now().strftime("%y%m%d-%H%M%S"),
        "plan_title": "", "program_name": "",
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

    # --- NEW States for Chatbot ---
    ss.setdefault('api_key_global', '')
    ss.setdefault('chatbot_messages', [{"role": "assistant", "content": "สวัสดีครับ ผมคือ PA Chat ผู้ช่วยอัจฉริยะด้านการตรวจสอบ คุณสามารถอัปโหลดเอกสารเพื่อให้ผมช่วยตอบคำถามได้ครับ"}])
    ss.setdefault('doc_context_local', "") # For pre-loaded docs, empty for now
    ss.setdefault('doc_context_uploaded', "") # For user-uploaded docs
    ss.setdefault('last_uploaded_files', set())


def next_id(prefix, df, col):
    if df.empty: return f"{prefix}-001"
    nums = [int(str(x).split("-")[-1]) for x in df[col] if str(x).split("-")[-1].isdigit()]
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
                sheet_name = "Data" if "Data" in xls.sheet_names else 0
                uploaded_df = pd.read_excel(xls, sheet_name=sheet_name)
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
    texts = (findings_df["issue_title"].fillna("") + " " + findings_df["issue_detail"].fillna(""))
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    return vec, X

def search_candidates(query_text, findings_df, vec, X, top_k=8):
    qv = vec.transform([query_text])
    sims = cosine_similarity(qv, X)[0]
    out = findings_df.copy()
    out["sim_score"] = sims
    out["year_norm"] = (out["year"] - out["year"].min()) / (out["year"].max() - out["year"].min()) if "year" in out.columns and out["year"].nunique() > 1 else 0.0
    out["sev_norm"] = out.get("severity", 3) / 5
    out["score"] = out["sim_score"]*0.65 + out["sev_norm"]*0.25 + out["year_norm"]*0.10
    cols = ["finding_id","year","unit","program","issue_title","issue_detail","cause_category","cause_detail","recommendation","outcomes_impact","severity","score", "sim_score"]
    return out.sort_values("score", ascending=False).head(top_k)[[c for c in cols if c in out.columns]]

def create_excel_template():
    df = pd.DataFrame(columns=["finding_id", "issue_title", "unit", "program", "year", "cause_category", "cause_detail", "issue_detail", "recommendation", "outcomes_impact", "severity"])
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

# ----------------- START: Custom CSS -----------------
st.markdown("""
<style>
body { font-family: 'Kanit', sans-serif; }
button[data-baseweb="tab"] {
    border: 1px solid #007bff; border-radius: 8px; padding: 10px 15px;
    margin: 5px 5px 5px 0px; transition: background-color 0.3s, color 0.3s;
    font-weight: bold; color: #007bff !important; background-color: #ffffff;
    box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1); border-bottom: none !important;
    &::after { content: none !important; }
}
button[data-baseweb="tab"][aria-selected="true"] { box-shadow: 2px 2px 5px rgba(0,0,0,0.2); }

/* Group 1: 1-5 (Blue) */
div[data-baseweb="tab-list"] button:nth-of-type(-n+5) { border-color: #007bff; color: #007bff !important; }
div[data-baseweb="tab-list"] button:nth-of-type(-n+5)[aria-selected="true"] { background-color: #007bff; color: white !important; }
/* Group 2: 6-7 (Purple) */
div[data-baseweb="tab-list"] button:nth-of-type(6), div[data-baseweb="tab-list"] button:nth-of-type(7) { border-color: #6f42c1; color: #6f42c1 !important; }
div[data-baseweb="tab-list"] button:nth-of-type(6)[aria-selected="true"], div[data-baseweb="tab-list"] button:nth-of-type(7)[aria-selected="true"] { background-color: #6f42c1; color: white !important; }
/* Group 3: 8 (Gold) */
div[data-baseweb="tab-list"] button:nth-of-type(8) { border-color: #ffc107; color: #cc9900 !important; }
div[data-baseweb="tab-list"] button:nth-of-type(8)[aria-selected="true"] { background-color: #ffc107; color: #333333 !important; }
/* Group 4: 9 (Green - NEW) */
div[data-baseweb="tab-list"] button:nth-of-type(9) { border-color: #28a745; color: #28a745 !important; }
div[data-baseweb="tab-list"] button:nth-of-type(9)[aria-selected="true"] { background-color: #28a745; color: white !important; }

div[data-baseweb="tab-list"] { border-bottom: none !important; margin-bottom: 15px; flex-wrap: wrap; gap: 10px; }
h4 { color: #007bff !important; border-bottom: 2px solid #e0e0e0; padding-bottom: 5px; }
</style>
""", unsafe_allow_html=True)
# ----------------- END: Custom CSS -----------------

# ----------------- Tab Definitions -----------------
tab_plan, tab_logic, tab_method, tab_kpi, tab_risk, tab_issue, tab_preview, tab_assist, tab_chatbot = st.tabs([
    "1. ระบุ แผน & 6W2H",
    "2. ระบุ Logic Model",
    "3. ระบุ Methods",
    "4. ระบุ KPIs",
    "5. ระบุ Risks",
    "6. ค้นหาข้อตรวจพบที่ผ่านมา",
    "7. สรุปข้อมูล (Preview)",
    "🤖 ให้ PA Assist ช่วยแนะนำ",
    "💬 PA Chat (ถาม-ตอบ)" # NEW TAB
])

# ----------------- Tab 1: ระบุ แผน & 6W2H -----------------
with tab_plan:
    st.subheader("ข้อมูลแผน (Plan) - กรุณาระบุข้อมูล")
    with st.container(border=True):
        c1, c2, c3 = st.columns([2,2,1])
        plan["plan_title"] = c1.text_input("ชื่อแผน/เรื่องที่จะตรวจ", plan["plan_title"])
        plan["program_name"] = c1.text_input("ชื่อโครงการ/แผนงาน", plan["program_name"])
        plan["objectives"] = c1.text_area("วัตถุประสงค์การตรวจ", plan["objectives"])
        plan["scope"] = c2.text_area("ขอบเขตการตรวจ", plan["scope"])
        plan["assumptions"] = c2.text_area("สมมุติฐาน/ข้อจำกัดข้อมูล", plan["assumptions"])
        c3.text_input("Plan ID", plan["plan_id"], disabled=True)
        plan["status"] = c3.selectbox("สถานะ", ["Draft","Published"], index=0)

    st.divider()
    st.subheader("สรุปเรื่องที่ตรวจสอบ (6W2H)")
    with st.container(border=True):
        st.markdown("##### 🚀 สร้าง 6W2H อัตโนมัติด้วย AI")
        uploaded_text = st.text_area("วางข้อความที่นี่เพื่อให้ AI ช่วยสรุป 6W2H:", height=200, key="uploaded_text")
        
        if st.button("🚀 สร้าง 6W2H จากข้อความ", type="primary", key="6w2h_button"):
            if not uploaded_text: st.error("กรุณาวางข้อความในช่องก่อน")
            elif not st.session_state.api_key_global: st.error("กรุณากรอก API Key ใน Sidebar ด้านซ้ายก่อน")
            else:
                with st.spinner("กำลังประมวลผล..."):
                    try:
                        user_prompt = f"จากข้อความนี้: --- {uploaded_text} --- กรุณาสรุปเป็น 6W2H (Who, Whom, What, Where, When, Why, How, How much) ในรูปแบบ key: value"
                        client = OpenAI(api_key=st.session_state.api_key_global, base_url="https://api.opentyphoon.ai/v1")
                        response = client.chat.completions.create(
                            model="typhoon-v2.1-12b-instruct",
                            messages=[{"role": "user", "content": user_prompt}],
                            temperature=0.7, max_tokens=1024, top_p=0.9
                        )
                        llm_output = response.choices[0].message.content
                        for line in llm_output.strip().split('\n'):
                            if ':' in line:
                                key, value = map(str.strip, line.split(':', 1))
                                normalized_key = key.lower().replace(' ', '_')
                                if normalized_key in plan: plan[normalized_key] = value
                        st.success("สร้าง 6W2H เรียบร้อยแล้ว! ข้อมูลถูกเติมในช่องด้านล่างแล้ว")
                        st.balloons()
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการเรียกใช้ AI: {e}")

    with st.container(border=True):
        st.markdown("##### ⭐กรุณาระบุข้อมูล เพื่อนำไปใช้ประมวลผล")
        cc1, cc2, cc3 = st.columns(3)
        plan["who"] = cc1.text_input("Who (ใคร)", value=plan["who"])
        plan["whom"] = cc1.text_input("Whom (เพื่อใคร)", value=plan["whom"])
        plan["what"] = cc1.text_input("What (ทำอะไร)", value=plan["what"])
        plan["where"] = cc1.text_input("Where (ที่ไหน)", value=plan["where"])
        plan["when"] = cc2.text_input("When (เมื่อใด)", value=plan["when"])
        plan["why"] = cc2.text_area("Why (ทำไม)", value=plan["why"])
        plan["how"] = cc3.text_area("How (อย่างไร)", value=plan["how"])
        plan["how_much"] = cc3.text_input("How much (เท่าไร)", value=plan["how_much"])

# ----------------- Tab 2: Logic Model -----------------
with tab_logic:
    st.subheader("ระบุข้อมูล Logic Model: Input → Activities → Output → Outcome → Impact")
    st.dataframe(logic_df, use_container_width=True, hide_index=True)
    with st.expander("➕ เพิ่มรายการใน Logic Model"):
        with st.form("logic_form"):
            c1, c2, c3 = st.columns(3)
            typ = c1.selectbox("ประเภท", ["Input","Activity","Output","Outcome","Impact"])
            desc = c1.text_input("คำอธิบาย/รายละเอียด")
            metric = c1.text_input("ตัวชี้วัด/metric")
            unit = c2.text_input("หน่วย")
            target = c2.text_input("เป้าหมาย")
            source = c3.text_input("แหล่งข้อมูล")
            if st.form_submit_button("เพิ่ม Logic Item", type="primary"):
                new_row = pd.DataFrame([{"item_id": next_id("LG", logic_df, "item_id"), "plan_id": plan["plan_id"], "type": typ, "description": desc, "metric": metric, "unit": unit, "target": target, "source": source}])
                st.session_state["logic_items"] = pd.concat([logic_df, new_row], ignore_index=True)
                st.rerun()

# ----------------- Remaining Tabs (3, 4, 5) - Condensed for brevity -----------------
with tab_method:
    st.subheader("ระบุวิธีการเก็บข้อมูล (Methods)")
    st.dataframe(methods_df, use_container_width=True, hide_index=True)
    # Add Method form...

with tab_kpi:
    st.subheader("ระบุตัวชี้วัด (KPIs)")
    st.dataframe(kpis_df, use_container_width=True, hide_index=True)
    # Add KPI form...

with tab_risk:
    st.subheader("ระบุความเสี่ยง (Risks)")
    st.dataframe(risks_df, use_container_width=True, hide_index=True)
    # Add Risk form...

# ----------------- Tab 6: ค้นหาข้อตรวจพบที่ผ่านมา -----------------
with tab_issue:
    st.subheader("🔎 แนะนำประเด็นตรวจจากรายงานเก่า (Issue Suggestions)")
    with st.container(border=True):
        st.download_button(label="⬇️ ดาวน์โหลดไฟล์แม่แบบ FindingsLibrary.xlsx", data=create_excel_template(), file_name="FindingsLibrary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        uploaded = st.file_uploader("อัปโหลด FindingsLibrary.csv หรือ .xlsx", type=["csv", "xlsx", "xls"])
    findings_df = load_findings(uploaded=uploaded)
    if findings_df.empty:
        st.info("ไม่พบข้อมูล Findings โปรดอัปโหลดไฟล์ หรือวางไฟล์ FindingsLibrary.csv ในโฟลเดอร์เดียวกัน")
    else:
        st.success(f"พบข้อมูล Findings ทั้งหมด {len(findings_df)} รายการ")
        vec, X = build_tfidf_index(findings_df)
        seed = f"Who:{plan.get('who','')} What:{plan.get('what','')} Outputs:{' | '.join(logic_df[logic_df['type']=='Output']['description'].tolist())} Outcomes:{' | '.join(logic_df[logic_df['type']=='Outcome']['description'].tolist())}"
        
        def refresh_query_text(new_seed):
            st.session_state["issue_query_text"] = new_seed
            st.session_state["ref_seed"] = new_seed

        if "issue_query_text" not in st.session_state or st.session_state.get("ref_seed") != seed and st.session_state.get("issue_query_text") == st.session_state.get("ref_seed"):
            refresh_query_text(seed)

        c_query_area, c_refresh_btn = st.columns([6, 1])
        query_text = c_query_area.text_area("**สรุปบริบทที่ใช้ค้นหา (แก้ไขได้):**", st.session_state["issue_query_text"], height=140, key="issue_query_text_area")
        c_refresh_btn.button("🔄", on_click=refresh_query_text, args=(seed,), help="อัปเดตช่องค้นหาด้วยข้อมูลล่าสุด")
        
        if st.button("ค้นหาประเด็นที่ใกล้เคียง", type="primary", key="search_button_fix"):
            results = search_candidates(query_text, findings_df, vec, X, top_k=8)
            st.session_state["issue_results"] = results
            st.success(f"พบประเด็นที่เกี่ยวข้อง {len(results)} รายการ")
            
        results = st.session_state.get("issue_results", pd.DataFrame())
        if not results.empty:
            st.subheader("ผลลัพธ์การค้นหา")
            for i, row in results.reset_index(drop=True).iterrows():
                with st.container(border=True):
                    st.markdown(f"**{row.get('issue_title', '(N/A)')}** (หน่วย: {row.get('unit', '-')}, ปี: {int(row.get('year', 0))})")
                    with st.expander("รายละเอียด/ข้อเสนอแนะ (เดิม)"):
                        st.write(row.get("issue_detail", "-"))
                        st.markdown(f"**คะแนนความเกี่ยวข้อง**: {row.get('score', 0):.3f}")
                    c1, c2 = st.columns([3,1])
                    rationale = c1.text_area("เหตุผลที่ควรตรวจ", key=f"rat_{i}", value=f"อ้างอิงกรณีเดิม ปี {int(row.get('year', 0))}")
                    if c2.button("➕ เพิ่มเข้าแผน", key=f"add_{i}"):
                        new_issue = pd.DataFrame([{"issue_id": next_id("ISS", audit_issues_df, "issue_id"), "plan_id": plan["plan_id"], "title": row.get('issue_title'), "rationale": rationale, "source_finding_id": row.get('finding_id'), "issue_detail": row.get('issue_detail'), "recommendation": row.get('recommendation')}])
                        st.session_state["audit_issues"] = pd.concat([audit_issues_df, new_issue], ignore_index=True)
                        st.success("เพิ่มประเด็นเข้าแผนแล้ว ✅")
                        st.rerun()
        st.markdown("### ประเด็นที่ถูกเพิ่มเข้าแผน")
        st.dataframe(st.session_state["audit_issues"], use_container_width=True, hide_index=True)

# ----------------- Tab 7: สรุปข้อมูล (Preview) -----------------
with tab_preview:
    st.subheader("สรุปแผน (Preview)")
    with st.container(border=True):
        st.markdown(f"**Plan ID:** {plan['plan_id']}  \n**ชื่อแผนงาน:** {plan['plan_title']}")
    st.markdown("### สรุป 6W2H")
    st.json(plan)
    # Display other dataframes and download links...
    
# ----------------- Tab 8: ให้ PA Assist ช่วยแนะนำประเด็นการตรวจสอบ -----------------
with tab_assist:
    st.subheader("💡 PA Audit Assist (ขับเคลื่อนด้วย LLM)")
    if st.button("🚀 สร้างคำแนะนำจาก AI", type="primary", key="llm_assist_button"):
        if not st.session_state.api_key_global:
            st.error("กรุณากรอก API Key ใน Sidebar ด้านซ้ายก่อน")
        else:
            with st.spinner("กำลังสร้างคำแนะนำ..."):
                try:
                    plan_summary = f"ข้อมูลแผน: {plan}\nLogic Model: {logic_df.to_string()}\nประเด็นจากรายงานเก่า: {audit_issues_df.to_string()}"
                    user_prompt = f"จากข้อมูลแผนการตรวจสอบนี้: --- {plan_summary} --- กรุณาช่วยสร้าง: 1. ประเด็นการตรวจสอบที่ควรให้ความสำคัญ, 2. ข้อตรวจพบที่คาดว่าจะพบ, 3. ร่างรายงานตรวจสอบที่จะเจอ. โดยใช้ Format: <ประเด็นการตรวจสอบ>[ข้อความ]</ประเด็นการตรวจสอบ>\n<ข้อตรวจพบ>[ข้อความ]</ข้อตรวจพบ>\n<ร่างรายงาน>[ข้อความ]</ร่างรายงาน>"
                    client = OpenAI(api_key=st.session_state.api_key_global, base_url="https://api.opentyphoon.ai/v1")
                    response = client.chat.completions.create(model="typhoon-v2.1-12b-instruct", messages=[{"role": "user", "content": user_prompt}], temperature=0.7, max_tokens=2048)
                    full_response = response.choices[0].message.content
                    st.session_state["gen_issues"] = full_response.split("<ประเด็นการตรวจสอบ>")[1].split("</ประเด็นการตรวจสอบ>")[0].strip()
                    st.session_state["gen_findings"] = full_response.split("<ข้อตรวจพบ>")[1].split("</ข้อตรวจพบ>")[0].strip()
                    st.session_state["gen_report"] = full_response.split("<ร่างรายงาน>")[1].split("</ร่างรายงาน>")[0].strip()
                    st.success("สร้างคำแนะนำจาก AI เรียบร้อยแล้ว ✅")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {e}")

    st.markdown("<h4>ประเด็นการตรวจสอบที่ควรให้ความสำคัญ</h4>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color:#f0f2f6;border:1px solid #ccc;padding:10px;border-radius:5px;height:200px;overflow-y:scroll;'>{st.session_state.get('gen_issues', '')}</div>", unsafe_allow_html=True)
    st.markdown("<h4>ข้อตรวจพบที่คาดว่าจะพบ</h4>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color:#f0f2f6;border:1px solid #ccc;padding:10px;border-radius:5px;height:200px;overflow-y:scroll;'>{st.session_state.get('gen_findings', '')}</div>", unsafe_allow_html=True)
    st.markdown("<h4>ร่างรายงานตรวจสอบ (Preview)</h4>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color:#f0f2f6;border:1px solid #ccc;padding:10px;border-radius:5px;height:400px;overflow-y:scroll;'>{st.session_state.get('gen_report', '')}</div>", unsafe_allow_html=True)


# ----------------- Tab 9: PA Chat (ถาม-ตอบ) - NEWLY ADDED -----------------
with tab_chatbot:
    st.subheader("💬 PA Chat - ผู้ช่วยอัจฉริยะ (RAG)")
    
    uploaded_files = st.file_uploader(
        "อัปโหลดเอกสารเพื่อเป็นบริบทให้ AI (PDF, TXT, CSV) - ข้อมูลจะถูกลบเมื่อปิดแอป",
        type=['pdf', 'txt', 'csv'],
        accept_multiple_files=True
    )
    
    current_uploaded_file_names = {f.name for f in uploaded_files}
    if uploaded_files and st.session_state.get('last_uploaded_files') != current_uploaded_file_names:
        with st.spinner("กำลังประมวลผลเอกสาร..."):
            st.session_state.doc_context_uploaded, _ = process_documents(uploaded_files, 'uploaded', MAX_CHARS_LIMIT, len(st.session_state.doc_context_local))
            st.session_state.last_uploaded_files = current_uploaded_file_names
            st.session_state.chatbot_messages = [{"role": "assistant", "content": "อัปเดตเอกสารใหม่เรียบร้อยแล้ว ผมพร้อมตอบคำถามโดยอ้างอิงจากเอกสารล่าสุดครับ"}]
            st.rerun()
    elif not uploaded_files and st.session_state.doc_context_uploaded:
        st.session_state.doc_context_uploaded = ""
        st.session_state.last_uploaded_files = set()
        st.session_state.chatbot_messages.append({"role": "assistant", "content": "ได้ทำการล้างเอกสารที่อัปโหลดออกแล้วครับ"})
        st.rerun()

    doc_context_len = len(st.session_state.doc_context_local) + len(st.session_state.doc_context_uploaded)
    st.info(f"💾 โหลดบริบทเอกสารรวม: {doc_context_len:,} ตัวอักษร (จำกัดสูงสุด: {MAX_CHARS_LIMIT:,})")

    chat_container = st.container(height=500, border=True)
    with chat_container:
        for message in st.session_state.chatbot_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่...", key="chat_input_main"):
        st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
        
        with chat_container: # Re-draw the user message inside the container immediately
             with st.chat_message("user"):
                st.markdown(prompt)

        with chat_container: # Stream the assistant's response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                if not st.session_state.api_key_global:
                    error_message = "เกิดข้อผิดพลาด: ไม่พบ API Key กรุณากรอก API Key ใน Sidebar ด้านซ้ายก่อน"
                    st.error(error_message)
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": error_message})
                    message_placeholder.markdown(error_message)
                else:
                    try:
                        doc_context = st.session_state.doc_context_local + st.session_state.doc_context_uploaded
                        system_prompt = f"คุณคือผู้ช่วย AI ด้าน Performance Audit จงใช้ข้อมูลจาก 'บริบทจากเอกสารภายใน' เป็นหลักในการตอบ: --- บริบทจากเอกสารภายใน:{doc_context} ---"
                        
                        messages_for_api = [{"role": "system", "content": system_prompt}] + st.session_state.chatbot_messages[-10:]

                        client = OpenAI(api_key=st.session_state.api_key_global, base_url="https://api.opentyphoon.ai/v1")
                        response_stream = client.chat.completions.create(
                            model="typhoon-v2.1-12b-instruct",
                            messages=messages_for_api,
                            temperature=0.5,
                            max_tokens=3072,
                            stream=True
                        )
                        response = message_placeholder.write_stream(response_stream)
                        st.session_state.chatbot_messages.append({"role": "assistant", "content": response})

                    except Exception as e:
                        error_message = f"เกิดข้อผิดพลาดขณะเชื่อมต่อ API: {e}"
                        st.error(error_message)
                        st.session_state.chatbot_messages.append({"role": "assistant", "content": error_message})
                        message_placeholder.markdown(error_message)
# ----------------- END: Tab 9 -----------------
