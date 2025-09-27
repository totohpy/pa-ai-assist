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

# กำหนดข้อจำกัดสำหรับ RAG
MAX_CHARS_LIMIT = 100000 # กำหนดการจำกัดที่ 100,000 ตัวอักษร

# ตั้งค่าหน้าเพจ
st.set_page_config(page_title="Planning Studio (+ Issue Suggestions)", page_icon="🧭", layout="wide")

# ----------------- Utility Functions สำหรับ RAG Chatbot -----------------

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

def load_rag_context(uploaded_files_dict):
    """Loads text from uploaded files and stores in session state."""
    full_context = ""
    
    for file_name, file_data in uploaded_files_dict.items():
        if file_data["type"] == "pdf" or file_data["type"] == "txt":
            # file_data["content"] is already text, apply limit here for safety although it's applied during upload
            full_context += f"*** เอกสาร: {file_name} ***\n{file_data['content'][:MAX_CHARS_LIMIT]}\n\n"
    
    st.session_state["rag_docs_context"] = full_context

    if len(full_context) > 0:
        st.success(f"โหลดบริบทจากเอกสารแล้ว (ความยาว: {len(full_context)} อักขระ)")
    else:
        st.info("ยังไม่มีเอกสารที่โหลดเข้าสู่บริบท RAG")

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
    ss.setdefault("logic_items", pd.DataFrame(columns=["item_id","plan_id","type","description","metric","unit","target","source"]))
    ss.setdefault("methods", pd.DataFrame(columns=["method_id","plan_id","type","tool_ref","sampling","questions","linked_issue","data_source","frequency"]))
    ss.setdefault("kpis", pd.DataFrame(columns=["kpi_id","plan_id","level","name","formula","numerator","denominator","unit","baseline","target","frequency","data_source","quality_requirements"]))
    ss.setdefault("risks", pd.DataFrame(columns=["risk_id","plan_id","description","category","likelihood","impact","mitigation","hypothesis"]))
    ss.setdefault("audit_issues", pd.DataFrame(columns=["issue_id","plan_id","title","rationale","linked_kpi","proposed_methods","source_finding_id","issue_detail", "recommendation"]))
    ss.setdefault("gen_issues", "")
    ss.setdefault("gen_findings", "")
    ss.setdefault("gen_report", "")
    ss.setdefault("issue_results", pd.DataFrame())
    # เพิ่ม state สำหรับเก็บค่า Seed อ้างอิงและข้อความค้นหา
    ss.setdefault("ref_seed", "") 
    ss.setdefault("issue_query_text", "")
    # *** เพิ่ม state สำหรับ Chatbot ***
    ss.setdefault("chatbot_messages", [])
    ss.setdefault("rag_docs_context", "")
    ss.setdefault("uploaded_files_content", {}) # Stores filename: {"type": "pdf"|"txt", "content": "...", "size": 1234}

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

    # 1. Try to load the pre-existing database file
    findings_db_path = "FindingsLibrary.csv"
    if os.path.exists(findings_db_path):
        try:
            findings_df = pd.read_csv(findings_db_path)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ FindingsLibrary.csv: {e}")
            findings_df = pd.DataFrame()

    # 2. If a new file is uploaded, combine it with the existing data
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

    # 3. Clean and return the combined dataframe
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
    
# Function to create an empty Excel template
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

# ----------------- Tab Definitions -----------------
# **เพิ่ม tab_chatbot กลับเข้ามา**
tab_plan, tab_logic, tab_method, tab_kpi, tab_risk, tab_issue, tab_preview, tab_assist, tab_chatbot = st.tabs([
    "1. ระบุ แผน & 6W2H", 
    "2. ระบุ Logic Model", 
    "3. ระบุ Methods", 
    "4. ระบุ KPIs", 
    "5. ระบุ Risks", 
    "6. ค้นหาข้อตรวจพบที่ผ่านมา", 
    "7. สรุปข้อมูล (Preview)", 
    "8. ให้ PA Assist ช่วยแนะนำประเด็นการตรวจสอบ ✨✨",
    "9. RAG Chatbot 💬" # New tab
]) 

# ----------------- Tab 1: ระบุ แผน & 6W2H -----------------
with tab_plan:
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

    with st.container(border=True):
        st.markdown("##### 🚀 สร้าง 6W2H อัตโนมัติด้วย AI")
        st.write("คัดลอกข้อความจากไฟล์ของคุณแล้วนำมาวางในช่องด้านล่างนี้")
        uploaded_text = st.text_area("ระบุข้อความเกี่ยวกับเรื่องที่จะตรวจสอบ ที่ต้องการให้ AI ช่วยสรุป 6W2H", height=200, key="uploaded_text")
        st.markdown("💡 **ยังไม่มี API Key?** คลิก [ที่นี่](https://playground.opentyphoon.ai/settings/api-key) เพื่อรับ key ฟรี!")
        api_key_6w2h = st.text_input("กรุณากรอก API Key เพื่อใช้บริการ AI:", type="password", key="api_key_6w2h")

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
                        error_type = type(e).__name__
                        if error_type in ["AuthenticationError", "RateLimitError"]:
                            st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ API: ({error_type}) โปรดตรวจสอบ API Key หรือขีดจำกัดการใช้งาน (Rate Limit) ของคุณ\nรายละเอียด: {e}")
                        else:
                            st.error(f"เกิดข้อผิดพลาดขณะทำงาน: ({error_type}) โปรดลองอีกครั้ง\nรายละเอียด: {e}")
            
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
    st.subheader("รายการ Logic Model / Theory of Change")
    # ... (Code for Tab 2 content remains the same)
    
    # Logic Item Addition Form
    with st.form("logic_form", clear_on_submit=True, border=True):
        st.markdown("##### เพิ่มรายการ Logic Model")
        c1, c2 = st.columns([1,3])
        with c1:
            item_type = st.selectbox("ประเภท", ["Input","Activity","Output","Outcome","Impact"], key="logic_type")
        with c2:
            description = st.text_input("รายละเอียด", key="logic_desc")
        
        c3, c4, c5 = st.columns(3)
        with c3:
            metric = st.text_input("ตัวชี้วัด", key="logic_metric")
        with c4:
            unit = st.text_input("หน่วย", key="logic_unit")
        with c5:
            target = st.text_input("เป้าหมาย", key="logic_target")
        
        source = st.text_input("แหล่งอ้างอิง", key="logic_source")
        
        submitted = st.form_submit_button("➕ เพิ่มรายการ")
        if submitted and description:
            new_id = next_id(item_type[0].upper(), logic_df, "item_id")
            new_row = {
                "item_id": new_id,
                "plan_id": plan["plan_id"],
                "type": item_type,
                "description": description,
                "metric": metric,
                "unit": unit,
                "target": target,
                "source": source
            }
            st.session_state["logic_items"] = pd.concat([logic_df, pd.DataFrame([new_row])], ignore_index=True)
            st.success(f"เพิ่มรายการ {item_type} ID: {new_id} เรียบร้อย")
        elif submitted and not description:
            st.error("กรุณากรอกรายละเอียด")

    st.divider()

    # Logic Item Display and Editor
    st.markdown("##### ตารางรายการ Logic Model")
    if logic_df.empty:
        st.info("ยังไม่มีรายการ Logic Model")
    else:
        # Re-order and display the dataframe
        display_df = logic_df[["item_id", "type", "description", "metric", "unit", "target", "source"]]
        
        # In-line editor
        edited_df = st.data_editor(
            display_df,
            column_config={
                "item_id": st.column_config.TextColumn("ID", disabled=True),
                "type": st.column_config.SelectboxColumn("ประเภท", options=["Input","Activity","Output","Outcome","Impact"]),
                "description": st.column_config.TextColumn("รายละเอียด", required=True),
                "metric": st.column_config.TextColumn("ตัวชี้วัด"),
                "unit": st.column_config.TextColumn("หน่วย"),
                "target": st.column_config.TextColumn("เป้าหมาย"),
                "source": st.column_config.TextColumn("แหล่งอ้างอิง")
            },
            hide_index=True,
            num_rows="dynamic"
        )
        
        # Save button for edited data
        if st.button("💾 บันทึกการแก้ไข Logic Model"):
            # Ensure required column is present before merge
            if "plan_id" not in edited_df.columns:
                 edited_df["plan_id"] = plan["plan_id"]
            if "item_id" not in edited_df.columns:
                 edited_df["item_id"] = logic_df["item_id"] # Re-attach IDs if they somehow disappeared
            
            # Merge back with the original columns (like plan_id)
            final_df = pd.merge(logic_df.drop(columns=["type", "description", "metric", "unit", "target", "source"], errors='ignore'),
                                edited_df,
                                on="item_id",
                                how="right",
                                suffixes=('_old', ''))

            # Re-index the IDs for new rows added via dynamic rows
            for index, row in final_df.iterrows():
                if not pd.isna(row["type"]) and not str(row["item_id"]).startswith(row["type"][0].upper()):
                    final_df.loc[index, "item_id"] = next_id(row["type"][0].upper(), final_df, "item_id")
                    
            st.session_state["logic_items"] = final_df.dropna(subset=["description"]).reset_index(drop=True)
            st.success("บันทึก Logic Model เรียบร้อยแล้ว")

# ----------------- Tab 3: ระบุ Methods -----------------
with tab_method:
    st.subheader("รายการ วิธีการตรวจสอบ (Audit Methods)")
    # ... (Code for Tab 3 content remains the same)
    
    # Method Addition Form
    with st.form("method_form", clear_on_submit=True, border=True):
        st.markdown("##### เพิ่มรายการวิธีการตรวจสอบ")
        c1, c2 = st.columns([1,3])
        with c1:
            method_type = st.selectbox("ประเภท", ["Observation","Interview","Survey","Document Review","Data Analysis","Simulation/Test"], key="method_type")
        with c2:
            tool_ref = st.text_input("เครื่องมือ/อ้างอิง (เช่น แบบสอบถาม, รหัสโค้ด)", key="method_tool")
            
        questions = st.text_area("คำถาม/ขั้นตอนสำคัญ", key="method_questions", height=100)
        
        c3, c4, c5 = st.columns(3)
        with c3:
            sampling = st.text_input("วิธีการสุ่มตัวอย่าง", key="method_sampling")
        with c4:
            data_source = st.text_input("แหล่งข้อมูล", key="method_source")
        with c5:
            frequency = st.text_input("ความถี่/ระยะเวลา", key="method_frequency")
        
        linked_issue = st.multiselect(
            "เชื่อมโยงกับประเด็นตรวจสอบ (Issue)", 
            options=audit_issues_df["issue_id"].tolist(),
            key="method_linked_issue"
        )
        
        submitted = st.form_submit_button("➕ เพิ่มรายการ")
        if submitted and questions:
            new_id = next_id("M", methods_df, "method_id")
            new_row = {
                "method_id": new_id,
                "plan_id": plan["plan_id"],
                "type": method_type,
                "tool_ref": tool_ref,
                "sampling": sampling,
                "questions": questions,
                "linked_issue": ", ".join(linked_issue),
                "data_source": data_source,
                "frequency": frequency
            }
            st.session_state["methods"] = pd.concat([methods_df, pd.DataFrame([new_row])], ignore_index=True)
            st.success(f"เพิ่มรายการ Method ID: {new_id} เรียบร้อย")
        elif submitted and not questions:
            st.error("กรุณากรอกคำถาม/ขั้นตอนสำคัญ")
    
    st.divider()

    # Method Display and Editor
    st.markdown("##### ตารางวิธีการตรวจสอบ")
    if methods_df.empty:
        st.info("ยังไม่มีรายการวิธีการตรวจสอบ")
    else:
        display_df = methods_df[["method_id", "type", "tool_ref", "questions", "sampling", "data_source", "frequency", "linked_issue"]]
        edited_df = st.data_editor(
            display_df,
            column_config={
                "method_id": st.column_config.TextColumn("ID", disabled=True),
                "type": st.column_config.SelectboxColumn("ประเภท", options=["Observation","Interview","Survey","Document Review","Data Analysis","Simulation/Test"]),
                "tool_ref": st.column_config.TextColumn("เครื่องมือ/อ้างอิง"),
                "questions": st.column_config.TextColumn("คำถาม/ขั้นตอนสำคัญ", required=True),
                "sampling": st.column_config.TextColumn("วิธีการสุ่มตัวอย่าง"),
                "data_source": st.column_config.TextColumn("แหล่งข้อมูล"),
                "frequency": st.column_config.TextColumn("ความถี่/ระยะเวลา"),
                "linked_issue": st.column_config.TextColumn("เชื่อมโยงกับ Issue ID")
            },
            hide_index=True,
            num_rows="dynamic"
        )
        
        if st.button("💾 บันทึกการแก้ไข Methods"):
             # Simple merge back to preserve original plan_id
            final_df = pd.merge(methods_df.drop(columns=["type", "tool_ref", "questions", "sampling", "linked_issue", "data_source", "frequency"], errors='ignore'),
                                edited_df,
                                on="method_id",
                                how="right",
                                suffixes=('_old', ''))

            st.session_state["methods"] = final_df.dropna(subset=["questions"]).reset_index(drop=True)
            st.success("บันทึก Methods เรียบร้อยแล้ว")

# ----------------- Tab 4: ระบุ KPIs -----------------
with tab_kpi:
    st.subheader("รายการตัวชี้วัด (Key Performance Indicators - KPIs)")
    # ... (Code for Tab 4 content remains the same)
    
    # KPI Addition Form
    with st.form("kpi_form", clear_on_submit=True, border=True):
        st.markdown("##### เพิ่มรายการตัวชี้วัด")
        c1, c2 = st.columns([1,3])
        with c1:
            level = st.selectbox("ระดับ", ["Process","Output","Outcome","Impact"], key="kpi_level")
        with c2:
            # *** แก้ไข: ลบ required=True ออกจาก st.text_input ***
            name = st.text_input("ชื่อตัวชี้วัด", key="kpi_name") 
            
        c3, c4 = st.columns(2)
        with c3:
            numerator = st.text_input("ตัวเศษ (Numerator)", key="kpi_numerator")
            denominator = st.text_input("ตัวส่วน (Denominator)", key="kpi_denominator")
            formula = st.text_input("สูตรคำนวณ", value=f"({numerator}) / ({denominator})" if numerator and denominator else "", key="kpi_formula")
        with c4:
            unit = st.text_input("หน่วย", key="kpi_unit")
            baseline = st.text_input("ค่าฐาน (Baseline)", key="kpi_baseline")
            target = st.text_input("ค่าเป้าหมาย (Target)", key="kpi_target")
            
        c5, c6 = st.columns(2)
        with c5:
            frequency = st.text_input("ความถี่ในการวัด", key="kpi_frequency")
        with c6:
            data_source = st.text_input("แหล่งข้อมูล", key="kpi_data_source")

        quality_requirements = st.text_area("ข้อกำหนดด้านคุณภาพข้อมูล", key="kpi_quality", height=80)
        
        submitted = st.form_submit_button("➕ เพิ่มรายการ")
        if submitted and name:
            new_id = next_id("K", kpis_df, "kpi_id")
            new_row = {
                "kpi_id": new_id,
                "plan_id": plan["plan_id"],
                "level": level,
                "name": name,
                "formula": formula,
                "numerator": numerator,
                "denominator": denominator,
                "unit": unit,
                "baseline": baseline,
                "target": target,
                "frequency": frequency,
                "data_source": data_source,
                "quality_requirements": quality_requirements
            }
            st.session_state["kpis"] = pd.concat([kpis_df, pd.DataFrame([new_row])], ignore_index=True)
            st.success(f"เพิ่มรายการ KPI ID: {new_id} เรียบร้อย")
        elif submitted and not name:
            st.error("กรุณากรอกชื่อตัวชี้วัด")

    st.divider()

    # KPI Display and Editor
    st.markdown("##### ตารางตัวชี้วัด")
    if kpis_df.empty:
        st.info("ยังไม่มีรายการตัวชี้วัด")
    else:
        # Simplified display columns
        display_cols = ["kpi_id", "level", "name", "formula", "unit", "target", "data_source"]
        display_df = kpis_df[[c for c in display_cols if c in kpis_df.columns]].copy()
        
        edited_df = st.data_editor(
            display_df,
            column_config={
                "kpi_id": st.column_config.TextColumn("ID", disabled=True),
                "level": st.column_config.SelectboxColumn("ระดับ", options=["Process","Output","Outcome","Impact"]),
                "name": st.column_config.TextColumn("ชื่อตัวชี้วัด", required=True),
                "formula": st.column_config.TextColumn("สูตรคำนวณ"),
                "unit": st.column_config.TextColumn("หน่วย"),
                "target": st.column_config.TextColumn("เป้าหมาย"),
                "data_source": st.column_config.TextColumn("แหล่งข้อมูล")
            },
            hide_index=True,
            num_rows="dynamic"
        )
        
        if st.button("💾 บันทึกการแก้ไข KPIs"):
            # Ensure all original columns are preserved during merge
            kpi_cols_to_drop = [c for c in kpis_df.columns if c not in edited_df.columns]
            final_df = pd.merge(kpis_df.drop(columns=kpi_cols_to_drop, errors='ignore'),
                                edited_df,
                                on="kpi_id",
                                how="right",
                                suffixes=('_old', ''))

            # Re-index the IDs for new rows added via dynamic rows if necessary
            # (Assuming new rows won't have kpi_id, so they will be NaN or empty string)
            st.session_state["kpis"] = final_df.dropna(subset=["name"]).reset_index(drop=True)
            st.success("บันทึก KPIs เรียบร้อยแล้ว")

# ----------------- Tab 5: ระบุ Risks -----------------
with tab_risk:
    st.subheader("รายการความเสี่ยง (Audit Risks)")
    # ... (Code for Tab 5 content remains the same)
    
    # Risk Addition Form
    with st.form("risk_form", clear_on_submit=True, border=True):
        st.markdown("##### เพิ่มรายการความเสี่ยง")
        description = st.text_area("รายละเอียดความเสี่ยง/ปัญหาที่อาจเกิดขึ้น", key="risk_desc", height=80, required=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            category = st.text_input("ประเภทความเสี่ยง", key="risk_cat")
        with c2:
            likelihood = st.slider("โอกาสเกิด (Likelihood)", 1, 5, 3, key="risk_like")
        with c3:
            impact = st.slider("ผลกระทบ (Impact)", 1, 5, 3, key="risk_impact")
            
        mitigation = st.text_area("มาตรการ/แนวทางลดความเสี่ยง", key="risk_mitigation", height=80)
        hypothesis = st.text_area("สมมุติฐานการตรวจสอบ (Audit Hypothesis)", key="risk_hypothesis", height=80)
        
        submitted = st.form_submit_button("➕ เพิ่มรายการ")
        if submitted and description:
            new_id = next_id("R", risks_df, "risk_id")
            new_row = {
                "risk_id": new_id,
                "plan_id": plan["plan_id"],
                "description": description,
                "category": category,
                "likelihood": likelihood,
                "impact": impact,
                "mitigation": mitigation,
                "hypothesis": hypothesis
            }
            st.session_state["risks"] = pd.concat([risks_df, pd.DataFrame([new_row])], ignore_index=True)
            st.success(f"เพิ่มรายการ Risk ID: {new_id} เรียบร้อย")
        elif submitted and not description:
            st.error("กรุณากรอกรายละเอียดความเสี่ยง")

    st.divider()

    # Risk Display and Editor
    st.markdown("##### ตารางความเสี่ยง")
    if risks_df.empty:
        st.info("ยังไม่มีรายการความเสี่ยง")
    else:
        # Calculate Risk Score (L x I)
        display_df = risks_df.copy()
        display_df["Score"] = display_df["likelihood"] * display_df["impact"]
        display_df = display_df[["risk_id", "description", "category", "likelihood", "impact", "Score", "mitigation", "hypothesis"]]
        
        edited_df = st.data_editor(
            display_df,
            column_config={
                "risk_id": st.column_config.TextColumn("ID", disabled=True),
                "description": st.column_config.TextColumn("รายละเอียดความเสี่ยง", required=True),
                "category": st.column_config.TextColumn("ประเภท"),
                "likelihood": st.column_config.NumberColumn("โอกาสเกิด (1-5)", min_value=1, max_value=5, format="%d"),
                "impact": st.column_config.NumberColumn("ผลกระทบ (1-5)", min_value=1, max_value=5, format="%d"),
                "Score": st.column_config.NumberColumn("คะแนนความเสี่ยง", disabled=True),
                "mitigation": st.column_config.TextColumn("มาตรการลดความเสี่ยง"),
                "hypothesis": st.column_config.TextColumn("สมมุติฐานการตรวจสอบ"),
            },
            hide_index=True,
            num_rows="dynamic"
        )
        
        if st.button("💾 บันทึกการแก้ไข Risks"):
             # Simple merge back to preserve original plan_id
            risks_cols_to_drop = [c for c in risks_df.columns if c not in edited_df.columns]
            final_df = pd.merge(risks_df.drop(columns=risks_cols_to_drop, errors='ignore'),
                                edited_df,
                                on="risk_id",
                                how="right",
                                suffixes=('_old', ''))
            
            # Re-index the IDs for new rows added via dynamic rows if necessary
            st.session_state["risks"] = final_df.dropna(subset=["description"]).drop(columns=["Score"], errors='ignore').reset_index(drop=True)
            st.success("บันทึก Risks เรียบร้อยแล้ว")

# ----------------- Tab 6: ค้นหาข้อตรวจพบที่ผ่านมา -----------------
with tab_issue:
    st.subheader("ค้นหาข้อตรวจพบที่ผ่านมาเพื่อเป็นแนวทาง (Prior Findings Search)")

    with st.container(border=True):
        st.markdown("##### ฐานข้อมูลข้อตรวจพบ")
        
        findings_df = load_findings()
        
        if not findings_df.empty:
            st.info(f"ฐานข้อมูลปัจจุบันมีจำนวน {len(findings_df)} รายการ")
            
            c1, c2 = st.columns([1,1])
            with c1:
                uploaded_file = st.file_uploader("อัปโหลดไฟล์ CSV/Excel (ชีตชื่อ 'Data') เพื่อเพิ่มฐานข้อมูล", type=["csv", "xlsx", "xls"])
                if uploaded_file:
                    findings_df = load_findings(uploaded=uploaded_file) # Re-load with the new file
            with c2:
                template_data = create_excel_template()
                st.download_button(
                    label="⬇️ ดาวน์โหลด Template Excel",
                    data=template_data,
                    file_name="PriorFindingsTemplate.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            # Search input
            st.markdown("##### 🔍 ค้นหาข้อตรวจพบ")
            issue_query = st.text_area(
                "ใส่คำค้นหา (เช่น ชื่อโครงการ, ประเด็นที่สนใจ, ปัญหาที่คาดว่าจะพบ)", 
                height=100, 
                value=st.session_state.get("issue_query_text", "")
            )
            st.session_state["issue_query_text"] = issue_query # Update state for persistence

            if st.button("🔎 เริ่มค้นหาข้อตรวจพบ", type="primary", use_container_width=True):
                if issue_query and not findings_df.empty:
                    try:
                        vec, X = build_tfidf_index(findings_df)
                        results_df = search_candidates(issue_query, findings_df, vec, X, top_k=10)
                        st.session_state["issue_results"] = results_df
                        st.success("ค้นหาเสร็จสิ้น")
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการค้นหา: {e}")
                else:
                    st.warning("กรุณากรอกคำค้นหา และตรวจสอบว่ามีฐานข้อมูลข้อตรวจพบแล้ว")
        else:
            st.error("ไม่พบฐานข้อมูลข้อตรวจพบ กรุณาอัปโหลดไฟล์")

    st.divider()

    st.markdown("##### ผลลัพธ์การค้นหา")
    results_df = st.session_state.get("issue_results", pd.DataFrame())

    if not results_df.empty:
        # Display search results
        st.dataframe(
            results_df,
            column_config={
                "finding_id": "ID",
                "year": "ปี",
                "unit": "หน่วยงาน",
                "program": "โครงการ/แผนงาน",
                "issue_title": "ชื่อข้อตรวจพบ",
                "issue_detail": "รายละเอียดปัญหา",
                "cause_category": "ประเภทสาเหตุ",
                "cause_detail": "รายละเอียดสาเหตุ",
                "recommendation": "ข้อเสนอแนะ",
                "outcomes_impact": "ผลลัพธ์/ผลกระทบ",
                "severity": "ระดับความรุนแรง (1-5)",
                "score": st.column_config.NumberColumn("คะแนนรวม", format="%.3f"),
                "sim_score": st.column_config.NumberColumn("คะแนนความคล้าย", format="%.3f"),
            },
            hide_index=True,
            use_container_width=True
        )

        st.markdown("##### 📌 สร้างประเด็นตรวจสอบ (Issue) จากข้อตรวจพบที่พบ")
        
        # User selection to create a new issue
        selected_id = st.selectbox("เลือก ID ข้อตรวจพบที่ต้องการนำไปสร้าง Issue", options=[""] + results_df["finding_id"].tolist())
        
        if selected_id:
            selected_finding = results_df[results_df["finding_id"] == selected_id].iloc[0]
            
            with st.form("issue_creation_form", clear_on_submit=False, border=True):
                st.write(f"**นำข้อมูลจาก ID: {selected_id}**")
                
                title = st.text_input("ชื่อประเด็นตรวจสอบ (Issue Title)", value=selected_finding["issue_title"], key="new_issue_title")
                rationale = st.text_area("เหตุผล/ความเป็นมา", value=f"อ้างอิงจากข้อตรวจพบ ID: {selected_id} ของหน่วยงาน {selected_finding['unit']} ในโครงการ {selected_finding['program']} ปี {selected_finding['year']} ที่พบว่า {selected_finding['issue_title']}", key="new_issue_rationale")
                issue_detail = st.text_area("รายละเอียดปัญหา (นำมาจากข้อตรวจพบ)", value=selected_finding.get("issue_detail", ""), key="new_issue_detail")
                recommendation = st.text_area("ข้อเสนอแนะเบื้องต้น (นำมาจากข้อตรวจพบ)", value=selected_finding.get("recommendation", ""), key="new_issue_recommendation")

                linked_kpi = st.multiselect(
                    "เชื่อมโยงกับ KPI ID", 
                    options=kpis_df["kpi_id"].tolist(),
                    key="new_issue_linked_kpi"
                )
                
                # Automatically suggest methods based on current plan methods
                all_methods = methods_df["method_id"].tolist()
                proposed_methods = st.multiselect(
                    "วิธีการตรวจสอบที่เสนอ", 
                    options=all_methods,
                    default=all_methods, # Default to all current methods for suggestion
                    key="new_issue_proposed_methods"
                )

                issue_submitted = st.form_submit_button("➕ สร้างประเด็นตรวจสอบ (Issue)")
                
                if issue_submitted and title:
                    new_id = next_id("I", audit_issues_df, "issue_id")
                    new_issue_row = {
                        "issue_id": new_id,
                        "plan_id": plan["plan_id"],
                        "title": title,
                        "rationale": rationale,
                        "linked_kpi": ", ".join(linked_kpi),
                        "proposed_methods": ", ".join(proposed_methods),
                        "source_finding_id": selected_id,
                        "issue_detail": issue_detail,
                        "recommendation": recommendation
                    }
                    st.session_state["audit_issues"] = pd.concat([audit_issues_df, pd.DataFrame([new_issue_row])], ignore_index=True)
                    st.success(f"สร้างประเด็นตรวจสอบ Issue ID: {new_id} เรียบร้อยแล้ว")
                elif issue_submitted and not title:
                    st.error("กรุณากรอกชื่อประเด็นตรวจสอบ")
    else:
        st.info("ยังไม่มีผลลัพธ์การค้นหา")
        
    st.divider()

    st.markdown("##### ตารางประเด็นตรวจสอบ (Audit Issues)")
    # Issue Display and Editor
    if audit_issues_df.empty:
        st.info("ยังไม่มีประเด็นตรวจสอบที่ถูกบันทึก")
    else:
        display_df = audit_issues_df[["issue_id", "title", "rationale", "linked_kpi", "proposed_methods", "source_finding_id"]]
        
        edited_df = st.data_editor(
            display_df,
            column_config={
                "issue_id": st.column_config.TextColumn("ID", disabled=True),
                "title": st.column_config.TextColumn("ชื่อประเด็นตรวจสอบ", required=True),
                "rationale": st.column_config.TextColumn("เหตุผล/ความเป็นมา"),
                "linked_kpi": st.column_config.TextColumn("เชื่อมโยง KPI ID"),
                "proposed_methods": st.column_config.TextColumn("วิธีการตรวจสอบที่เสนอ"),
                "source_finding_id": st.column_config.TextColumn("อ้างอิงข้อตรวจพบเดิม"),
            },
            hide_index=True,
            num_rows="dynamic"
        )
        
        if st.button("💾 บันทึกการแก้ไข Issues"):
            issues_cols_to_drop = [c for c in audit_issues_df.columns if c not in edited_df.columns]
            final_df = pd.merge(audit_issues_df.drop(columns=issues_cols_to_drop, errors='ignore'),
                                edited_df,
                                on="issue_id",
                                how="right",
                                suffixes=('_old', ''))

            st.session_state["audit_issues"] = final_df.dropna(subset=["title"]).reset_index(drop=True)
            st.success("บันทึก Issues เรียบร้อยแล้ว")
            
# ----------------- Tab 7: สรุปข้อมูล (Preview) -----------------
with tab_preview:
    st.subheader("สรุปภาพรวมแผนการตรวจสอบ")
    
    st.markdown("#### 1. ข้อมูลแผนและ 6W2H")
    st.dataframe(pd.DataFrame(plan, index=["Detail"]).T, use_container_width=True)
    
    st.markdown("#### 2. Logic Model")
    if not logic_df.empty:
        st.dataframe(logic_df.drop(columns=["plan_id"], errors='ignore'), use_container_width=True)
    else: st.info("ไม่มีข้อมูล Logic Model")
    
    st.markdown("#### 3. KPIs")
    if not kpis_df.empty:
        st.dataframe(kpis_df.drop(columns=["plan_id", "numerator", "denominator", "baseline", "quality_requirements"], errors='ignore'), use_container_width=True)
    else: st.info("ไม่มีข้อมูล KPI")
    
    st.markdown("#### 4. Risks & Hypothesis")
    if not risks_df.empty:
        display_risks = risks_df.copy()
        display_risks["Score"] = display_risks["likelihood"] * display_risks["impact"]
        st.dataframe(display_risks.drop(columns=["plan_id"], errors='ignore'), use_container_width=True)
    else: st.info("ไม่มีข้อมูล Risks")
    
    st.markdown("#### 5. Audit Issues")
    if not audit_issues_df.empty:
        st.dataframe(audit_issues_df.drop(columns=["plan_id", "issue_detail", "recommendation"], errors='ignore'), use_container_width=True)
    else: st.info("ไม่มีข้อมูล Issues")
    
    st.markdown("#### 6. Audit Methods")
    if not methods_df.empty:
        st.dataframe(methods_df.drop(columns=["plan_id"], errors='ignore'), use_container_width=True)
    else: st.info("ไม่มีข้อมูล Methods")

    st.divider()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        df_download_link(st.session_state["logic_items"], "logic_model.csv", "⬇️ ดาวน์โหลด Logic Model (CSV)")
    with c2:
        df_download_link(st.session_state["kpis"], "kpis.csv", "⬇️ ดาวน์โหลด KPIs (CSV)")
    with c3:
        df_download_link(st.session_state["audit_issues"], "audit_issues.csv", "⬇️ ดาวน์โหลด Issues (CSV)")

# ----------------- Tab 8: ให้ PA Assist ช่วยแนะนำประเด็นการตรวจสอบ -----------------
with tab_assist:
    st.subheader("🤖 PA Assist - ช่วยสร้างประเด็นและข้อตรวจพบที่คาดว่าจะพบ")
    
    with st.container(border=True):
        st.markdown("##### ข้อมูลอ้างอิง (Context) สำหรับ AI")
        
        # Combine all relevant plan data into a single context string
        context_parts = []
        context_parts.append(f"Plan ID: {plan['plan_id']}")
        context_parts.append(f"ชื่อแผน/เรื่อง: {plan['plan_title']}")
        context_parts.append(f"โครงการ/แผนงาน: {plan['program_name']}")
        context_parts.append(f"วัตถุประสงค์: {plan['objectives']}")
        context_parts.append(f"ขอบเขต: {plan['scope']}")
        context_parts.append(f"สมมุติฐาน/ข้อจำกัด: {plan['assumptions']}")
        context_parts.append("\n--- 6W2H ---")
        context_parts.append(f"Who (ใคร): {plan['who']}")
        context_parts.append(f"Whom (เพื่อใคร): {plan['whom']}")
        context_parts.append(f"What (ทำอะไร): {plan['what']}")
        context_parts.append(f"Where (ที่ไหน): {plan['where']}")
        context_parts.append(f"When (เมื่อใด): {plan['when']}")
        context_parts.append(f"Why (ทำไม): {plan['why']}")
        context_parts.append(f"How (อย่างไร): {plan['how']}")
        context_parts.append(f"How much (เท่าไร): {plan['how_much']}")
        
        if not logic_df.empty:
            context_parts.append("\n--- Logic Model ---")
            context_parts.append(logic_df[["type", "description", "target"]].to_markdown(index=False))
            
        if not kpis_df.empty:
            context_parts.append("\n--- KPIs ---")
            context_parts.append(kpis_df[["name", "level", "formula", "target", "data_source"]].to_markdown(index=False))

        if not risks_df.empty:
            context_parts.append("\n--- Risks and Hypothesis ---")
            display_risks = risks_df.copy()
            display_risks["Score"] = display_risks["likelihood"] * display_risks["impact"]
            context_parts.append(display_risks[["description", "category", "Score", "hypothesis"]].to_markdown(index=False))
        
        ai_context = "\n".join(context_parts)
        
        st.caption("ข้อมูลที่จะถูกส่งให้ AI เพื่อพิจารณา:")
        with st.expander("แสดงข้อมูล Context"):
            st.code(ai_context, language="markdown")
            
        st.markdown("💡 **ยังไม่มี API Key?** คลิก [ที่นี่](https://playground.opentyphoon.ai/settings/api-key) เพื่อรับ key ฟรี!")
        api_key_assist = st.text_input("กรุณากรอก API Key เพื่อใช้บริการ AI Assist:", type="password", key="api_key_assist")

        if st.button("🚀 ให้ AI Assist แนะนำประเด็นการตรวจสอบและข้อตรวจพบที่คาดว่าจะพบ", type="primary", use_container_width=True):
            if not api_key_assist:
                st.error("กรุณากรอก API Key ก่อนใช้งาน")
            elif not any([plan['plan_title'], plan['objectives'], plan['who'], not logic_df.empty, not kpis_df.empty, not risks_df.empty]):
                 st.error("กรุณากรอกข้อมูลแผนเบื้องต้น (เช่น ชื่อแผน, วัตถุประสงค์) หรือเพิ่มรายการใน Logic Model/KPIs/Risks ก่อน")
            else:
                with st.spinner("กำลังประมวลผลคำแนะนำจาก AI..."):
                    try:
                        system_prompt = f"""
คุณคือ PA Assistant ผู้เชี่ยวชาญด้านการตรวจสอบผลการดำเนินงาน (Performance Audit)
เป้าหมายของคุณคือการวิเคราะห์ 'ข้อมูลแผนการตรวจสอบ' ที่ได้รับ และเสนอแนะสิ่งต่อไปนี้:
1. **ประเด็นการตรวจสอบที่ควรให้ความสำคัญ (Key Audit Issues)**: เสนอประเด็นสำคัญอย่างน้อย 3 ข้อ โดยพิจารณาจาก:
    - ความเชื่อมโยงของ Logic Model และ KPIs
    - ความเสี่ยงที่ระบุไว้ใน Risks and Hypothesis
    - ความสมบูรณ์ของ 6W2H และวัตถุประสงค์
    - ประเด็นควรมีความคมชัดและเน้นที่ประสิทธิภาพ ประสิทธิผล หรือความประหยัด
2. **ข้อตรวจพบที่คาดว่าจะพบ (Potential Findings)**: สำหรับแต่ละประเด็นที่เสนอ ให้คาดการณ์ว่าอาจพบข้อตรวจพบอะไรได้บ้าง พร้อมให้ **ระดับโอกาสในการพบ** (ต่ำ-ปานกลาง-สูง)
3. **สรุปรายงานการประเมินความเสี่ยงเบื้องต้น (Initial Risk Assessment Report)**: สรุปเป็นรายงานสั้นๆ โดยเน้นที่ความเสี่ยงสูงสุดและสมมุติฐานที่สำคัญ
---
**ข้อมูลแผนการตรวจสอบ:**
{ai_context}
---
**รูปแบบการตอบกลับที่ต้องการ:**
คุณต้องตอบกลับเป็นภาษาไทย และแบ่งผลลัพธ์ออกเป็น 3 ส่วน ตามลำดับดังนี้:
[KEY AUDIT ISSUES]
[รายการประเด็นการตรวจสอบที่ควรให้ความสำคัญในรูปแบบ Bullet Point]
...
[POTENTIAL FINDINGS]
[รายการข้อตรวจพบที่คาดว่าจะพบและระดับโอกาสในการพบ ในรูปแบบ Bullet Point โดยเชื่อมโยงกับ Key Audit Issues]
...
[INITIAL RISK ASSESSMENT REPORT]
[สรุปรายงานการประเมินความเสี่ยงเบื้องต้นสั้นๆ]
"""
                        client = OpenAI(
                            api_key=api_key_assist,
                            base_url="https://api.opentyphoon.ai/v1"
                        )
                        
                        response = client.chat.completions.create(
                            model="typhoon-v2.1-12b-instruct",
                            messages=[{"role": "user", "content": system_prompt}],
                            temperature=0.7,
                            max_tokens=3072,
                            top_p=0.9,
                        )
                        llm_output = response.choices[0].message.content
                        
                        # Parse the output into 3 sections
                        sections = {
                            "issues": "",
                            "findings": "",
                            "report": ""
                        }
                        
                        current_section = None
                        for line in llm_output.split('\n'):
                            line = line.strip()
                            if line == "[KEY AUDIT ISSUES]":
                                current_section = "issues"
                                continue
                            elif line == "[POTENTIAL FINDINGS]":
                                current_section = "findings"
                                continue
                            elif line == "[INITIAL RISK ASSESSMENT REPORT]":
                                current_section = "report"
                                continue
                            
                            if current_section and line:
                                sections[current_section] += line + "\n"
                        
                        st.session_state["gen_issues"] = sections["issues"].strip()
                        st.session_state["gen_findings"] = sections["findings"].strip()
                        st.session_state["gen_report"] = sections["report"].strip()

                        st.success("AI Assist ประมวลผลและสร้างคำแนะนำเรียบร้อยแล้ว")
                        st.balloons()
                        # Force a rerun to display the results in the markdown blocks
                        st.rerun() 

                    except Exception as e:
                        error_type = type(e).__name__
                        if error_type in ["AuthenticationError", "RateLimitError"]:
                             error_message = f"เกิดข้อผิดพลาดในการเชื่อมต่อ API: ({error_type}) โปรดตรวจสอบ API Key หรือขีดจำกัดการใช้งาน (Rate Limit) ของคุณ\nรายละเอียด: {e}"
                        else:
                             error_message = f"เกิดข้อผิดพลาดขณะทำงาน: ({error_type}) โปรดลองอีกครั้ง\nรายละเอียด: {e}"
                             
                        st.error(error_message)
                        st.session_state["gen_issues"] = ""
                        st.session_state["gen_findings"] = ""
                        st.session_state["gen_report"] = ""

    st.markdown("<h4 style='color:blue;'>ประเด็นการตรวจสอบที่ควรให้ความสำคัญ</h4>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color: #f0f2f6; border: 1px solid #ccc; padding: 10px; border-radius: 5px; height: 200px; overflow-y: scroll;'>{st.session_state.get('gen_issues', 'กดปุ่มเพื่อเริ่มรับคำแนะนำ')}</div>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='color:blue;'>ข้อตรวจพบที่คาดว่าจะพบ (พร้อมระดับโอกาส)</h4>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color: #f0f2f6; border: 1px solid #ccc; padding: 10px; border-radius: 5px; height: 200px; overflow-y: scroll;'>{st.session_state.get('gen_findings', 'กดปุ่มเพื่อเริ่มรับคำแนะนำ')}</div>", unsafe_allow_html=True)

    st.markdown("<h4 style='color:blue;'>สรุปรายงานการประเมินความเสี่ยงเบื้องต้น</h4>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color: #f0f2f6; border: 1px solid #ccc; padding: 10px; border-radius: 5px; height: 200px; overflow-y: scroll;'>{st.session_state.get('gen_report', 'กดปุ่มเพื่อเริ่มรับคำแนะนำ')}</div>", unsafe_allow_html=True)
    
# ----------------- Tab 9: RAG Chatbot -----------------
with tab_chatbot:
    st.subheader("💬 RAG Chatbot - ถาม-ตอบจากเอกสารที่อัปโหลด")

    # 1. File Uploader and Context Loading
    with st.container(border=True):
        st.markdown("##### อัปโหลดเอกสารเพื่อใช้ในการถาม-ตอบ (RAG Context)")
        
        # Helper function to clear context
        def clear_rag_context():
            st.session_state["uploaded_files_content"] = {}
            st.session_state["rag_docs_context"] = ""
            st.session_state["chatbot_messages"] = []
            st.success("ล้างบริบทเอกสารและประวัติแชทเรียบร้อย")

        uploaded_files = st.file_uploader(
            "อัปโหลดไฟล์ PDF หรือ TXT (จำกัดขนาดรวม 100,000 ตัวอักษร)", 
            type=["pdf", "txt"], 
            accept_multiple_files=True,
            key="rag_uploader"
        )
        
        # Load button logic (handles file reading)
        if st.button("🔄 โหลด/อัปเดตบริบท RAG", type="secondary"):
            if not uploaded_files:
                clear_rag_context()
            else:
                new_files_content = {}
                for file in uploaded_files:
                    file_name = file.name
                    # Check if file is new or modified (based on size)
                    if file_name not in st.session_state["uploaded_files_content"] or file.size != st.session_state["uploaded_files_content"].get(file_name, {}).get("size", 0):
                        
                        file_type = file.type
                        text = ""
                        
                        if "pdf" in file_type:
                            text = read_pdf_text_from_uploaded(file)
                            file_type = "pdf"
                        elif "text/plain" in file_type:
                            # Read as string
                            stringio = StringIO(file.getvalue().decode("utf-8"))
                            text = stringio.read()
                            file_type = "txt"
                        else:
                            # Skip unknown file types
                            continue
                        
                        # Apply character limit
                        text = text[:MAX_CHARS_LIMIT]
                        
                        new_files_content[file_name] = {"type": file_type, "content": text, "size": file.size}
                    else:
                        # Use cached content if file is the same
                        new_files_content[file_name] = st.session_state["uploaded_files_content"][file_name]

                st.session_state["uploaded_files_content"] = new_files_content
                
                # Combine context from session state
                load_rag_context(new_files_content)

        st.button("🗑️ ล้างบริบทและแชท", on_click=clear_rag_context)
        
        # Display current context status
        doc_names = list(st.session_state["uploaded_files_content"].keys())
        context_len = len(st.session_state.get("rag_docs_context", ""))
        st.caption(f"เอกสารในบริบท: {', '.join(doc_names) if doc_names else 'ไม่มี'} | ความยาวรวม: {context_len} อักขระ")

    st.divider()

    # 2. Chat UI and Logic
    st.markdown("💡 **ยังไม่มี API Key?** คลิก [ที่นี่](https://playground.opentyphoon.ai/settings/api-key) เพื่อรับ key ฟรี!")
    api_key_chatbot = st.text_input("กรุณากรอก API Key เพื่อใช้บริการ Chatbot:", type="password", key="api_key_chatbot")

    # Display chat messages
    for message in st.session_state.chatbot_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("ถามคำถามเกี่ยวกับเอกสารที่อัปโหลด..."):
        if not api_key_chatbot:
            st.error("กรุณากรอก API Key ก่อนใช้งาน")
        else:
            # Add user message to chat history
            st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get assistant response (API call)
            with st.chat_message("assistant"):
                # Check for context
                doc_context = st.session_state.get("rag_docs_context", "ไม่พบเอกสารภายในที่โหลด")
                
                try:
                    client = OpenAI(
                        api_key=api_key_chatbot,
                        base_url="https://api.opentyphoon.ai/v1"
                    )

                    system_prompt = f"""
คุณคือผู้ช่วย Chatbot ด้านการตรวจสอบผลการดำเนินงาน (Performance Audit) และมีความรู้ในการตอบคำถามจากบริบทที่ได้รับ
เป้าหมายของคุณคือการตอบคำถามของผู้ใช้โดยใช้ข้อมูลจาก 'บริบทจากเอกสารภายใน' เป็นหลัก
- หากข้อมูลในเอกสารภายในขัดแย้งกับความรู้ทั่วไป ให้ยึดข้อมูลในเอกสารเป็นหลักและอาจกล่าวถึงความขัดแย้งนั้น
- หากไม่พบคำตอบทั้งในเอกสารและความรู้ทั่วไป ให้ตอบว่า \"ขออภัยครับ ไม่พบข้อมูลที่เกี่ยวข้องทั้งในเอกสารและฐานข้อมูลของผม\"

---
**บริบทจากเอกสารภายใน:**
{doc_context}
---

จากข้อมูลข้างต้นนี้ จงตอบคำถามล่าสุดของผู้ใช้
\"\"\"
"""
                    
                    messages_for_api = [
                        {"role": "system", "content": system_prompt}
                    ]
                    # Add chat history, but keep it concise (last 10 messages including system prompt)
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
                    error_type = type(e).__name__
                    if error_type in ["AuthenticationError", "RateLimitError"]:
                         error_message = f"เกิดข้อผิดพลาดในการเชื่อมต่อ API: ({error_type}) โปรดตรวจสอบ API Key หรือขีดจำกัดการใช้งาน (Rate Limit) ของคุณ\nรายละเอียด: {e}"
                    else:
                         error_message = f"เกิดข้อผิดพลาดขณะทำงาน: ({error_type}) โปรดลองอีกครั้ง\nรายละเอียด: {e}"
                         
                    st.error(error_message)
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": error_message})
