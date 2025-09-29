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
# จากเดิมมี from PyPDF2 import PdfReader แต่ถูกลบออกแล้วเนื่องจากไม่มีการใช้ Chatbot

# ตั้งค่าหน้าเพจ
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
    # ลบ state ที่เกี่ยวข้องกับ Chatbot ออกทั้งหมด

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
# **ลบ tab_chatbot ออกจากรายการ**
tab_plan, tab_logic, tab_method, tab_kpi, tab_risk, tab_issue, tab_preview, tab_assist = st.tabs([
    "1. ระบุ แผน & 6W2H", 
    "2. ระบุ Logic Model", 
    "3. ระบุ Methods", 
    "4. ระบุ KPIs", 
    "5. ระบุ Risks", 
    "6. ค้นหาข้อตรวจพบที่ผ่านมา", 
    "7. สรุปข้อมูล (Preview)", 
    # **เปลี่ยนชื่อ Tab 8**
    "🤖 ให้ PA Assist ช่วยแนะนำประเด็นการตรวจสอบ ✨✨" 
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
                        # **แก้ไข: ลบ repetition_penalty ออก**
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
    st.subheader("🔎 แนะนำประเด็นตรวจจากรายงานเก่า (Issue Suggestions)")
    st.write("***กรุณาอัพโหลดฐานข้อมูล (ถ้าไม่มีจะใช้ฐานข้อมูลในระบบ)***")

    
    with st.container(border=True):
        st.download_button(
            label="⬇️ ดาวน์โหลดไฟล์แม่แบบ FindingsLibrary.xlsx",
            data=create_excel_template(),
            file_name="FindingsLibrary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        uploaded = st.file_uploader("อัปโหลด FindingsLibrary.csv หรือ .xlsx", type=["csv", "xlsx", "xls"])
    
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
        
        # Define a function to overwrite the text area's state with the latest seed
        def refresh_query_text(new_seed):
            st.session_state["issue_query_text"] = new_seed
            st.session_state["ref_seed"] = new_seed 

        # ------------------- การอัปเดตอัตโนมัติ -------------------
        # A. กรณีโหลดครั้งแรกสุด หรือค่าว่าง ให้ใส่ seed ใหม่และตั้งค่า ref_seed
        if "issue_query_text" not in st.session_state or st.session_state["issue_query_text"] == "":
            st.session_state["issue_query_text"] = seed
            st.session_state["ref_seed"] = seed
            
        # B. กรณี seed เปลี่ยน และค่าในช่องค้นหายังเป็นค่าเดิมจาก seed เก่า
        elif st.session_state.get("ref_seed") != seed and st.session_state.get("issue_query_text") == st.session_state.get("ref_seed"):
            st.session_state["issue_query_text"] = seed
            st.session_state["ref_seed"] = seed 
        # --------------------------------------------------------------------------------------------------------------

        # ใช้ Columns เพื่อจัดวางช่องค้นหาและปุ่มให้อยู่ข้างกัน
        c_query_area, c_refresh_btn = st.columns([6, 1])

        with c_query_area:
            query_text = st.text_area(
                "**สรุปบริบทที่ใช้ค้นหา (แก้ไขได้):**", 
                st.session_state["issue_query_text"], 
                height=140, 
                key="issue_query_text"
            )
        
        with c_refresh_btn:
            st.markdown("<br>", unsafe_allow_html=True) # เพิ่มช่องว่างจัดแนวปุ่ม
            st.button(
                "🔄 ดึงข้อมูลจากหน้าก่อนหน้า", 
                on_click=refresh_query_text,
                args=(seed,), # ส่งค่า seed ล่าสุดไปให้ฟังก์ชัน
                help="คลิกเพื่ออัปเดตช่องค้นหาด้วยข้อมูลล่าสุดจากแท็บ 'ระบุ แผน & 6W2H' และ 'ระบุ Logic Model' (จะล้างข้อมูลที่คุณเคยแก้ไข)",
                type="secondary"
            )
        
        # The search button logic
        if st.button("ค้นหาประเด็นที่ใกล้เคียง", type="primary", key="search_button_fix"):
            # Ensure we use the value stored in the session state for the search
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
                        st.caption("💡 **คำอธิบาย:** **คะแนนความเกี่ยวข้อง** (ยิ่งสูงยิ่งดี) = ความคล้ายคลึงของข้อความ + ความรุนแรงของปัญหา + ความใหม่ของข้อมูล")
                        st.caption("**Similarity Score** คือค่าความคล้ายคลึงระหว่างข้อความในแผนงานของคุณกับรายงานเก่า (0.000 - 1.000)")
    
                    c1, c2 = st.columns([3,1])
                    with c1:
                        default_rat = f"อ้างอิงกรณีเดิม ปี {year_txt} | หน่วย: {unit_txt}"
                        st.text_area("เหตุผลที่ควรตรวจ (สำหรับแผนนี้)", key=f"rat_{i}", value=default_rat)
                        st.text_input("KPI ที่เกี่ยว (ถ้ามี)", key=f"kpi_{i}")
                        st.text_input("วิธีเก็บข้อมูลที่เสนอ", key=f"mth_{i}", value="สัมภาษณ์/สังเกต/ตรวจเอกสาร")
    
                    with c2:
                        if st.button("➕ เพิ่มเข้าแผน", key=f"add_{i}", type="secondary"):
                            rationale_val = st.session_state.get(f"rat_{i}", "")
                            linked_kpi_val = st.session_state.get(f"kpi_{i}", "")
                            proposed_methods_val = st.session_state.get(f"mth_{i}", "")
                            issue_detail_val = row.get("issue_detail", "")
                            recommendation_val = row.get("recommendation", "")
    
                            cols = ["issue_id","plan_id","title","rationale","linked_kpi","proposed_methods","source_finding_id","issue_detail","recommendation"]
                            
                            if "audit_issues" not in st.session_state or not isinstance(st.session_state["audit_issues"], pd.DataFrame):
                                st.session_state["audit_issues"] = pd.DataFrame(columns=cols)
                            
                            for c in cols:
                                if c not in st.session_state["audit_issues"].columns:
                                    st.session_state["audit_issues"][c] = pd.Series(dtype="object")
                            
                            curr = st.session_state["audit_issues"]
                            new_id = next_id("ISS", curr, "issue_id")
    
                            title_val = title_txt
                            finding_id = row.get("finding_id", "")
    
                            new = pd.DataFrame([{
                                "issue_id": new_id,
                                "plan_id": plan.get("plan_id",""),
                                "title": title_val,
                                "rationale": rationale_val,
                                "linked_kpi": linked_kpi_val,
                                "proposed_methods": proposed_methods_val,
                                "source_finding_id": finding_id,
                                "issue_detail": issue_detail_val,
                                "recommendation": recommendation_val
                            }])
    
                            st.session_state["audit_issues"] = pd.concat([st.session_state["audit_issues"], new], ignore_index=True)
                            st.success("เพิ่มประเด็นเข้าแผนแล้ว ✅")
                            st.rerun()
                            
        if not st.session_state.get("issue_results", pd.DataFrame()).empty:
            st.divider()
        st.markdown("### ประเด็นที่ถูกเพิ่มเข้าแผน")
        st.dataframe(st.session_state["audit_issues"], use_container_width=True, hide_index=True)

# ----------------- Tab 7: สรุปข้อมูล (Preview) -----------------
with tab_preview:
    st.subheader("สรุปแผน (Preview)")
    with st.container(border=True):
        st.markdown(f"**Plan ID:** {plan['plan_id']}  \n**ชื่อแผนงาน:** {plan['plan_title']}  \n**โครงการ:** {plan['program_name']}  \n**หน่วยรับตรวจ:** {plan['who']}")
    st.markdown("### สรุปเรื่องที่ตรวจสอบ (จาก 6W2H)")
    with st.container(border=True):
        intro = f"""
- **Who**: {plan['who']}
- **Whom**: {plan['whom']}
- **What**: {plan['what']}
- **Where**: {plan['where']}
- **When**: {plan['when']}
- **Why**: {plan['why']}
- **How**: {plan['how']}
- **How much**: {plan['how_much']}
"""
        st.markdown(intro)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Logic Model")
        st.dataframe(st.session_state["logic_items"], use_container_width=True, hide_index=True)
        df_download_link(st.session_state["logic_items"], "logic_items.csv", "⬇️ ดาวน์โหลด Logic Items (CSV)")
    with c2:
        st.markdown("### Methods")
        st.dataframe(st.session_state["methods"], use_container_width=True, hide_index=True)
        df_download_link(st.session_state["methods"], "methods.csv", "⬇️ ดาวน์โหลด Methods (CSV)")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### KPIs")
        st.dataframe(st.session_state["kpis"], use_container_width=True, hide_index=True)
        df_download_link(st.session_state["kpis"], "kpis.csv", "⬇️ ดาวน์โหลด KPIs (CSV)")
    with c4:
        st.markdown("### Risks")
        st.dataframe(st.session_state["risks"], use_container_width=True, hide_index=True)
        df_download_link(st.session_state["risks"], "risks.csv", "⬇️ ดาวน์โหลด Risks (CSV)")

    st.markdown("### Audit Issues ที่เพิ่มเข้ามา")
    if not st.session_state["audit_issues"].empty:
        display_issues_df = st.session_state["audit_issues"].copy()
        display_issues_df = display_issues_df.rename(columns={
            "issue_id": "รหัสประเด็น",
            "title": "ชื่อประเด็น",
            "rationale": "เหตุผลที่ควรตรวจ",
            "issue_detail": "รายละเอียด",
            "recommendation": "ข้อเสนอแนะ"
        })
        display_cols = ["รหัสประเด็น", "ชื่อประเด็น", "เหตุผลที่ควรตรวจ", "รายละเอียด", "ข้อเสนอแนะ"]
        st.dataframe(display_issues_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("ยังไม่มีประเด็นการตรวจสอบที่เพิ่มเข้ามาในแผน")

    if not st.session_state["audit_issues"].empty:
        df_download_link(st.session_state["audit_issues"], "audit_issues.csv", "⬇️ ดาวน์โหลด Audit Issues (CSV)")

    st.divider()
    plan_df = pd.DataFrame([plan])
    df_download_link(plan_df, "plan.csv", "⬇️ ดาวน์โหลด Plan (CSV)")
    st.success("พร้อมเชื่อม Glide / Sheets ต่อได้ทันที")
    
# ----------------- Tab 8: ให้ PA Assist ช่วยแนะนำประเด็นการตรวจสอบ -----------------
with tab_assist:
    st.subheader("💡 PA Audit Assist (ขับเคลื่อนด้วย LLM)")
    st.write("🤖 สร้างคำแนะนำประเด็นการตรวจสอบจาก AI")
    st.markdown("💡 **ยังไม่มี API Key?** คลิก [ที่นี่](https://playground.opentyphoon.ai/settings/api-key) เพื่อรับ key ฟรี!")
    api_key = st.text_input("กรุณากรอก API Key เพื่อใช้บริการ AI:", type="password", key="api_key_assist")

    if st.button("🚀 สร้างคำแนะนำจาก AI", type="primary", key="llm_assist_button"):
        if not api_key:
            st.error("กรุณากรอก API Key ก่อนใช้งาน")
        else:
            with st.spinner("กำลังสร้างคำแนะนำ..."):
                try:
                    issues_for_llm = st.session_state['audit_issues'][['title', 'rationale']]
                    plan_summary = f"""
ชื่อแผน/เรื่องที่จะตรวจ: {plan['plan_title']}
ชื่อโครงการ/แผนงาน: {plan['program_name']}
วัตถุประสงค์: {plan['objectives']}
ขอบเขต: {plan['scope']}
สมมุติฐาน/ข้อจำกัด: {plan['assumptions']}
---
6W2H:
ใคร (Who): {plan['who']}
ถึงใคร (Whom): {plan['whom']}
ทำอะไร (What): {plan['what']}
ที่ไหน (Where): {plan['where']}
เมื่อใด (When): {plan['when']}
ทำไม (Why): {plan['why']}
อย่างไร (How): {plan['how']}
เท่าไร (How much): {plan['how_much']}
---
Logic Model:
{st.session_state['logic_items'].to_string()}
---
ประเด็นที่เพิ่มจากรายงานเก่า:
{issues_for_llm.to_string()}
"""
                    user_prompt = f"""
จากข้อมูลแผนการตรวจสอบด้านล่างนี้ กรุณาช่วยสร้างคำแนะนำ 3 อย่าง ได้แก่
1. ประเด็นการตรวจสอบที่ควรให้ความสำคัญ
2. ข้อตรวจพบที่คาดว่าจะพบ (พร้อมระบุระดับโอกาสที่จะเจอ: สูง/กลาง/ต่ำ)
3. ร่างรายงานตรวจสอบที่จะเจอ
---
{plan_summary}
---
กรุณาสร้างคำตอบตามรูปแบบด้านล่างนี้เท่านั้น:
<ประเด็นการตรวจสอบที่ควรให้ความสำคัญ>
[ข้อความสำหรับส่วนที่ 1]
</ประเด็นการตรวจสอบที่ควรให้ความสำคัญ>

<ข้อตรวจพบที่คาดว่าจะพบ>
[ข้อความสำหรับส่วนที่ 2]
</ข้อตรวจพบที่คาดว่าจะพบ>

<ร่างรายงานตรวจสอบที่จะเจอ>
[ข้อความสำหรับส่วนที่ 3]
</ร่างรายงานตรวจสอบที่จะเจอ>
"""

                    client = OpenAI(
                        api_key=api_key,
                        base_url="https://api.opentyphoon.ai/v1"
                    )
                    
                    messages = [
                        {"role": "system", "content": "คุณคือผู้เชี่ยวชาญด้านการตรวจสอบผลสัมฤทธิ์และประสิทธิภาพการดำเนินงาน (Performance Audit) กรุณาตอบโดยมุ่งเน้นการสร้างคำแนะนำตามรูปแบบที่ต้องการเท่านั้น"},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    # **แก้ไข: ลบ repetition_penalty ออก**
                    response = client.chat.completions.create(
                        model="typhoon-v2.1-12b-instruct",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2048,
                        top_p=0.9,
                    )

                    full_response = response.choices[0].message.content

                    issue_start = full_response.find("<ประเด็นการตรวจสอบที่ควรให้ความสำคัญ>") + len("<ประเด็นการตรวจสอบที่ควรให้ความสำคัญ>")
                    issue_end = full_response.find("</ประเด็นการตรวจสอบที่ควรให้ความสำคัญ>")
                    issues_text = full_response[issue_start:issue_end].strip()
                    
                    finding_start = full_response.find("<ข้อตรวจพบที่คาดว่าจะพบ>") + len("<ข้อตรวจพบที่คาดว่าจะพบ>")
                    finding_end = full_response.find("</ข้อตรวจพบที่คาดว่าจะพบ>")
                    findings_text = full_response[finding_start:finding_end].strip()

                    report_start = full_response.find("<ร่างรายงานตรวจสอบที่จะเจอ>") + len("<ร่างรายงานตรวจสอบที่จะเจอ>")
                    report_end = full_response.find("</ร่างรายงานตรวจสอบที่จะเจอ>")
                    report_text = full_response[report_start:report_end].strip()

                    st.session_state["gen_issues"] = issues_text
                    st.session_state["gen_findings"] = findings_text
                    st.session_state["gen_report"] = report_text

                    st.success("สร้างคำแนะนำจาก AI เรียบร้อยแล้ว ✅")

                except Exception as e:
                    # ปรับปรุงการแสดงข้อผิดพลาด API
                    error_type = type(e).__name__
                    if "APIError" in error_type or "AuthenticationError" in error_type:
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

    st.markdown("<h4 style='color:blue;'>ร่างรายงานตรวจสอบ (Preview)</h4>", unsafe_allow_html=True)
    st.markdown(f"<div style='background-color: #f0f2f6; border: 1px solid #ccc; padding: 10px; border-radius: 5px; height: 400px; overflow-y: scroll;'>{st.session_state.get('gen_report', '')}</div>", unsafe_allow_html=True)


