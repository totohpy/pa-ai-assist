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
    """อ่านไฟล์ทั้งหมดจากคลังข้อมูล"""
    text = ""
    if not os.path.isdir(folder_path):
        return text # คืนค่าว่างถ้าไม่พบโฟลเดอร์

    try:
        files_in_doc = os.listdir(folder_path)
        progress_bar = st.sidebar.progress(0, text=f"กำลังโหลดเอกสารจากคลังข้อมูล... (0/{len(files_in_doc)})")
        for i, filename in enumerate(files_in_doc):
            if len(text) >= MAX_CHARS_LIMIT:
                st.warning(f"ถึงขีดจำกัดข้อมูลจากคลังข้อมูลแล้ว ({MAX_CHARS_LIMIT:,} ตัวอักษร)")
                break
            
            file_path = os.path.join(folder_path, filename)
            try:
                if filename.endswith('.pdf'):
                    with open(file_path, 'rb') as f:
                        reader = PdfReader(f)
                        for page in reader.pages:
                            text += page.extract_text() or ""
                elif filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text += f.read()
                elif filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    text += df.head(15).to_string()
            except Exception as e:
                print(f"Could not read file {filename}: {e}")
            
            progress_bar.progress((i + 1) / len(files_in_doc), text=f"กำลังโหลดเอกสาร... ({i+1}/{len(files_in_doc)})")
        progress_bar.empty()
                
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการเข้าถึงคลังข้อมูล: {e}")
    
    return text[:MAX_CHARS_LIMIT]

def process_documents(files, source_type, limit, current_len=0):
    """ฟังก์ชันสำหรับอ่านข้อความจากไฟล์ที่อัปโหลด"""
    text = ""
    for file in files:
        if current_len + len(text) >= limit:
            st.warning(f"ถึงขีดจำกัดจำนวนตัวอักษรสูงสุด ({limit:,}) แล้ว ไฟล์บางส่วนอาจไม่ถูกประมวลผล")
            break
        try:
            if file.name.endswith('.pdf'):
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            elif file.name.endswith('.txt'):
                text += file.getvalue().decode("utf-8")
            elif file.name.endswith('.csv'):
                df = pd.read_csv(file)
                text += df.head(15).to_string()
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์ {file.name}: {e}")
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
    ss.setdefault("gen_issues", "")
    ss.setdefault("gen_findings", "")
    ss.setdefault("gen_report", "")
    ss.setdefault("issue_results", pd.DataFrame())
    ss.setdefault("ref_seed", "")
    ss.setdefault("issue_query_text", "")
    ss.setdefault('api_key_global', '')
    
    ss.setdefault('chatbot_messages', [{"role": "assistant", "content": "สวัสดีครับ ผมคือ PA Chat ผู้ช่วยอัจฉริยะด้านการตรวจสอบ"}])
    ss.setdefault('doc_context_uploaded', "")
    ss.setdefault('last_uploaded_files', set())

    if 'doc_context_local' not in ss:
        ss.doc_context_local = load_local_documents()
        if ss.doc_context_local and os.path.isdir('Doc'):
             ss.chatbot_messages.append({"role": "assistant", "content": f"ผมได้โหลดเอกสาร {len(os.listdir('Doc'))} ฉบับ เป็นฐานความรู้เรียบร้อยแล้วครับ"})

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

init_state()
plan = st.session_state["plan"]
logic_df = st.session_state["logic_items"]
methods_df = st.session_state["methods"]
kpis_df = st.session_state["kpis"]
risks_df = st.session_state["risks"]
audit_issues_df = st.session_state["audit_issues"]

st.title("🧭 Planning Studio – Performance Audit")

# --- MODIFIED: ย้าย st.info ไปไว้ใน st.expander ---
with st.expander("💡 คำแนะนำการใช้งาน (คลิกเพื่ออ่าน)"):
    st.info(
        "ก่อนใช้งานแท็บ **'6. ค้นหาข้อตรวจพบที่ผ่านมา'** หรือ **'🤖 ให้ PA Assistant แนะนำฯ'** "
        "กรุณากรอกข้อมูลในแท็บ **'1. ระบุ แผน & 6W2H'** ก่อน เพื่อให้ AI ได้รับบริบทที่ครบถ้วนและให้คำแนะนำที่แม่นยำที่สุด"
    )

# --- st.markdown ที่แก้ไขแล้ว ---
st.markdown("""
<style> 
    body { font-family: 'Kanit', sans-serif; } 

    /* --- Base Style for All Tabs (ลักษณะปุ่มทึบ) --- */
    button[data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 14px; /* --- MODIFIED: ลด Padding ภายในปุ่ม --- */
        margin: 2px; /* --- MODIFIED: ปรับ Margin เล็กน้อย --- */
        font-weight: bold;
        color: white !important; 
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s ease-in-out;
    }

    /* --- สไตล์ของแท็บที่ถูกเลือก (ดึงให้ลอยขึ้นและสีสว่างขึ้น) --- */
    button[data-baseweb="tab"][aria-selected="true"] {
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        transform: translateY(-2px);
        filter: brightness(115%); /* --- NEW: ทำให้สีสว่างขึ้น 15% --- */
    }
    
    /* --- สไตล์เมื่อนำเมาส์ไปชี้ (ลอยขึ้นเล็กน้อย) --- */
    button[data-baseweb="tab"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
    }

    /* --- Group 1: 1-5 (สีน้ำเงิน) --- */
    div[data-baseweb="tab-list"] button:nth-of-type(-n+5) {
        background-color: #0d6efd; 
    }

    /* --- Group 2: 6-7 (สีม่วง) --- */
    div[data-baseweb="tab-list"] button:nth-of-type(6), 
    div[data-baseweb="tab-list"] button:nth-of-type(7) {
        background-color: #6f42c1;
    }

    /* --- Group 3: 8 (สีทอง) --- */
    div[data-baseweb="tab-list"] button:nth-of-type(8) {
        background-color: #ffc107;
        color: #343a40 !important;
    }

    /* --- Group 4: 9 (สีเขียว) --- */
    div[data-baseweb="tab-list"] button:nth-of-type(9) {
        background-color: #198754;
    }

    /* --- General Layout --- */
    div[data-baseweb="tab-list"] { 
        border-bottom: none !important; 
        margin-bottom: 15px; 
        flex-wrap: wrap; 
        gap: 6px; /* --- MODIFIED: ลดระยะห่างระหว่างปุ่ม --- */
    } 
    h4 { 
        color: #007bff !important; 
        border-bottom: 2px solid #e0e0e0; 
        padding-bottom: 5px; 
    } 
</style>
""", unsafe_allow_html=True)

tab_plan, tab_logic, tab_method, tab_kpi, tab_risk, tab_issue, tab_preview, tab_assist, tab_chatbot = st.tabs(["1. ระบุ แผน & 6W2H", "2. ระบุ Logic Model", "3. ระบุ Methods", "4. ระบุ KPIs", "5. ระบุ Risks", "6. ค้นหาข้อตรวจพบที่ผ่านมา", "7. สรุปข้อมูล (Preview)", "✨PA Assistant แนะนำประเด็นการตรวจสอบ", "💬 PA Chat (ถาม-ตอบ)"]) 

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
        uploaded_text = st.text_area("ระบุข้อความเกี่ยวกับเรื่องที่จะตรวจสอบ ที่ต้องการให้ AI ช่วยสรุป 6W2H", height=200, key="uploaded_text")
        
        if st.button("🚀 สร้าง 6W2H จากข้อความ", type="primary", key="6w2h_button"):
            if not uploaded_text: st.error("กรุณาวางข้อความในช่องก่อน")
            elif not st.session_state.api_key_global: st.error("กรุณากรอก API Key ที่ Sidebar ด้านซ้ายก่อนใช้งาน")
            else:
                with st.spinner("กำลังประมวลผล..."):
                    try:
                        user_prompt = f"จากข้อความด้านล่างนี้ กรุณาสรุปและแยกแยะข้อมูลให้เป็น 6W2H ได้แก่ Who, Whom, What, Where, When, Why, How, และ How much โดยให้อยู่ในรูปแบบ key-value ที่ชัดเจน\nข้อความ: --- {uploaded_text} ---"
                        client = OpenAI(api_key=st.session_state.api_key_global, base_url="https://api.opentyphoon.ai/v1")
                        response = client.chat.completions.create(model="typhoon-v2.1-12b-instruct", messages=[{"role": "user", "content": user_prompt}], temperature=0.7, max_tokens=1024, top_p=0.9)
                        llm_output = response.choices[0].message.content
                        for line in llm_output.strip().split('\n'):
                            if ':' in line:
                                key, value = line.split(':', 1)
                                normalized_key = key.strip().lower().replace(' ', '_')
                                if normalized_key in st.session_state.plan: st.session_state.plan[normalized_key] = value.strip()
                        st.success("สร้าง 6W2H เรียบร้อยแล้ว! กรุณาตรวจสอบข้อมูลด้านล่าง")
                        st.balloons()
                    except Exception as e: st.error(f"เกิดข้อผิดพลาดในการเรียกใช้ AI: {e}")
        
    st.markdown("##### ⭐กรุณาระบุข้อมูล เพื่อนำไปใช้ประมวลผล")
    with st.container(border=True):
        cc1, cc2, cc3 = st.columns(3)
        st.session_state.plan["who"] = cc1.text_input("Who (ใคร)", value=st.session_state.plan["who"], key="who_input")
        st.session_state.plan["whom"] = cc1.text_input("Whom (เพื่อใคร)", value=st.session_state.plan["whom"], key="whom_input")
        st.session_state.plan["what"] = cc1.text_input("What (ทำอะไร)", value=st.session_state.plan["what"], key="what_input")
        st.session_state.plan["where"] = cc1.text_input("Where (ที่ไหน)", value=st.session_state.plan["where"], key="where_input")
        st.session_state.plan["when"] = cc2.text_input("When (เมื่อใด)", value=st.session_state.plan["when"], key="when_input")
        st.session_state.plan["why"] = cc2.text_area("Why (ทำไม)", value=st.session_state.plan["why"], key="why_input")
        st.session_state.plan["how"] = cc3.text_area("How (อย่างไร)", value=st.session_state.plan["how"], key="how_input")
        st.session_state.plan["how_much"] = cc3.text_input("How much (เท่าไร)", value=st.session_state.plan["how_much"], key="how_much_input")

with tab_logic:
    st.subheader("ระบุข้อมูล Logic Model: Input → Activities → Output → Outcome → Impact")
    st.dataframe(logic_df, use_container_width=True, hide_index=True)
    with st.expander("➕ เพิ่มรายการใน Logic Model"):
        with st.container(border=True):
            colA, colB, colC = st.columns(3)
            typ = colA.selectbox("ประเภท", ["Input","Activity","Output","Outcome","Impact"])
            desc = colA.text_input("คำอธิบาย/รายละเอียด")
            metric = colA.text_input("ตัวชี้วัด/metric (เช่น จำนวน, สัดส่วน)")
            unit = colB.text_input("หน่วย", value="หน่วย", key="logic_unit")
            target = colB.text_input("เป้าหมาย", value="", key="logic_target")
            source = colC.text_input("แหล่งข้อมูล", value="", key="logic_source")
            if st.button("เพิ่ม Logic Item", type="primary", key="add_logic_item_btn"):
                new_row = pd.DataFrame([{"item_id": next_id("LG", logic_df, "item_id"),"plan_id": plan["plan_id"],"type": typ, "description": desc, "metric": metric,"unit": unit, "target": target, "source": source}])
                st.session_state["logic_items"] = pd.concat([logic_df, new_row], ignore_index=True)
                st.rerun()

with tab_method:
    st.subheader("ระบุวิธีการเก็บข้อมูล (Methods)")
    st.dataframe(methods_df, use_container_width=True, hide_index=True)
    with st.expander("➕ เพิ่ม Method"):
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            mtype = c1.selectbox("ชนิด", ["observe","interview","questionnaire","document"])
            tool_ref = c1.text_input("รหัส/อ้างอิงเครื่องมือ", value="")
            sampling = c1.text_input("วิธีคัดเลือกตัวอย่าง", value="")
            questions = c2.text_area("คำถาม/ประเด็นหลัก")
            linked_issue = c2.text_input("โยงประเด็นตรวจ", value="")
            data_source = c3.text_input("แหล่งข้อมูล", value="", key="method_data_source")
            frequency = c3.text_input("ความถี่", value="ครั้งเดียว", key="method_frequency")
            if st.button("เพิ่ม Method", type="primary", key="add_method_btn"):
                new_row = pd.DataFrame([{"method_id": next_id("MT", methods_df, "method_id"),"plan_id": plan["plan_id"],"type": mtype, "tool_ref": tool_ref, "sampling": sampling,"questions": questions, "linked_issue": linked_issue,"data_source": data_source, "frequency": frequency}])
                st.session_state["methods"] = pd.concat([methods_df, new_row], ignore_index=True)
                st.rerun()

with tab_kpi:
    st.subheader("ระบุตัวชี้วัด (KPIs)")
    st.dataframe(kpis_df, use_container_width=True, hide_index=True)
    with st.expander("➕ เพิ่ม KPI เอง"):
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            level = col1.selectbox("ระดับ", ["output","outcome"])
            name = col1.text_input("ชื่อ KPI")
            formula = col1.text_input("สูตร/นิยาม")
            numerator = col2.text_input("ตัวตั้ง (numerator)")
            denominator = col2.text_input("ตัวหาร (denominator)")
            unit = col2.text_input("หน่วย", value="%", key="kpi_unit")
            baseline = col3.text_input("Baseline", value="")
            target = col3.text_input("Target", value="")
            freq = col3.text_input("ความถี่", value="รายไตรมาส")
            data_src = col3.text_input("แหล่งข้อมูล", value="", key="kpi_data_source")
            quality = col3.text_input("ข้อกำหนดคุณภาพข้อมูล", value="ถูกต้อง/ทันเวลา", key="kpi_quality")
            if st.button("เพิ่ม KPI", type="primary", key="add_kpi_btn"):
                new_row = pd.DataFrame([{"kpi_id": next_id("KPI", kpis_df, "kpi_id"),"plan_id": plan["plan_id"],"level": level, "name": name, "formula": formula,"numerator": numerator, "denominator": denominator, "unit": unit,"baseline": baseline, "target": target, "frequency": freq,"data_source": data_src, "quality_requirements": quality}])
                st.session_state["kpis"] = pd.concat([kpis_df, new_row], ignore_index=True)
                st.rerun()

with tab_risk:
    st.subheader("ระบุความเสี่ยง (Risks)")
    st.dataframe(risks_df, use_container_width=True, hide_index=True)
    with st.expander("➕ เพิ่ม Risk"):
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            desc = c1.text_area("คำอธิบายความเสี่ยง")
            category = c1.selectbox("หมวด", ["policy","org","data","process","people"])
            likelihood = c2.select_slider("โอกาสเกิด (1-5)", options=[1,2,3,4,5], value=3)
            impact = c2.select_slider("ผลกระทบ (1-5)", options=[1,2,3,4,5], value=3)
            mitigation = c3.text_area("มาตรการลดความเสี่ยง")
            hypothesis = c3.text_input("สมมุติฐานที่ต้องทดสอบ")
            if st.button("เพิ่ม Risk", type="primary", key="add_risk_btn"):
                new_row = pd.DataFrame([{"risk_id": next_id("RSK", risks_df, "risk_id"),"plan_id": plan["plan_id"],"description": desc, "category": category,"likelihood": likelihood, "impact": impact,"mitigation": mitigation, "hypothesis": hypothesis}])
                st.session_state["risks"] = pd.concat([risks_df, new_row], ignore_index=True)
                st.rerun()

with tab_issue:
    st.subheader("🔎 แนะนำประเด็นตรวจจากรายงานเก่า (Issue Suggestions)")
    with st.expander("อัปโหลดและจัดการฐานข้อมูลข้อตรวจพบ (Findings Library)"):
        st.write("คุณสามารถอัปโหลดไฟล์ .csv หรือ .xlsx ที่มีข้อมูลข้อตรวจพบเก่าเพื่อใช้ในการค้นหา หรือดาวน์โหลดไฟล์แม่แบบไปใช้งาน")
        st.download_button(
            label="⬇️ ดาวน์โหลดไฟล์แม่แบบ FindingsLibrary.xlsx",
            data=create_excel_template(),
            file_name="FindingsLibrary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        uploaded = st.file_uploader("อัปโหลด FindingsLibrary.csv หรือ .xlsx", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    
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
                "🔄", 
                on_click=refresh_query_text,
                args=(seed,),
                help="คลิกเพื่ออัปเดตช่องค้นหาด้วยข้อมูลล่าสุด",
                type="secondary"
            )
        
        top_k_slider = st.slider("ปรับจำนวนผลลัพธ์ที่ต้องการแสดง:", min_value=1, max_value=20, value=8)

        if st.button("ค้นหาประเด็นที่ใกล้เคียง", type="primary", key="search_button_fix"):
            search_value = st.session_state.get("issue_query_text", seed)
            results = search_candidates(search_value, findings_df, vec, X, top_k=top_k_slider)
            st.session_state["issue_results"] = results
            st.success(f"พบประเด็นที่เกี่ยวข้อง {len(results)} รายการ")
            
        results = st.session_state.get("issue_results", pd.DataFrame())
        
        if not results.empty:
            st.divider()
            st.subheader("ผลลัพธ์การค้นหา")
            for i, row in results.reset_index(drop=True).iterrows():
                with st.container(border=True):
                    title_txt = row.get("issue_title", "(ไม่มีชื่อประเด็น)")
                    year_txt = int(row["year"]) if "year" in row and str(row["year"]).isdigit() else row.get("year", "-")
                    st.markdown(f"**{title_txt}** \nหน่วย: {row.get('unit', '-')} • โครงการ: {row.get('program', '-')} • ปี {year_txt}")
                    st.caption(f"สาเหตุ: *{row.get('cause_category', '-')}* — {row.get('cause_detail', '-')}")
    
                    with st.expander("รายละเอียด/ข้อเสนอแนะ (เดิม)"):
                        st.write(row.get("issue_detail", "-"))
                        st.caption("ข้อเสนอแนะเดิม: " + (row.get("recommendation", "") or "-"))
                        st.markdown(f"**ผลกระทบที่อาจเกิดขึ้น:** {row.get('outcomes_impact','-')}  •  <span style='color:red;'>**คะแนนความเกี่ยวข้อง**</span>: {row.get('score',0):.3f} (<span style='color:blue;'>**Similarity Score**</span>={row.get('sim_score',0):.3f})", unsafe_allow_html=True)
                        st.caption("💡 **คำอธิบาย:** **คะแนนความเกี่ยวข้อง** = ความคล้ายคลึงของข้อความ + ความรุนแรงของปัญหา + ความใหม่ของข้อมูล")
    
                    c1, c2 = st.columns([3,1])
                    with c1:
                        st.text_area("เหตุผลที่ควรตรวจ (สำหรับแผนนี้)", key=f"rat_{i}", value=f"อ้างอิงกรณีเดิม ปี {year_txt} | หน่วย: {row.get('unit', '-')}")
                        st.text_input("KPI ที่เกี่ยว (ถ้ามี)", key=f"kpi_{i}")
                        st.text_input("วิธีเก็บข้อมูลที่เสนอ", key=f"mth_{i}", value="สัมภาษณ์/สังเกต/ตรวจเอกสาร")
    
                    with c2:
                        if st.button("➕ เพิ่มเข้าแผน", key=f"add_{i}", type="secondary"):
                            new_row = pd.DataFrame([{"issue_id": next_id("ISS", audit_issues_df, "issue_id"),"plan_id": plan.get("plan_id",""),"title": title_txt,"rationale": st.session_state.get(f"rat_{i}", ""),"linked_kpi": st.session_state.get(f"kpi_{i}", ""),"proposed_methods": st.session_state.get(f"mth_{i}", ""),"source_finding_id": row.get("finding_id", ""),"issue_detail": row.get("issue_detail", ""),"recommendation": row.get("recommendation", "")}])
                            st.session_state["audit_issues"] = pd.concat([audit_issues_df, new_row], ignore_index=True)
                            st.success("เพิ่มประเด็นเข้าแผนแล้ว ✅")
                            st.rerun()
                            
        if not st.session_state.get("issue_results", pd.DataFrame()).empty:
            st.divider()
        st.markdown("### ประเด็นที่ถูกเพิ่มเข้าแผน")
        st.dataframe(st.session_state["audit_issues"], use_container_width=True, hide_index=True)

with tab_preview:
    st.subheader("สรุปแผน (Preview)")
    with st.container(border=True):
        st.markdown(f"**Plan ID:** {plan['plan_id']}  \n**ชื่อแผนงาน:** {plan['plan_title']}  \n**โครงการ:** {plan['program_name']}  \n**หน่วยรับตรวจ:** {plan['who']}")
    st.markdown("### สรุปเรื่องที่ตรวจสอบ (จาก 6W2H)")
    with st.container(border=True):
        st.markdown(f"- **Who**: {plan['who']}\n- **Whom**: {plan['whom']}\n- **What**: {plan['what']}\n- **Where**: {plan['where']}\n- **When**: {plan['when']}\n- **Why**: {plan['why']}\n- **How**: {plan['how']}\n- **How much**: {plan['how_much']}")
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
        display_issues_df = st.session_state["audit_issues"].copy().rename(columns={"issue_id": "รหัสประเด็น", "title": "ชื่อประเด็น","rationale": "เหตุผลที่ควรตรวจ", "issue_detail": "รายละเอียด","recommendation": "ข้อเสนอแนะ"})
        display_cols = ["รหัสประเด็น", "ชื่อประเด็น", "เหตุผลที่ควรตรวจ", "รายละเอียด", "ข้อเสนอแนะ"]
        st.dataframe(display_issues_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("ยังไม่มีประเด็นการตรวจสอบที่เพิ่มเข้ามาในแผน")
    if not st.session_state["audit_issues"].empty:
        df_download_link(st.session_state["audit_issues"], "audit_issues.csv", "⬇️ ดาวน์โหลด Audit Issues (CSV)")
    st.divider()
    plan_df = pd.DataFrame([plan])
    df_download_link(plan_df, "plan.csv", "⬇️ ดาวน์โหลด Plan (CSV)")
    st.success("พร้อมเชื่อมต่อ 🤖 ให้ PA Assistant แนะนำประเด็นการตรวจสอบ ✨✨")

with tab_assist:
    st.subheader("💡 PA Audit Assistant (ขับเคลื่อนด้วย LLM)")
    st.write("🤖 สร้างคำแนะนำประเด็นการตรวจสอบจาก AI")

    if st.button("🚀 สร้างคำแนะนำจาก AI", type="primary", key="llm_assist_button"):
        if not st.session_state.api_key_global:
            st.error("กรุณากรอก API Key ที่ Sidebar ด้านซ้ายก่อนใช้งาน")
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
                        api_key=st.session_state.api_key_global,
                        base_url="https://api.opentyphoon.ai/v1"
                    )
                    
                    messages = [
                        {"role": "system", "content": "คุณคือผู้เชี่ยวชาญด้านการตรวจสอบผลสัมฤทธิ์และประสิทธิภาพการดำเนินงาน (Performance Audit)"},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    response = client.chat.completions.create(
                        model="typhoon-v2.1-12b-instruct",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2048,
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
                    st.error(f"เกิดข้อผิดพลาดในการเรียกใช้ AI: {e}")
                    st.session_state["gen_issues"] = ""
                    st.session_state["gen_findings"] = ""
                    st.session_state["gen_report"] = ""
    
    st.subheader("ผลลัพธ์จาก AI")

    with st.expander("1. ประเด็นการตรวจสอบที่ควรให้ความสำคัญ", expanded=True):
        st.write(st.session_state.get('gen_issues', "ยังไม่มีข้อมูล กด 'สร้างคำแนะนำจาก AI' เพื่อเริ่มต้น"))

    with st.expander("2. ข้อตรวจพบที่คาดว่าจะพบ (พร้อมระดับโอกาส)"):
        st.write(st.session_state.get('gen_findings', "ยังไม่มีข้อมูล"))

    with st.expander("3. ร่างรายงานตรวจสอบ (Preview)"):
        st.write(st.session_state.get('gen_report', "ยังไม่มีข้อมูล"))


with tab_chatbot:
    st.subheader("💬 PA Chat - ผู้ช่วยอัจฉริยะ (RAG)")
    
    with st.expander("อัปโหลดเอกสารเพิ่มเติม (PDF, TXT, CSV)"):
        st.info("ข้อมูลจากไฟล์ที่คุณอัปโหลดจะถูกนำไปรวมกับเอกสารในโฟลเดอร์ 'Doc' เพื่อใช้เป็นบริบทในการตอบคำถาม")
        uploaded_files = st.file_uploader(
            "เลือกไฟล์...",
            type=['pdf', 'txt', 'csv'],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

    current_uploaded_file_names = {f.name for f in uploaded_files}
    if uploaded_files and st.session_state.get('last_uploaded_files') != current_uploaded_file_names:
        with st.spinner("กำลังประมวลผลเอกสารที่อัปโหลด..."):
            st.session_state.doc_context_uploaded, _ = process_documents(uploaded_files, 'uploaded', MAX_CHARS_LIMIT, len(st.session_state.get('doc_context_local', '')))
            st.session_state.last_uploaded_files = current_uploaded_file_names
            st.session_state.chatbot_messages.append({"role": "assistant", "content": "อัปเดตเอกสารใหม่เรียบร้อยแล้วครับ"})
            st.rerun()
    elif not uploaded_files and st.session_state.doc_context_uploaded:
        st.session_state.doc_context_uploaded = ""
        st.session_state.last_uploaded_files = set()
        st.session_state.chatbot_messages.append({"role": "assistant", "content": "ได้ทำการล้างเอกสารที่อัปโหลดออกแล้วครับ"})
        st.rerun()

    local_len = len(st.session_state.get('doc_context_local', ''))
    uploaded_len = len(st.session_state.get('doc_context_uploaded', ''))
    
    with st.expander("ดูรายละเอียดบริบท (Context) ที่ใช้"):
        if local_len > 0:
            st.info(f"💾 บริบทจากโฟลเดอร์ 'Doc': {local_len:,} ตัวอักษร")
        if uploaded_len > 0:
            st.info(f"📤 บริบทจากไฟล์ที่อัปโหลด: {uploaded_len:,} ตัวอักษร")
        st.success(f"✅ บริบทรวมทั้งหมด: {(local_len + uploaded_len):,} ตัวอักษร (จำกัดสูงสุด: {MAX_CHARS_LIMIT:,})")

    chat_container = st.container(height=500, border=True)
    with chat_container:
        for message in st.session_state.chatbot_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่...", key="chat_input_main"):
        st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                api_key = st.session_state.api_key_global
                if not api_key:
                    error_message = "เกิดข้อผิดพลาด: ไม่พบ API Key กรุณากรอก API Key ที่ Sidebar ด้านซ้ายก่อนใช้งาน"
                    message_placeholder.error(error_message)
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": error_message})
                else:
                    try:
                        doc_context = st.session_state.get('doc_context_local', '') + st.session_state.get('doc_context_uploaded', '')
                        
                        system_prompt = f"""
คุณคือผู้ช่วย AI อัจฉริยะ (Expert Assistant) หน้าที่ของคุณคือตอบคำถามของผู้ใช้ให้ถูกต้องและครบถ้วนที่สุด โดยใช้แหล่งข้อมูลสองแหล่ง:
1.  **ข้อมูลจากเอกสารภายใน (Primary Source):** นี่คือเนื้อหาที่ดึงมาจากไฟล์ PDF ในโฟลเดอร์ "Doc" ของระบบ จงยึดข้อมูลนี้เป็นหลักในการตอบคำถามเสมอ
2.  **ความรู้ทั่วไปและข้อมูลจากอินเทอร์เน็ต (Secondary Source):** หากคำตอบไม่มีอยู่ในเอกสารภายใน ให้ใช้ความรู้ที่คุณมีจากการฝึกฝน (ซึ่งเทียบเท่าการค้นหาข้อมูลบนอินเทอร์เน็ต) เพื่อตอบคำถาม
**กฎการตอบ:**
- เมื่อตอบคำถาม ให้อ้างอิงเสมอว่าข้อมูลมาจากแหล่งใด (เช่น "จากเอกสาร [ชื่อไฟล์] ระบุว่า..." หรือ "จากเอกสารที่ให้มา" ระบุว่า...) หากไม่ทราบชื่อไฟล์ให้บอกว่า "จากเอกสารที่ให้มา"
- หากข้อมูลในเอกสารขัดแย้งกับข้อมูลทั่วไป ให้ยึดข้อมูลในเอกสารเป็นหลักและอาจกล่าวถึงความขัดแย้งนั้น
- หากไม่พบคำตอบทั้งในเอกสารและความรู้ทั่วไป ให้ตอบว่า "ขออภัยครับ ไม่พบข้อมูลที่เกี่ยวข้องทั้งในเอกสารและฐานข้อมูลของผม"
---
**บริบทจากเอกสารภายใน:**
{doc_context}
---
จากข้อมูลข้างต้นนี้ จงตอบคำถามล่าสุดของผู้ใช้
"""
                        messages_for_api = [
                            {"role": "system", "content": system_prompt}
                        ] + st.session_state.chatbot_messages[-10:]

                        client = OpenAI(api_key=api_key, base_url="https://api.opentyphoon.ai/v1")
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
                        error_message = f"เกิดข้อผิดพลาดขณะทำงาน: {e}"
                        message_placeholder.error(error_message)
                        st.session_state.chatbot_messages.append({"role": "assistant", "content": error_message})
