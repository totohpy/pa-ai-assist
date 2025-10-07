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

# Page config
st.set_page_config(page_title="Design Assistant", page_icon="✨", layout="wide")

# --- Sidebar Configuration (empty to inherit from main page) ---
with st.sidebar:
    pass

# --- Apply Consistent Styling ---
st.markdown(
    """
     <style>
    /* --- Overall App Color Theme --- */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #e0f2f1;
    }
    [data-testid="stSidebar"] {
        background-color: #e0f2f1;
        width: 250px !important;
    }
    
    /* --- Flexbox layout for Sidebar --- */
    /* This targets the inner container of the sidebar */
    [data-testid="stSidebar"] > div:first-child {
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    /* This makes the navigation take up all available space, pushing the footer down */
    [data-testid="stSidebarNav"] {
        flex-grow: 1;
        margin-top: 20px; /* Move navigation down */
    }
    .sidebar-footer {
        width: 100%;
        padding: 1rem;
        text-align: center; /* Center the footer content */
    }

    /* Remove Streamlit's default top padding */
    .block-container {
        padding-top: 2rem;
    }

    /* --- Feature Box Styling (Main Page) --- */
    .feature-link { text-decoration: none !important; color: inherit !important; }
    .feature-link:hover { text-decoration: none !important; color: inherit !important; }
    .feature-box {
        background-color: #e0f2f1;
        padding: 1rem 1rem;
        border-radius: 20px;
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        border: 1px solid #d0e0df;
    }
    .feature-box:hover {
        transform: translateY(-10px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    .feature-box .emoji { font-size: 1.6rem; line-height: 1; }
    .feature-box h3 { margin-top: 0.7rem; margin-bottom: 0.4rem; font-size: 1.2rem; }
    .feature-box p { color: #6c757d; font-size: 0.85rem; }
    
    /* --- Style the sidebar navigation --- */
    div[data-testid="stSidebarNav"] > ul > li > a {
        padding: 18px 40px !important; /* Increased padding for more height */
        font-size: 20px !important;    /* Larger font size */
        margin-bottom: 10px;
        border-radius: 8px;
        color: #263238 !important;     /* Darker text for inactive links */
        background-color: #b2dfdb;     /* Light teal for inactive links */
        border: 1px solid #9dbdb9;
        font-weight: 500;
    }
    
    /* Style the ACTIVE page link */
    div[data-testid="stSidebarNav"] a[aria-current="page"] {
        background-color: #80cbc4;     /* Dark teal for active link */
        color: #FFFFFF !important;     /* White text for active link */
        font-weight: 600;
        border: 1px solid #00796b;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ----------------- Helper Functions -----------------
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
    nums = [int(str(x).split("-")[-1]) for x in df[col] if str(x).split("-")[-1].isdigit()]
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
def build_tfidf_index(_findings_df: pd.DataFrame):
    texts = (_findings_df["issue_title"].fillna("") + " " + _findings_df["issue_detail"].fillna("") + " " + _findings_df["cause_detail"].fillna("") + " " + _findings_df["recommendation"].fillna(""))
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
    nodes, edges = [], []
    styles = { "Objective": "#E6E6FA", "Input": "#a9def9", "Activity": "#e4c1f9", "Output": "#fcf6bd", "Outcome": "#d0f4de", "Impact": "#ff99c8"}
    sequence = ["Objective", "Input", "Activity", "Output", "Outcome", "Impact"]
    nodes_exist = []
    for i, item_type in enumerate(sequence):
        items_df = df[df['type'] == item_type]
        if not items_df.empty:
            header = item_type
            desc_lines = [f"• {row.get('description', '')} {row.get('target', '') or row.get('metric', '')} {row.get('unit', '')}".strip() for _, row in items_df.iterrows()]
            label = f"{header}\n\n" + "\n".join(desc_lines)
            nodes.append(Node(id=item_type, label=label, color=styles.get(item_type), shape="box", font={'face': 'Kanit', 'align': 'left'}, level=i))
            nodes_exist.append(item_type)
    if len(nodes_exist) > 1:
        for i in range(len(nodes_exist)-1):
            edges.append(Edge(source=nodes_exist[i], target=nodes_exist[i+1], dashes=False, color="#000000"))
    config = Config(width='100%', height=600, directed=True, physics=False, hierarchical={"enabled": True, "direction": "LR", "sortMethod": "directed"})
    return nodes, edges, config

# Initialize session state and variables
init_state()
plan = st.session_state["plan"]
logic_df = st.session_state["logic_items"]
methods_df = st.session_state["methods"]
kpis_df = st.session_state["kpis"]
risks_df = st.session_state["risks"]
audit_issues_df = st.session_state["audit_issues"]

# --- Main App UI ---
st.title("✨ Design Assistant")

with st.expander("💡 คำแนะนำการใช้งาน"):
    st.info("กรุณาระบุข้อมูล อย่างน้อย **ระบุ แผน & 6W2H** ส่วนใดส่วนหนึ่ง เพื่อค้นหาข้อตรวจพบที่ผ่านมาและให้ PA Assistant แนะนำ ได้แม่นยำที่สุด")

st.markdown("""
<style> 
    h1 { font-size: 36px !important; }
    body { font-family: 'Kanit', sans-serif; } 
    button[data-baseweb="tab"] { border-radius: 2px; padding: 8px 6px; margin: 0px; font-size: 16px; letter-spacing: 0.3px; font-weight: normal; color: white !important; border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: all 0.2s ease-in-out; }
    button[data-baseweb="tab"][aria-selected="true"] { box-shadow: 0 4px 12px rgba(0,0,0,0.25); transform: translateY(-2px); opacity: 1; }
    button[data-baseweb="tab"]:hover { transform: translateY(-1px); box-shadow: 0 3px 8px rgba(0,0,0,0.15); }
    div[data-baseweb="tab-list"] button:nth-of-type(-n+5) { background-color: #A93C2D; }
    div[data-baseweb="tab-list"] button:nth-of-type(6), div[data-baseweb="tab-list"] button:nth-of-type(7) { background-color: #4D8076; }
    div[data-baseweb="tab-list"] button:nth-of-type(8) { background-color: #4A6A8A; }
    div[data-baseweb="tab-list"] { border-bottom: none !important; margin-bottom: 15px; flex-wrap: wrap; gap: 2px; } 
    h4 { color: #007bff !important; border-bottom: 2px solid #e0e0e0; padding-bottom: 5px; } 
</style>
""", unsafe_allow_html=True)

tab_plan, tab_logic, tab_method, tab_kpi, tab_risk, tab_issue, tab_preview, tab_assist = st.tabs([
    "1.&nbsp;ระบุ แผน & 6W2H", "2.&nbsp;ระบุ Logic Model", "3.&nbsp;ระบุ Methods", 
    "4.&nbsp;ระบุ KPIs", "5.&nbsp;ระบุ Risks", "🔍 ค้นหาข้อตรวจพบที่ผ่านมา", 
    "📋 สรุปข้อมูล (Preview)", "✨ PA Assistant แนะนำประเด็น"
]) 

with tab_plan:
    st.subheader("ข้อมูลแผน - กรุณาระบุข้อมูล")
    with st.container(border=True):
        c1, c2, c3 = st.columns([2,2,1])
        with c1:
            plan["plan_title"] = st.text_input("ชื่อแผน/เรื่องที่จะตรวจ", plan["plan_title"])
            plan["program_name"] = st.text_input("ชื่อโครงการ/แผนงาน", plan["program_name"])
            plan["objectives"] = st.text_area("วัตถุประสงค์การตรวจ", plan["objectives"])
        with c2:
            plan["scope"] = st.text_area("ขอบเขตการตรวจ", plan["scope"])
            plan["assumptions"] = st.text_area("สมมติฐาน/ข้อจำกัดข้อมูล", plan["assumptions"])
        with c3:
            st.text_input("Plan ID", plan["plan_id"], disabled=True)
            plan["status"] = st.selectbox("สถานะ", ["Draft","Published"], index=0)

    st.divider()
    st.subheader("สรุปเรื่องที่ตรวจสอบ (6W2H)")
    with st.container(border=True):
        st.markdown("##### 🚀 สร้าง 6W2H อัตโนมัติด้วย AI")
        st.write("คัดลอกข้อความจากไฟล์มาวางในช่องด้านล่างนี้")
        uploaded_text = st.text_area("ระบุข้อความเพื่อให้ AI ช่วยสรุป 6W2H", height=200, key="uploaded_text")

        if st.button("🚀 สร้าง 6W2H จากข้อความ", type="primary", key="6w2h_button"):
            if not uploaded_text:
                st.error("กรุณาวางข้อความในช่องก่อน")
            elif not st.session_state.api_key_global:
                st.error("ยังไม่ได้ตั้งค่า API Key, กรุณาติดต่อผู้ดูแลระบบ")
            else:
                with st.spinner("กำลังประมวลผล..."):
                    try:
                        user_prompt = f"จากข้อความด้านล่างนี้ กรุณาสรุปและแยกแยะข้อมูลให้เป็น 6W2H ได้แก่ Who, Whom, What, Where, When, Why, How, และ How much โดยให้อยู่ในรูปแบบ key-value ที่ชัดเจน\nข้อความ:\n---\n{uploaded_text}\n---\nรูปแบบที่ต้องการ:\nWho: [ข้อความ]\nWhom: [ข้อความ]\nWhat: [ข้อความ]\nWhere: [ข้อความ]\nWhen: [ข้อความ]\nWhy: [ข้อความ]\nHow: [ข้อความ]\nHow Much: [ข้อความ]"
                        client = OpenAI(api_key=st.session_state.api_key_global, base_url="https://api.opentyphoon.ai/v1")
                        response = client.chat.completions.create(model="typhoon-v2.1-12b-instruct", messages=[{"role": "user", "content": user_prompt}], temperature=0.7, max_tokens=1024, top_p=0.9)
                        st.session_state["6w2h_output"] = response.choices[0].message.content
                        st.success("สร้าง 6W2H เรียบร้อยแล้ว! ผลลัพธ์แสดงอยู่ด้านล่าง"); st.balloons(); st.rerun()
                    except Exception as e:
                        st.error(f"เกิดข้อผิดพลาดในการเรียกใช้ AI: {e}")

        if st.session_state.get("6w2h_output"):
            st.markdown("---")
            with st.expander("คลิกเพื่อดู/ซ่อนผลลัพธ์จาก AI ล่าสุด", expanded=True):
                st.info("ตรวจสอบและคัดลอกข้อมูลด้านล่างนี้ไปวางในช่องที่เกี่ยวข้อง:")
                with st.container(border=True):
                    st.markdown(st.session_state["6w2h_output"])

    st.markdown("##### ⭐กรุณาระบุข้อมูล เพื่อนำไปใช้ประมวลผล")
    with st.container(border=True):
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            plan["who"] = st.text_input("Who (ใคร)", plan["who"], key="who_input")
            plan["whom"] = st.text_input("Whom (เพื่อใคร)", plan["whom"], key="whom_input")
            plan["what"] = st.text_input("What (ทำอะไร)", plan["what"], key="what_input")
        with cc2:
            plan["where"] = st.text_input("Where (ที่ไหน)", plan["where"], key="where_input")
            plan["when"] = st.text_input("When (เมื่อใด)", plan["when"], key="when_input")
            plan["why"] = st.text_area("Why (ทำไม)", plan["why"], key="why_input")
        with cc3:
            plan["how"] = st.text_area("How (อย่างไร)", plan["how"], key="how_input")
            plan["how_much"] = st.text_input("How much (เท่าไร)", plan["how_much"], key="how_much_input")

with tab_logic:
    st.subheader("ระบุ Logic Model")
    with st.expander("➕ เพิ่มรายการใหม่ใน Logic Model", expanded=True):
        with st.container(border=True):
            colA, colB, colC = st.columns(3)
            with colA:
                typ = st.selectbox("ประเภท", ["Objective", "Input", "Activity", "Output", "Outcome", "Impact"], key="logic_type")
                desc = st.text_input("คำอธิบาย/รายละเอียด", key="logic_desc")
            with colB:
                metric = st.text_input("ตัวชี้วัด (จำนวน)", key="logic_metric")
                unit = st.text_input("หน่วย", value="", key="logic_unit")
            with colC:
                target = st.text_input("เป้าหมาย", value="", key="logic_target")
                source = st.text_input("แหล่งข้อมูล", value="", key="logic_source")
            
            if st.button("เพิ่ม Logic Item", type="primary", key="add_logic_item_btn"):
                if desc:
                    new_row = pd.DataFrame([{"item_id": next_id("LG", logic_df, "item_id"), "plan_id": plan["plan_id"], "type": typ, "description": desc, "metric": metric, "unit": unit, "target": target, "source": source}])
                    st.session_state.logic_items = pd.concat([logic_df, new_row], ignore_index=True)
                    st.success("เพิ่มข้อมูลเรียบร้อยแล้ว"); st.rerun()
                else:
                    st.warning("กรุณากรอก 'คำอธิบาย/รายละเอียด' ก่อนทำการเพิ่ม")
    
    st.markdown("---")
    st.markdown("##### 📝 ตาราง Logic Model (สามารถแก้ไขหรือลบแถวได้โดยตรง)")
    st.session_state.logic_items = st.data_editor(logic_df, column_config={"type": st.column_config.SelectboxColumn("ประเภท", options=["Objective", "Input", "Activity", "Output", "Outcome", "Impact"], required=True), "description": st.column_config.TextColumn("คำอธิบาย/รายละเอียด", required=True), "metric": st.column_config.TextColumn("ตัวชี้วัด (จำนวน)"), "target": st.column_config.TextColumn("เป้าหมาย"), "unit": st.column_config.TextColumn("หน่วย"), "source": st.column_config.TextColumn("แหล่งข้อมูล"), "item_id": st.column_config.TextColumn("ID", disabled=True), "plan_id": st.column_config.TextColumn("Plan ID", disabled=True)}, use_container_width=True, hide_index=True, key="logic_editor_main")
    
    cols = st.columns([0.85, 0.15])
    with cols[1]:
        if st.button("🧹 ล้างทั้งหมด (Reset)", use_container_width=True):
            st.session_state.logic_items = pd.DataFrame(columns=logic_df.columns)
            st.rerun()

    st.markdown("---")
    st.subheader("📊 Flowchart Logic Model (ลากจัดวางได้)")
    with st.container(border=True):
        if not logic_df.empty:
            try:
                nodes, edges, config = create_interactive_flowchart(logic_df)
                agraph(nodes=nodes, edges=edges, config=config)
            except Exception as e:
                st.error(f"ไม่สามารถสร้าง Flowchart ได้: {e}")
        else:
            st.info("กรุณาเพิ่มข้อมูลในฟอร์มด้านบนเพื่อสร้าง Flowchart")

with tab_method:
    st.subheader("ระบุวิธีการเก็บข้อมูล")
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
        quality = col3.text_input("ข้อกำหนดคุณภาพ", value="ถูกต้อง/ทันเวลา", key="kpi_quality")
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
            hypothesis = c3.text_input("สมมติฐานที่ต้องทดสอบ")
            if st.button("เพิ่ม Risk", type="primary", key="add_risk_btn"):
                new_row = pd.DataFrame([{"risk_id": next_id("RSK", risks_df, "risk_id"),"plan_id": plan["plan_id"],"description": desc, "category": category,"likelihood": likelihood, "impact": impact,"mitigation": mitigation, "hypothesis": hypothesis}])
                st.session_state["risks"] = pd.concat([risks_df, new_row], ignore_index=True)
                st.rerun()

with tab_issue:
    st.subheader("🔎 แนะนำประเด็นตรวจสอบจากรายงานเก่า")
    with st.expander("อัปโหลดและจัดการฐานข้อมูลข้อตรวจพบ"):
        st.write("คุณสามารถอัปโหลดไฟล์ .csv หรือ .xlsx ที่มีข้อมูลข้อตรวจพบเพื่อใช้ในการค้นหา ถ้าไม่มีจะค้นหาภายในคลังข้อมูล")
        st.download_button(label="⬇️ ดาวน์โหลดไฟล์แม่แบบ FindingsLibrary.xlsx", data=create_excel_template(), file_name="FindingsLibrary.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        uploaded = st.file_uploader("อัปโหลด FindingsLibrary.csv หรือ .xlsx", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    
    findings_df = load_findings(uploaded=uploaded)
    
    if findings_df.empty:
        st.info("ไม่พบข้อมูล Findings โปรดอัปโหลดไฟล์")
    else:
        st.success(f"พบข้อมูล Findings ทั้งหมด {len(findings_df)} รายการ")
        vec, X = build_tfidf_index(findings_df)
        seed = f"Who:{plan.get('who','')} What:{plan.get('what','')} Where:{plan.get('where','')} When:{plan.get('when','')} Why:{plan.get('why','')} Whom:{plan.get('whom','')} How:{plan.get('how','')} Outputs:{' | '.join(logic_df[logic_df['type']=='Output']['description'].tolist())} Outcomes:{' | '.join(logic_df[logic_df['type']=='Outcome']['description'].tolist())} Objective:{' | '.join(logic_df[logic_df['type']=='Objective']['description'].tolist())}"
        def refresh_query_text(new_seed):
            st.session_state["issue_query_text"] = new_seed
            st.session_state["ref_seed"] = new_seed 
        if "issue_query_text" not in st.session_state or not st.session_state["issue_query_text"]:
            st.session_state["issue_query_text"] = seed
            st.session_state["ref_seed"] = seed
        elif st.session_state.get("ref_seed") != seed and st.session_state.get("issue_query_text") == st.session_state.get("ref_seed"):
            st.session_state["issue_query_text"] = seed
            st.session_state["ref_seed"] = seed 
        c_query_area, c_refresh_btn = st.columns([6, 1])
        with c_query_area:
            query_text = st.text_area("**ข้อมูล (Context) ที่ใช้ค้นหา (แก้ไขได้):**", st.session_state["issue_query_text"], height=140, key="issue_query_text")
        with c_refresh_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            st.button("🔄", on_click=refresh_query_text, args=(seed,), help="คลิกเพื่ออัปเดตช่องค้นหาด้วยข้อมูลที่กรอกจากหน้าอื่นล่าสุด", type="secondary")
        top_k_slider = st.slider("ปรับจำนวนผลลัพธ์:", min_value=1, max_value=20, value=8)
        if st.button("ค้นหาประเด็นที่ใกล้เคียง", type="primary", key="search_button_fix"):
            st.session_state["issue_results"] = search_candidates(st.session_state.get("issue_query_text", seed), findings_df, vec, X, top_k=top_k_slider)
            st.success(f"พบประเด็นที่เกี่ยวข้อง {len(st.session_state['issue_results'])} รายการ")
        results = st.session_state.get("issue_results", pd.DataFrame())
        if not results.empty:
            st.divider(); st.subheader("ผลลัพธ์การค้นหา")
            for i, row in results.reset_index(drop=True).iterrows():
                with st.container(border=True):
                    title_txt = row.get("issue_title", "(ไม่มีชื่อประเด็น)")
                    year_txt = int(row["year"]) if "year" in row and str(row["year"]).isdigit() else row.get("year", "-")
                    st.markdown(f"**{title_txt}** \nหน่วย: {row.get('unit', '-')} • โครงการ: {row.get('program', '-')} • ปี: {year_txt}")
                    st.caption(f"สาเหตุ: *{row.get('cause_category', '-')}* — {row.get('cause_detail', '-')}")
                    with st.expander("รายละเอียด/ข้อเสนอแนะเดิม"):
                        st.write(row.get("issue_detail", "-")); st.caption("ข้อเสนอแนะเดิม: " + (row.get("recommendation", "") or "-"))
                        st.markdown(f"**ผลกระทบ:** {row.get('outcomes_impact','-')}  •  <span style='color:red;'>**คะแนนความเกี่ยวข้อง**</span>: {row.get('score',0):.3f} (<span style='color:blue;'>**Similarity**</span>={row.get('sim_score',0):.3f})", unsafe_allow_html=True)
                        st.caption("💡 **คำอธิบาย:** คะแนนความเกี่ยวข้อง = ความคล้ายคลึงข้อความ + ความรุนแรง + ความใหม่")
                    c1, c2 = st.columns([3,1])
                    with c1:
                        st.text_area("เหตุผลที่ควรตรวจ", key=f"rat_{i}", value=f"อ้างอิงกรณีเดิม ปี {year_txt} | หน่วย: {row.get('unit', '-')}")
                        st.text_input("KPI ที่เกี่ยว (ถ้ามี)", key=f"kpi_{i}"); st.text_input("วิธีเก็บข้อมูลที่เสนอ", key=f"mth_{i}", value="สัมภาษณ์/สังเกต/ตรวจเอกสาร")
                    with c2:
                        if st.button("➕ เพิ่มเป็นประเด็นตรวจสอบ", key=f"add_{i}", type="secondary"):
                            new_row = pd.DataFrame([{"issue_id": next_id("ISS", audit_issues_df, "issue_id"),"plan_id": plan.get("plan_id",""),"title": title_txt,"rationale": st.session_state.get(f"rat_{i}", ""),"linked_kpi": st.session_state.get(f"kpi_{i}", ""),"proposed_methods": st.session_state.get(f"mth_{i}", ""),"source_finding_id": row.get("finding_id", ""),"issue_detail": row.get("issue_detail", ""),"recommendation": row.get("recommendation", "")}])
                            st.session_state["audit_issues"] = pd.concat([audit_issues_df, new_row], ignore_index=True)
                            st.success("เพิ่มประเด็นเข้าแผนแล้ว ✅"); st.rerun()
        if not st.session_state.get("issue_results", pd.DataFrame()).empty: st.divider()
        st.markdown("### ประเด็นที่เพิ่มเข้าแผน"); st.dataframe(st.session_state["audit_issues"], use_container_width=True, hide_index=True)

with tab_preview:
    st.subheader("สรุปแผน (Preview)")
    with st.container(border=True):
        st.markdown(f"**Plan ID:** {plan['plan_id']}  \n**ชื่อแผน:** {plan['plan_title']}  \n**โครงการ:** {plan['program_name']}  \n**หน่วยรับตรวจ:** {plan['who']}")
    st.markdown("### สรุปเรื่องที่ตรวจสอบ (จาก 6W2H)")
    with st.container(border=True):
        st.markdown(f"- **Who**: {plan['who']}\n- **Whom**: {plan['whom']}\n- **What**: {plan['what']}\n- **Where**: {plan['where']}\n- **When**: {plan['when']}\n- **Why**: {plan['why']}\n- **How**: {plan['how']}\n- **How much**: {plan['how_much']}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Logic Model"); st.dataframe(st.session_state["logic_items"], use_container_width=True, hide_index=True)
        df_download_link(st.session_state["logic_items"], "logic_items.csv", "⬇️ ดาวน์โหลด Logic Items (CSV)")
    with c2:
        st.markdown("### Methods"); st.dataframe(st.session_state["methods"], use_container_width=True, hide_index=True)
        df_download_link(st.session_state["methods"], "methods.csv", "⬇️ ดาวน์โหลด Methods (CSV)")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### KPIs"); st.dataframe(st.session_state["kpis"], use_container_width=True, hide_index=True)
        df_download_link(st.session_state["kpis"], "kpis.csv", "⬇️ ดาวน์โหลด KPIs (CSV)")
    with c4:
        st.markdown("### Risks"); st.dataframe(st.session_state["risks"], use_container_width=True, hide_index=True)
        df_download_link(st.session_state["risks"], "risks.csv", "⬇️ ดาวน์โหลด Risks (CSV)")
    st.markdown("### ประเด็นตรวจสอบที่เพิ่มเข้ามา")
    if not st.session_state["audit_issues"].empty:
        display_issues_df = st.session_state["audit_issues"].copy().rename(columns={"issue_id": "รหัสประเด็น", "title": "ชื่อประเด็น","rationale": "เหตุผล", "issue_detail": "รายละเอียด","recommendation": "ข้อเสนอแนะ"})
        st.dataframe(display_issues_df[["รหัสประเด็น", "ชื่อประเด็น", "เหตุผล", "รายละเอียด", "ข้อเสนอแนะ"]], use_container_width=True, hide_index=True)
    else: st.info("ยังไม่มีประเด็นตรวจสอบที่เพิ่มเข้ามาในแผน")
    if not st.session_state["audit_issues"].empty: df_download_link(st.session_state["audit_issues"], "audit_issues.csv", "⬇️ ดาวน์โหลด Audit Issues (CSV)")
    st.divider(); plan_df = pd.DataFrame([plan]); df_download_link(plan_df, "plan.csv", "⬇️ ดาวน์โหลด Plan (CSV)")
    st.success("พร้อมเชื่อมต่อ 🤖 PA Assistant เพื่อแนะนำประเด็นตรวจสอบ ✨✨")

with tab_assist:
    st.subheader("💡 PA Assistant (AI/LLM)")
    st.write("🤖 สร้างคำแนะนำประเด็นตรวจสอบจาก AI")
    if st.button("🚀 สร้างคำแนะนำจาก AI", type="primary", key="llm_assist_button"):
        if not st.session_state.api_key_global: st.error("กรุณาตั้งค่า API Key ใน sidebar ก่อน")
        else:
            with st.spinner("กำลังสร้างคำแนะนำ..."):
                try:
                    issues_for_llm = st.session_state['audit_issues'][['title', 'rationale']]
                    plan_summary = f"ชื่อแผน/เรื่องที่จะตรวจ: {plan['plan_title']}\nชื่อโครงการ/แผนงาน: {plan['program_name']}\nวัตถุประสงค์: {plan['objectives']}\nขอบเขต: {plan['scope']}\nสมมติฐาน/ข้อจำกัด: {plan['assumptions']}\n---\n6W2H:\nใคร (Who): {plan['who']}\nถึงใคร (Whom): {plan['whom']}\nทำอะไร (What): {plan['what']}\nที่ไหน (Where): {plan['where']}\nเมื่อใด (When): {plan['when']}\nทำไม (Why): {plan['why']}\nอย่างไร (How): {plan['how']}\nเท่าไร (How much): {plan['how_much']}\n---\nLogic Model:\n{st.session_state['logic_items'].to_string()}\n---\nประเด็นจากรายงานเก่า:\n{issues_for_llm.to_string()}"
                    user_prompt = f"จากข้อมูลแผนการตรวจสอบด้านล่างนี้ กรุณาช่วยสร้างคำแนะนำ 3 อย่าง:\n1. ประเด็นตรวจสอบที่ควรให้ความสำคัญ อาจอ้างอิงถึงประเด็นเก่า, ข้อตรวจพบ, หรือสถานการณ์ปัจจุบัน พร้อมเหตุผล\n2. ข้อตรวจพบที่คาดว่าจะพบ (พร้อมระบุโอกาสที่จะเจอ: สูง/กลาง/ต่ำ) พร้อมเหตุผล\n3. ร่างรายงานตรวจสอบ วิเคราะห์ผลกระทบและสาเหตุของข้อตรวจพบที่คาดว่าจะพบ\n---\n{plan_summary}\n---\nกรุณาสร้างคำตอบตามรูปแบบนี้เท่านั้น:\n<ประเด็นตรวจสอบที่ควรให้ความสำคัญ>\n[ข้อความส่วนที่ 1]\n</ประเด็นตรวจสอบที่ควรให้ความสำคัญ>\n\n<ข้อตรวจพบที่คาดว่าจะพบ>\n[ข้อความส่วนที่ 2]\n</ข้อตรวจพบที่คาดว่าจะพบ>\n\n<ร่างรายงานตรวจสอบ>\n[ข้อความส่วนที่ 3]\n</ร่างรายงานตรวจสอบ>"
                    client = OpenAI(api_key=st.session_state.api_key_global, base_url="https://api.opentyphoon.ai/v1")
                    messages = [{"role": "system", "content": "คุณคือผู้เชี่ยวชาญด้านการตรวจสอบผลสัมฤทธิ์ (Performance Auditing)"}, {"role": "user", "content": user_prompt}]
                    response = client.chat.completions.create(model="typhoon-v2.1-12b-instruct", messages=messages, temperature=0.7, max_tokens=2048)
                    full_response = response.choices[0].message.content
                    issue_start = full_response.find("<ประเด็นตรวจสอบที่ควรให้ความสำคัญ>") + len("<ประเด็นตรวจสอบที่ควรให้ความสำคัญ>"); issue_end = full_response.find("</ประเด็นตรวจสอบที่ควรให้ความสำคัญ>"); st.session_state["gen_issues"] = full_response[issue_start:issue_end].strip()
                    finding_start = full_response.find("<ข้อตรวจพบที่คาดว่าจะพบ>") + len("<ข้อตรวจพบที่คาดว่าจะพบ>"); finding_end = full_response.find("</ข้อตรวจพบที่คาดว่าจะพบ>"); st.session_state["gen_findings"] = full_response[finding_start:finding_end].strip()
                    report_start = full_response.find("<ร่างรายงานตรวจสอบ>") + len("<ร่างรายงานตรวจสอบ>"); report_end = full_response.find("</ร่างรายงานตรวจสอบ>"); st.session_state["gen_report"] = full_response[report_start:report_end].strip()
                    st.success("สร้างคำแนะนำจาก AI เรียบร้อยแล้ว ✅")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการเรียกใช้ AI: {e}"); st.session_state["gen_issues"] = ""; st.session_state["gen_findings"] = ""; st.session_state["gen_report"] = ""
    st.subheader("ผลลัพธ์จาก AI")
    with st.expander("1. ประเด็นตรวจสอบที่ควรให้ความสำคัญ", expanded=True):
        st.write(st.session_state.get('gen_issues', "ยังไม่มีข้อมูล กด 'สร้างคำแนะนำจาก AI' เพื่อเริ่มต้น"))
    with st.expander("2. ข้อตรวจพบที่คาดว่าจะพบ"):
        st.write(st.session_state.get('gen_findings', "ยังไม่มีข้อมูล"))
    with st.expander("3. ร่างรายงานตรวจสอบ (Preview)"):
        st.write(st.session_state.get('gen_report', "ยังไม่มีข้อมูล"))
