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

# Page setup
st.set_page_config(page_title="Planning Studio (+ Findings Suggestions)", page_icon="🧭", layout="wide")

# ----------------- ⚙️ Central Settings -----------------
with st.sidebar:
    st.title("⚙️ Central Settings")
    st.info("The API Key is now set by the administrator.")

    # --- Load API Key from Streamlit Secrets ---
    try:
        st.session_state.api_key_global = st.secrets["api_key"]
    except KeyError:
        st.session_state.api_key_global = ""
        st.warning("AI features are unavailable. Please contact the administrator.")
    except Exception as e:
        st.session_state.api_key_global = ""
        st.error(f"Error loading API Key: {e}")

    st.markdown("---")
    st.markdown("PA Planning Studio By PAO1 Audit Intelligence Nexus")

# ----------------- Chatbot Functions -----------------
MAX_CHARS_LIMIT = 200000

@st.cache_data(show_spinner=False)
def load_local_documents(folder_path="Doc"):
    """Reads all files from the data store."""
    text = ""
    if not os.path.isdir(folder_path):
        return text 

    try:
        files_in_doc = os.listdir(folder_path)
        progress_bar = st.sidebar.progress(0, text=f"Loading documents... (0/{len(files_in_doc)})")
        for i, filename in enumerate(files_in_doc):
            if len(text) >= MAX_CHARS_LIMIT:
                st.warning(f"Data limit reached ({MAX_CHARS_LIMIT:,} characters).")
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
            
            progress_bar.progress((i + 1) / len(files_in_doc), text=f"Loading documents... ({i+1}/{len(files_in_doc)})")
        progress_bar.empty()
                
    except Exception as e:
        st.error(f"Error accessing data store: {e}")
    
    return text[:MAX_CHARS_LIMIT]

def process_documents(files, source_type, limit, current_len=0):
    """Function to read text from uploaded files."""
    text = ""
    for file in files:
        if current_len + len(text) >= limit:
            st.warning(f"Max character limit ({limit:,}) reached. Some files may not be processed.")
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
            st.error(f"Error reading file {file.name}: {e}")
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
    ss.setdefault("6w2h_output", "") # <-- FIX: Added this line
    
    ss.setdefault('chatbot_messages', [{"role": "assistant", "content": "Hello! I am your PA Chat assistant."}])
    ss.setdefault('doc_context_uploaded', "")
    ss.setdefault('last_uploaded_files', set())

    if 'doc_context_local' not in ss:
        ss.doc_context_local = load_local_documents()
        if ss.doc_context_local and os.path.isdir('Doc'):
             ss.chatbot_messages.append({"role": "assistant", "content": f"I have loaded {len(os.listdir('Doc'))} documents as a knowledge base."})

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
        except Exception as e: st.error(f"Error reading FindingsLibrary.csv: {e}")
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'): uploaded_df = pd.read_csv(uploaded)
            elif uploaded.name.endswith(('.xlsx', '.xls')):
                xls = pd.ExcelFile(uploaded)
                sheet_name = "Data" if "Data" in xls.sheet_names else 0
                uploaded_df = pd.read_excel(xls, sheet_name=sheet_name)
            if not uploaded_df.empty:
                findings_df = pd.concat([findings_df, uploaded_df], ignore_index=True)
                st.success(f"Uploaded '{uploaded.name}' and merged with the existing database.")
        except Exception as e: st.error(f"Error reading uploaded file: {e}")
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

with st.expander("💡 Usage Guide"):
    st.info(
        "Please provide information in at least the **Plan & 6W2H** section to get the most accurate suggestions from the PA Assistant."
    )

st.markdown("""
<style> 
    body { font-family: 'Kanit', sans-serif; } 
    button[data-baseweb="tab"] {
        border-radius: 10px;
        padding: 6px 6px;
        margin: 1px;
        font-weight: normal;
        color: white !important; 
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s ease-in-out;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        transform: translateY(-2px);
        opacity: 0.75;
    }
    button[data-baseweb="tab"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
    }
    div[data-baseweb="tab-list"] button:nth-of-type(-n+5) { background-color: #A93C2D; }
    div[data-baseweb="tab-list"] button:nth-of-type(6), 
    div[data-baseweb="tab-list"] button:nth-of-type(7) { background-color: #4D8076; }
    div[data-baseweb="tab-list"] button:nth-of-type(8) { background-color: #4A6A8A; }
    div[data-baseweb="tab-list"] button:nth-of-type(9) { background-color: #4A6A8A; }
    div[data-baseweb="tab-list"] { 
        border-bottom: none !important; 
        margin-bottom: 15px; 
        flex-wrap: wrap; 
        gap: 2px;
    } 
    h4 { 
        color: #007bff !important; 
        border-bottom: 2px solid #e0e0e0; 
        padding-bottom: 5px; 
    } 
</style>
""", unsafe_allow_html=True)

tab_plan, tab_logic, tab_method, tab_kpi, tab_risk, tab_issue, tab_preview, tab_assist, tab_chatbot = st.tabs(["1. Specify Plan & 6W2H", "2. Specify Logic Model", "3. Specify Methods", "4. Specify KPIs", "5. Specify Risks", "6. 🔍 Search Past Findings", "7. 📋 Preview", "8. ✨ PA Assistant", "9. 💬 PA Chat"]) 

with tab_plan:
    st.subheader("Plan Information - Please provide details")
    with st.container(border=True):
        c1, c2, c3 = st.columns([2,2,1])
        with c1:
            plan["plan_title"] = st.text_input("Plan Name / Audit Topic", plan["plan_title"])
            plan["program_name"] = st.text_input("Project / Program Name", plan["program_name"])
            plan["objectives"] = st.text_area("Audit Objectives", plan["objectives"])
        with c2:
            plan["scope"] = st.text_area("Audit Scope", plan["scope"])
            plan["assumptions"] = st.text_area("Assumptions / Data Limitations", plan["assumptions"])
        with c3:
            st.text_input("Plan ID", plan["plan_id"], disabled=True)
            plan["status"] = st.selectbox("Status", ["Draft","Published"], index=0)

    st.divider()
    st.subheader("Audit Topic Summary (6W2H)")
    with st.container(border=True):
        st.markdown("##### 🚀 Automatically generate 6W2H with AI")
        st.write("Copy text from your file and paste it in the box below.")
        uploaded_text = st.text_area("Enter text about the topic for AI to summarize into 6W2H", height=200, key="uploaded_text")

        if st.button("🚀 Generate 6W2H from Text", type="primary", key="6w2h_button"):
            if not uploaded_text:
                st.error("Please paste text in the box first.")
            elif not st.session_state.api_key_global:
                st.error("API Key not set. Please contact the administrator.")
            else:
                with st.spinner("Processing..."):
                    try:
                        user_prompt = f"""
From the text below, please summarize and extract the information into the 6W2H format: Who, Whom, What, Where, When, Why, How, and How much, in a clear key-value format.
Text:
---
{uploaded_text}
---
Desired Format:
Who: [Text]
Whom: [Text]
What: [Text]
Where: [Text]
When: [Text]
Why: [Text]
How: [Text]
How Much: [Text]
"""
                        client = OpenAI(
                            api_key=st.session_state.api_key_global,
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
                        
                        # --- FIX: Save the output to session state ---
                        st.session_state["6w2h_output"] = llm_output

                        st.success("6W2H generated successfully! The result is displayed below.")
                        st.balloons()
                        st.rerun() # Rerun to display the saved state immediately

                    except Exception as e:
                        st.error(f"Error calling AI: {e}")

        # --- FIX: Display the saved output outside the button's 'if' block ---
        if st.session_state.get("6w2h_output"):
            st.markdown("---")
            with st.expander("Click to view/hide the latest AI result", expanded=True):
                st.info("Review and copy the information below to the relevant fields:")
                with st.container(border=True):
                    st.markdown(st.session_state["6w2h_output"])

    st.markdown("##### ⭐ Please provide the following information for processing")
    with st.container(border=True):
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.session_state.plan["who"] = st.text_input("Who", key="who_input")
            st.session_state.plan["whom"] = st.text_input("Whom", key="whom_input")
            st.session_state.plan["what"] = st.text_input("What", key="what_input")
            st.session_state.plan["where"] = st.text_input("Where", key="where_input")
        with cc2:
            st.session_state.plan["when"] = st.text_input("When", key="when_input")
            st.session_state.plan["why"] = st.text_area("Why", key="why_input")
        with cc3:
            st.session_state.plan["how"] = st.text_area("How", key="how_input")
            st.session_state.plan["how_much"] = st.text_input("How much", key="how_much_input")

with tab_logic:
    st.subheader("Specify Logic Model: Input → Activities → Output → Outcome → Impact")
    st.dataframe(logic_df, use_container_width=True, hide_index=True)
    with st.expander("➕ Add Item to Logic Model"):
        with st.container(border=True):
            colA, colB, colC = st.columns(3)
            typ = colA.selectbox("Type", ["Input","Activity","Output","Outcome","Impact"])
            desc = colA.text_input("Description / Details")
            metric = colA.text_input("Metric (e.g., number, proportion)")
            unit = colB.text_input("Unit", value="unit", key="logic_unit")
            target = colB.text_input("Target", value="", key="logic_target")
            source = colC.text_input("Data Source", value="", key="logic_source")
            if st.button("Add Logic Item", type="primary", key="add_logic_item_btn"):
                new_row = pd.DataFrame([{"item_id": next_id("LG", logic_df, "item_id"),"plan_id": plan["plan_id"],"type": typ, "description": desc, "metric": metric,"unit": unit, "target": target, "source": source}])
                st.session_state["logic_items"] = pd.concat([logic_df, new_row], ignore_index=True)
                st.rerun()

with tab_method:
    st.subheader("Specify Data Collection Methods")
    st.dataframe(methods_df, use_container_width=True, hide_index=True)
    with st.expander("➕ Add Method"):
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            mtype = c1.selectbox("Type", ["observe","interview","questionnaire","document"])
            tool_ref = c1.text_input("Tool Reference ID", value="")
            sampling = c1.text_input("Sampling Method", value="")
            questions = c2.text_area("Key Questions / Topics")
            linked_issue = c2.text_input("Link to Audit Issue", value="")
            data_source = c3.text_input("Data Source", value="", key="method_data_source")
            frequency = c3.text_input("Frequency", value="One-time", key="method_frequency")
            if st.button("Add Method", type="primary", key="add_method_btn"):
                new_row = pd.DataFrame([{"method_id": next_id("MT", methods_df, "method_id"),"plan_id": plan["plan_id"],"type": mtype, "tool_ref": tool_ref, "sampling": sampling,"questions": questions, "linked_issue": linked_issue,"data_source": data_source, "frequency": frequency}])
                st.session_state["methods"] = pd.concat([methods_df, new_row], ignore_index=True)
                st.rerun()

with tab_kpi:
    st.subheader("Specify KPIs")
    st.dataframe(kpis_df, use_container_width=True, hide_index=True)
    with st.expander("➕ Add KPI Manually"):
        col1, col2, col3 = st.columns(3)
        level = col1.selectbox("Level", ["output","outcome"])
        name = col1.text_input("KPI Name")
        formula = col1.text_input("Formula / Definition")
        numerator = col2.text_input("Numerator")
        denominator = col2.text_input("Denominator")
        unit = col2.text_input("Unit", value="%", key="kpi_unit")
        baseline = col3.text_input("Baseline", value="")
        target = col3.text_input("Target", value="")
        freq = col3.text_input("Frequency", value="Quarterly")
        data_src = col3.text_input("Data Source", value="", key="kpi_data_source")
        quality = col3.text_input("Data Quality Requirements", value="Accurate/Timely", key="kpi_quality")
        if st.button("Add KPI", type="primary", key="add_kpi_btn"):
            new_row = pd.DataFrame([{"kpi_id": next_id("KPI", kpis_df, "kpi_id"),"plan_id": plan["plan_id"],"level": level, "name": name, "formula": formula,"numerator": numerator, "denominator": denominator, "unit": unit,"baseline": baseline, "target": target, "frequency": freq,"data_source": data_src, "quality_requirements": quality}])
            st.session_state["kpis"] = pd.concat([kpis_df, new_row], ignore_index=True)
            st.rerun()

with tab_risk:
    st.subheader("Specify Risks")
    st.dataframe(risks_df, use_container_width=True, hide_index=True)
    with st.expander("➕ Add Risk"):
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            desc = c1.text_area("Risk Description")
            category = c1.selectbox("Category", ["policy","org","data","process","people"])
            likelihood = c2.select_slider("Likelihood (1-5)", options=[1,2,3,4,5], value=3)
            impact = c2.select_slider("Impact (1-5)", options=[1,2,3,4,5], value=3)
            mitigation = c3.text_area("Mitigation Measures")
            hypothesis = c3.text_input("Hypothesis to Test")
            if st.button("Add Risk", type="primary", key="add_risk_btn"):
                new_row = pd.DataFrame([{"risk_id": next_id("RSK", risks_df, "risk_id"),"plan_id": plan["plan_id"],"description": desc, "category": category,"likelihood": likelihood, "impact": impact,"mitigation": mitigation, "hypothesis": hypothesis}])
                st.session_state["risks"] = pd.concat([risks_df, new_row], ignore_index=True)
                st.rerun()

with tab_issue:
    st.subheader("🔎 Suggest Audit Issues from Past Reports (Findings Suggestions)")
    with st.expander("Upload and Manage Findings Library"):
        st.write("You can upload a .csv or .xlsx file with past findings to use for searching. Download the template file to get started.")
        st.download_button(
            label="⬇️ Download FindingsLibrary.xlsx Template",
            data=create_excel_template(),
            file_name="FindingsLibrary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        uploaded = st.file_uploader("Upload FindingsLibrary.csv or .xlsx", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    
    findings_df = load_findings(uploaded=uploaded)
    
    if findings_df.empty:
        st.info("No Findings data found. Please upload a file or ensure FindingsLibrary.csv is available.")
    else:
        st.success(f"Found {len(findings_df)} total Findings records.")
        vec, X = build_tfidf_index(findings_df)
        
        seed = f"""
Who:{plan.get('who','')} What:{plan.get('what','')} Where:{plan.get('where','')}
When:{plan.get('when','')} Why:{plan.get('why','')} Whom:{plan.get('whom','')} How:{plan.get('how','')}
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
                "**Search Context (Editable):**", 
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
                help="Click to update the search box with the latest information",
                type="secondary"
            )
        
        top_k_slider = st.slider("Adjust number of results to display:", min_value=1, max_value=20, value=8)

        if st.button("Find Similar Issues", type="primary", key="search_button_fix"):
            search_value = st.session_state.get("issue_query_text", seed)
            results = search_candidates(search_value, findings_df, vec, X, top_k=top_k_slider)
            st.session_state["issue_results"] = results
            st.success(f"Found {len(results)} related issues.")
            
        results = st.session_state.get("issue_results", pd.DataFrame())
        
        if not results.empty:
            st.divider()
            st.subheader("Search Results")
            for i, row in results.reset_index(drop=True).iterrows():
                with st.container(border=True):
                    title_txt = row.get("issue_title", "(No issue title)")
                    year_txt = int(row["year"]) if "year" in row and str(row["year"]).isdigit() else row.get("year", "-")
                    st.markdown(f"**{title_txt}** \nUnit: {row.get('unit', '-')} • Program: {row.get('program', '-')} • Year: {year_txt}")
                    st.caption(f"Cause: *{row.get('cause_category', '-')}* — {row.get('cause_detail', '-')}")
    
                    with st.expander("Original Details / Recommendation"):
                        st.write(row.get("issue_detail", "-"))
                        st.caption("Original Recommendation: " + (row.get("recommendation", "") or "-"))
                        st.markdown(f"**Potential Impact:** {row.get('outcomes_impact','-')}  •  <span style='color:red;'>**Relevance Score**</span>: {row.get('score',0):.3f} (<span style='color:blue;'>**Similarity Score**</span>={row.get('sim_score',0):.3f})", unsafe_allow_html=True)
                        st.caption("💡 **Explanation:** **Relevance Score** = Text Similarity + Severity + Recency")
    
                    c1, c2 = st.columns([3,1])
                    with c1:
                        st.text_area("Rationale for this audit", key=f"rat_{i}", value=f"Reference to case from Year {year_txt} | Unit: {row.get('unit', '-')}")
                        st.text_input("Related KPI (if any)", key=f"kpi_{i}")
                        st.text_input("Proposed method", key=f"mth_{i}", value="Interview/Observation/Document Review")
    
                    with c2:
                        if st.button("➕ Add as Audit Issue", key=f"add_{i}", type="secondary"):
                            new_row = pd.DataFrame([{"issue_id": next_id("ISS", audit_issues_df, "issue_id"),"plan_id": plan.get("plan_id",""),"title": title_txt,"rationale": st.session_state.get(f"rat_{i}", ""),"linked_kpi": st.session_state.get(f"kpi_{i}", ""),"proposed_methods": st.session_state.get(f"mth_{i}", ""),"source_finding_id": row.get("finding_id", ""),"issue_detail": row.get("issue_detail", ""),"recommendation": row.get("recommendation", "")}])
                            st.session_state["audit_issues"] = pd.concat([audit_issues_df, new_row], ignore_index=True)
                            st.success("Issue added to the plan. ✅")
                            st.rerun()
                            
        if not st.session_state.get("issue_results", pd.DataFrame()).empty:
            st.divider()
        st.markdown("### Issues Added to Plan")
        st.dataframe(st.session_state["audit_issues"], use_container_width=True, hide_index=True)

with tab_preview:
    st.subheader("Plan Summary (Preview)")
    with st.container(border=True):
        st.markdown(f"**Plan ID:** {plan['plan_id']}  \n**Plan Name:** {plan['plan_title']}  \n**Project:** {plan['program_name']}  \n**Auditee:** {plan['who']}")
    st.markdown("### Audit Topic Summary (from 6W2H)")
    with st.container(border=True):
        st.markdown(f"- **Who**: {plan['who']}\n- **Whom**: {plan['whom']}\n- **What**: {plan['what']}\n- **Where**: {plan['where']}\n- **When**: {plan['when']}\n- **Why**: {plan['why']}\n- **How**: {plan['how']}\n- **How much**: {plan['how_much']}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Logic Model")
        st.dataframe(st.session_state["logic_items"], use_container_width=True, hide_index=True)
        df_download_link(st.session_state["logic_items"], "logic_items.csv", "⬇️ Download Logic Items (CSV)")
    with c2:
        st.markdown("### Methods")
        st.dataframe(st.session_state["methods"], use_container_width=True, hide_index=True)
        df_download_link(st.session_state["methods"], "methods.csv", "⬇️ Download Methods (CSV)")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### KPIs")
        st.dataframe(st.session_state["kpis"], use_container_width=True, hide_index=True)
        df_download_link(st.session_state["kpis"], "kpis.csv", "⬇️ Download KPIs (CSV)")
    with c4:
        st.markdown("### Risks")
        st.dataframe(st.session_state["risks"], use_container_width=True, hide_index=True)
        df_download_link(st.session_state["risks"], "risks.csv", "⬇️ Download Risks (CSV)")
    st.markdown("### Audit Issues Added")
    if not st.session_state["audit_issues"].empty:
        display_issues_df = st.session_state["audit_issues"].copy().rename(columns={"issue_id": "Issue ID", "title": "Issue Title","rationale": "Rationale", "issue_detail": "Details","recommendation": "Recommendation"})
        display_cols = ["Issue ID", "Issue Title", "Rationale", "Details", "Recommendation"]
        st.dataframe(display_issues_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No audit issues have been added to the plan yet.")
    if not st.session_state["audit_issues"].empty:
        df_download_link(st.session_state["audit_issues"], "audit_issues.csv", "⬇️ Download Audit Issues (CSV)")
    st.divider()
    plan_df = pd.DataFrame([plan])
    df_download_link(plan_df, "plan.csv", "⬇️ Download Plan (CSV)")
    st.success("Ready to connect to 🤖 PA Assistant for audit issue recommendations. ✨✨")

with tab_assist:
    st.subheader("💡 PA Assistant (AI/LLM)")
    st.write("🤖 Generate audit issue recommendations from AI")

    if st.button("🚀 Generate Recommendations from AI", type="primary", key="llm_assist_button"):
        if not st.session_state.api_key_global:
            st.error("Please set the API Key in the sidebar first.")
        else:
            with st.spinner("Generating recommendations..."):
                try:
                    issues_for_llm = st.session_state['audit_issues'][['title', 'rationale']]
                    plan_summary = f"""
Plan/Topic Name: {plan['plan_title']}
Project/Program Name: {plan['program_name']}
Objectives: {plan['objectives']}
Scope: {plan['scope']}
Assumptions/Limitations: {plan['assumptions']}
---
6W2H:
Who: {plan['who']}
To Whom: {plan['whom']}
What: {plan['what']}
Where: {plan['where']}
When: {plan['when']}
Why: {plan['why']}
How: {plan['how']}
How much: {plan['how_much']}
---
Logic Model:
{st.session_state['logic_items'].to_string()}
---
Issues from Past Reports:
{issues_for_llm.to_string()}
"""
                    user_prompt = f"""
Based on the audit plan information below, please provide 3 types of recommendations:
1.  Key audit issues to focus on, possibly referencing past issues, findings, or current events, with reasons why a performance auditor should prioritize them.
2.  Expected findings (with a high/medium/low probability of occurrence), with reasons that might reference past issues, news, or current situations.
3.  A draft audit report section, analyzing the potential impact and causes of the expected findings.
---
{plan_summary}
---
Please structure your response only in the following format:
<Key Audit Issues>
[Text for section 1]
</Key Audit Issues>

<Expected Findings>
[Text for section 2]
</Expected Findings>

<Draft Audit Report Section>
[Text for section 3]
</Draft Audit Report Section>
"""

                    client = OpenAI(
                        api_key=st.session_state.api_key_global,
                        base_url="https://api.opentyphoon.ai/v1"
                    )
                    
                    messages = [
                        {"role": "system", "content": "You are an expert in Performance Auditing."},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    response = client.chat.completions.create(
                        model="typhoon-v2.1-12b-instruct",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=2048,
                    )

                    full_response = response.choices[0].message.content

                    issue_start = full_response.find("<Key Audit Issues>") + len("<Key Audit Issues>")
                    issue_end = full_response.find("</Key Audit Issues>")
                    issues_text = full_response[issue_start:issue_end].strip()
                    
                    finding_start = full_response.find("<Expected Findings>") + len("<Expected Findings>")
                    finding_end = full_response.find("</Expected Findings>")
                    findings_text = full_response[finding_start:finding_end].strip()

                    report_start = full_response.find("<Draft Audit Report Section>") + len("<Draft Audit Report Section>")
                    report_end = full_response.find("</Draft Audit Report Section>")
                    report_text = full_response[report_start:report_end].strip()

                    st.session_state["gen_issues"] = issues_text
                    st.session_state["gen_findings"] = findings_text
                    st.session_state["gen_report"] = report_text

                    st.success("Recommendations generated from AI. ✅")

                except Exception as e:
                    st.error(f"Error calling AI: {e}")
                    st.session_state["gen_issues"] = ""
                    st.session_state["gen_findings"] = ""
                    st.session_state["gen_report"] = ""
    
    st.subheader("AI Results")

    with st.expander("1. Key Audit Issues to Focus On", expanded=True):
        st.write(st.session_state.get('gen_issues', "No data yet. Click 'Generate Recommendations from AI' to start."))

    with st.expander("2. Expected Findings (with probability)"):
        st.write(st.session_state.get('gen_findings', "No data yet."))

    with st.expander("3. Draft Audit Report (Preview)"):
        st.write(st.session_state.get('gen_report', "No data yet."))


with tab_chatbot:
    st.subheader("💬 PA Chat - Intelligent Assistant (Typhoon AI)")
    
    with st.expander("Upload Additional Documents (PDF, TXT, CSV)"):
        st.info("Data from your uploaded files will be combined with the system's document library to answer questions.")
        uploaded_files = st.file_uploader(
            "Select files...",
            type=['pdf', 'txt', 'csv'],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

    current_uploaded_file_names = {f.name for f in uploaded_files}
    if uploaded_files and st.session_state.get('last_uploaded_files') != current_uploaded_file_names:
        with st.spinner("Processing uploaded documents..."):
            st.session_state.doc_context_uploaded, _ = process_documents(uploaded_files, 'uploaded', MAX_CHARS_LIMIT, len(st.session_state.get('doc_context_local', '')))
            st.session_state.last_uploaded_files = current_uploaded_file_names
            st.session_state.chatbot_messages.append({"role": "assistant", "content": "New documents have been updated."})
            st.rerun()
    elif not uploaded_files and st.session_state.doc_context_uploaded:
        st.session_state.doc_context_uploaded = ""
        st.session_state.last_uploaded_files = set()
        st.session_state.chatbot_messages.append({"role": "assistant", "content": "Uploaded documents have been cleared."})
        st.rerun()

    local_len = len(st.session_state.get('doc_context_local', ''))
    uploaded_len = len(st.session_state.get('doc_context_uploaded', ''))
    
    with st.expander("View Context Details"):
        if local_len > 0:
            st.info(f"💾 Content from system library: {local_len:,} characters")
        if uploaded_len > 0:
            st.info(f"📤 Content from uploaded files: {uploaded_len:,} characters")
        st.success(f"✅ Total combined content: {(local_len + uploaded_len):,} characters (Max limit: {MAX_CHARS_LIMIT:,})")

    chat_container = st.container(height=500, border=True)
    with chat_container:
        for message in st.session_state.chatbot_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Type your question here...", key="chat_input_main"):
        st.session_state.chatbot_messages.append({"role": "user", "content": prompt})
        
        with chat_container:
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                api_key = st.session_state.api_key_global
                if not api_key:
                    error_message = "Error: API Key not found. Please contact the administrator to set it up."
                    message_placeholder.error(error_message)
                    st.session_state.chatbot_messages.append({"role": "assistant", "content": error_message})
                else:
                    try:
                        doc_context = st.session_state.get('doc_context_local', '') + st.session_state.get('doc_context_uploaded', '')
                        
                        system_prompt = f"""
You are an expert AI assistant. Your job is to answer the user's questions accurately and completely using two sources of information:
1.  **Internal Documents (Primary Source):** This is content extracted from files in the system. Always prioritize this information in your answers.
2.  **General Knowledge (Secondary Source):** If the answer is not in the internal documents, use your general training knowledge.
**Answering Rules:**
- When answering, always state the source (e.g., "According to the document [filename]...", "Based on the provided documents..."). If the filename is unknown, just say "Based on the provided documents".
- If internal document information conflicts with general knowledge, prioritize the document's information.
- If you cannot find the answer in either source, respond with "I'm sorry, I couldn't find relevant information in the documents or my knowledge base."
---
**Internal Document Context:**
{doc_context}
---
Based on the above information, answer the user's latest question.
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
                        error_message = f"An error occurred while processing: {e}"
                        message_placeholder.error(error_message)
                        st.session_state.chatbot_messages.append({"role": "assistant", "content": error_message})
