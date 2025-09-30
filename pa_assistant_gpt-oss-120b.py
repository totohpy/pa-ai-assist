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
import graphviz
from streamlit_mermaid import st_mermaid

# ตั้งค่าหน้าเพจ
st.set_page_config(page_title="Planning Studio (+ Findings Suggestions)", page_icon="🧭", layout="wide")

# ----------------- ⚙️ การตั้งค่ากลาง -----------------
with st.sidebar:
    st.title("⚙️ การตั้งค่ากลาง")
    st.info("API Key ถูกตั้งค่าโดยผู้ดูแลระบบผ่าน Streamlit Secrets")

    try:
        st.session_state.api_key_global = st.secrets["api_key"]
        st.success("API Key ถูกโหลดจากระบบ Secrets เรียบร้อยแล้ว")
    except KeyError:
        st.session_state.api_key_global = ""
        st.error("ไม่พบ API Key ใน Secrets, กรุณาตั้งค่าในหน้าตั้งค่าของแอป")
    except Exception as e:
        st.session_state.api_key_global = ""
        st.error(f"เกิดข้อผิดพลาดในการโหลด API Key: {e}")


    st.markdown("---")
    st.markdown("PA Planning Studio By PAO1 Audit Intelligence Nexus")


# ----------------- ฟังก์ชันต่างๆ -----------------
def init_state():
    ss = st.session_state
    ss.setdefault("plan", {"plan_id": "PLN-" + datetime.now().strftime("%y%m%d-%H%M%S"),"plan_title": "","program_name": "","who": "", "what": "", "where": "", "when": "", "why": "", "how": "", "how_much": "", "whom": "","objectives": "", "scope": "", "assumptions": "", "status": "Draft"})
    logic_cols = ["item_id","plan_id","type","description","metric","unit","target","source"]
    ss.setdefault("logic_items", pd.DataFrame(columns=logic_cols))
    ss.setdefault("methods", pd.DataFrame(columns=["method_id","plan_id","type","tool_ref","sampling","questions","linked_issue","data_source","frequency"]))
    ss.setdefault("kpis", pd.DataFrame(columns=["kpi_id","plan_id","level","name","formula","numerator","denominator","unit","baseline","target","frequency","data_source","quality_requirements"]))
    ss.setdefault("risks", pd.DataFrame(columns=["risk_id","plan_id","description","category","likelihood","impact","mitigation","hypothesis"]))
    ss.setdefault("audit_issues", pd.DataFrame(columns=["issue_id","plan_id","title","rationale","linked_kpi","proposed_methods","source_finding_id","issue_detail", "recommendation"]))
    ss.setdefault("gen_issues", ""); ss.setdefault("gen_findings", ""); ss.setdefault("gen_report", "")
    ss.setdefault("issue_results", pd.DataFrame()); ss.setdefault("ref_seed", ""); ss.setdefault("issue_query_text", "")
    ss.setdefault('api_key_global', ''); ss.setdefault("6w2h_output", "")
    ss.setdefault('chatbot_messages', [{"role": "assistant", "content": "สวัสดีครับ ผมคือ PA Chat ผู้ช่วยอัจฉริยะ"}])
    ss.setdefault('doc_context_uploaded', ""); ss.setdefault('last_uploaded_files', set())
    if 'doc_context_local' not in ss:
        pass


def next_id(prefix, df, col):
    if df.empty: return f"{prefix}-001"
    nums = [int(str(x).split("-")[-1]) for x in df[col] if str(x).split("-")[-1].isdigit()]
    n = max(nums) + 1 if nums else 1
    return f"{prefix}-{n:03d}"

def create_mermaid_flowchart(df: pd.DataFrame):
    type_to_id = {
        "Objective": "Objective", "Input": "Input", "Activity": "Activity",
        "Output": "Output", "Outcome": "Outcome", "Impact": "Impact"
    }
    sequence = ["Objective", "Input", "Activity", "Output", "Outcome", "Impact"]
    
    mermaid_string = "graph TD\n"
    
    styles = {
        "Objective": "fill:#E6E6FA,stroke:#333,stroke-width:2px",
        "Input": "fill:#a9def9,stroke:#333,stroke-width:2px",
        "Activity": "fill:#e4c1f9,stroke:#333,stroke-width:2px",
        "Output": "fill:#fcf6bd,stroke:#333,stroke-width:2px",
        "Outcome": "fill:#d0f4de,stroke:#333,stroke-width:2px",
        "Impact": "fill:#ff99c8,stroke:#333,stroke-width:2px",
    }
    for key, value in styles.items():
        mermaid_string += f"  classDef {key}Style {value}\n"
        
    nodes_exist = []
    for item_type in sequence:
        items_df = df[df['type'] == item_type]
        if not items_df.empty:
            node_id = type_to_id[item_type]
            header = f'<strong>{item_type}</strong>'
            description_lines = []
            for _, row in items_df.iterrows():
                desc = str(row.get('description', '') or ''); metric = str(row.get('metric', '') or '')
                target = str(row.get('target', '') or ''); unit = str(row.get('unit', '') or '')
                number = target if target else metric
                line_parts = [f"• {desc}"]
                if number: line_parts.append(number)
                if unit: line_parts.append(unit)
                description_lines.append(" ".join(part for part in line_parts if part))
            
            content_body = "<br/>".join(description_lines)
            content = f"<div style='text-align: left;'>{header}<br/>{content_body}</div>"
            mermaid_string += f'  {node_id}["{content}"]\n'
            mermaid_string += f'  class {node_id} {node_id}Style\n'
            nodes_exist.append(node_id)
    
    # --- ส่วนการเชื่อมเส้น ---
    if len(nodes_exist) > 1:
        mermaid_string += "  " + " --> ".join(nodes_exist) + "\n"

    #! <<< ลบส่วนการสร้างกล่อง 3E อัตโนมัติออกทั้งหมดแล้ว
            
    return mermaid_string

# --- Main App ---
init_state()
plan = st.session_state.get("plan", {})
logic_df = st.session_state.get("logic_items", pd.DataFrame())

st.title("🧭 Planning Studio – Performance Audit")

tab_plan, tab_logic, tab_method, tab_kpi, tab_risk, tab_issue, tab_preview, tab_assist, tab_chatbot = st.tabs([
    "1. ระบุ แผน & 6W2H", "2. ระบุ Logic Model", "3. ระบุ Methods", "4. ระบุ KPIs", 
    "5. ระบุ Risks", "6. 🔍ค้นหาข้อตรวจพบที่ผ่านมา", "7. 📋สรุปข้อมูล (Preview)", 
    "8. ✨PA Assistant แนะนำประเด็น", "9. 💬 PA Chat"
])

with tab_plan:
    # This section's code is lengthy and unchanged, so it's omitted for brevity.
    # You can copy it from any of the previous full code versions.
    st.subheader("ข้อมูลแผน - กรุณาระบุข้อมูล")
    pass

with tab_logic:
    st.subheader("ระบุ Logic Model")

    with st.expander("➕ เพิ่มรายการใหม่ใน Logic Model", expanded=True):
        with st.container(border=True):
            colA, colB, colC = st.columns(3)
            with colA:
                typ = st.selectbox("ประเภท", ["Objective", "Input", "Activity", "Output", "Outcome", "Impact"], key="logic_type")
                desc = st.text_input("คำอธิบาย/รายละเอียด", key="logic_desc")
                metric = st.text_input("ตัวชี้วัด (จำนวน)", key="logic_metric")
            with colB:
                unit = st.text_input("หน่วย", value="", key="logic_unit")
                target = st.text_input("เป้าหมาย", value="", key="logic_target")
            with colC:
                source = st.text_input("แหล่งข้อมูล", value="", key="logic_source")
            
            if st.button("เพิ่ม Logic Item", type="primary", key="add_logic_item_btn"):
                if desc:
                    new_row = pd.DataFrame([{
                        "item_id": next_id("LG", st.session_state.logic_items, "item_id"), 
                        "plan_id": plan.get("plan_id", ""), 
                        "type": typ, "description": desc, "metric": metric, 
                        "unit": unit, "target": target, "source": source
                    }])
                    st.session_state.logic_items = pd.concat([st.session_state.logic_items, new_row], ignore_index=True)
                    st.success("เพิ่มข้อมูลเรียบร้อยแล้ว"); st.rerun()
                else:
                    st.warning("กรุณากรอก 'คำอธิบาย/รายละเอียด' ก่อนทำการเพิ่ม")
    
    st.markdown("---")
    
    st.markdown("##### 📝 ตาราง Logic Model (สามารถแก้ไขหรือลบแถวได้โดยตรง)")
    edited_df = st.data_editor(
        st.session_state.logic_items,
        column_config={
            "type": st.column_config.SelectboxColumn("ประเภท", options=["Objective", "Input", "Activity", "Output", "Outcome", "Impact"], required=True),
            "description": st.column_config.TextColumn("คำอธิบาย/รายละเอียด", required=True),
            "metric": st.column_config.TextColumn("ตัวชี้วัด (จำนวน)"),
            "target": st.column_config.TextColumn("เป้าหมาย"),
            "unit": st.column_config.TextColumn("หน่วย"),
            "source": st.column_config.TextColumn("แหล่งข้อมูล"),
            "item_id": st.column_config.TextColumn("ID", disabled=True),
            "plan_id": st.column_config.TextColumn("Plan ID", disabled=True),
        },
        use_container_width=True,
        hide_index=True,
        key="logic_editor_main"
    )

    st.session_state.logic_items = edited_df
    
    cols = st.columns([0.85, 0.15])
    with cols[1]:
        if st.button("🧹 ล้างทั้งหมด (Reset)", use_container_width=True):
            empty_df = pd.DataFrame(columns=st.session_state.logic_items.columns)
            st.session_state.logic_items = empty_df
            st.rerun()

    st.markdown("---")

    st.subheader("📊 Flowchart Logic Model")
    with st.container(border=True):
        if not st.session_state.logic_items.empty:
            try:
                base_height = 300
                num_rows = len(st.session_state.logic_items)
                num_types = st.session_state.logic_items['type'].nunique()
                dynamic_height = base_height + (num_rows * 20) + (num_types * 50)
                chart_height = min(max(dynamic_height, 400), 1200)
                
                mermaid_chart = create_mermaid_flowchart(st.session_state.logic_items)
                st_mermaid(mermaid_chart, height=f"{chart_height}px")
            except Exception as e:
                st.error(f"ไม่สามารถสร้าง Flowchart ได้: {e}")
        else:
            st.info("กรุณาเพิ่มข้อมูลในฟอร์มด้านบนเพื่อสร้าง Flowchart")

with tab_method:
    # This section's code is lengthy and unchanged.
    st.subheader("ระบุวิธีการเก็บข้อมูล")
    pass

with tab_kpi:
    # This section's code is lengthy and unchanged.
    st.subheader("ระบุตัวชี้วัด (KPIs)")
    pass

with tab_risk:
    # This section's code is lengthy and unchanged.
    st.subheader("ระบุความเสี่ยง (Risks)")
    pass

with tab_issue:
    # This section's code is lengthy and unchanged.
    st.subheader("🔎 แนะนำประเด็นตรวจสอบจากรายงานเก่า")
    pass

with tab_preview:
    # This section's code is lengthy and unchanged.
    st.subheader("สรุปแผน (Preview)")
    pass

with tab_assist:
    # This section's code is lengthy and unchanged.
    st.subheader("💡 PA Assistant (AI/LLM)")
    pass

with tab_chatbot:
    # This section's code is lengthy and unchanged.
    st.subheader("💬 PA Chat - ผู้ช่วยอัจฉริยะ")
    pass
