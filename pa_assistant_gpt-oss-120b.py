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
from streamlit_mermaid import st_mermaid #! <<< Library สำหรับ Mermaid

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


# ----------------- ฟังก์ชันต่างๆ (ย่อส่วนที่ไม่เปลี่ยน) -----------------
MAX_CHARS_LIMIT = 200000

def init_state():
    ss = st.session_state
    ss.setdefault("plan", {"plan_id": "PLN-" + datetime.now().strftime("%y%m%d-%H%M%S"),"plan_title": "","program_name": "","who": "", "what": "", "where": "", "when": "", "why": "", "how": "", "how_much": "", "whom": "","objectives": "", "scope": "", "assumptions": "", "status": "Draft"})
    # แก้ไขคอลัมน์ให้ครบถ้วน
    logic_cols = ["item_id","plan_id","type","description","metric","unit","target","source"]
    ss.setdefault("logic_items", pd.DataFrame(columns=logic_cols))
    ss.setdefault("methods", pd.DataFrame(columns=["method_id","plan_id","type","tool_ref","sampling","questions","linked_issue","data_source","frequency"]))
    ss.setdefault("kpis", pd.DataFrame(columns=["kpi_id","plan_id","level","name","formula","numerator","denominator","unit","baseline","target","frequency","data_source","quality_requirements"]))
    ss.setdefault("risks", pd.DataFrame(columns=["risk_id","plan_id","description","category","likelihood","impact","mitigation","hypothesis"]))
    ss.setdefault("audit_issues", pd.DataFrame(columns=["issue_id","plan_id","title","rationale","linked_kpi","proposed_methods","source_finding_id","issue_detail", "recommendation"]))
    # (session state อื่นๆ เหมือนเดิม)

def next_id(prefix, df, col):
    if df.empty: return f"{prefix}-001"
    nums = [int(str(x).split("-")[-1]) for x in df[col] if str(x).split("-")[-1].isdigit()]
    n = max(nums) + 1 if nums else 1
    return f"{prefix}-{n:03d}"

#! <<< START: ฟังก์ชันสร้าง Flowchart ด้วย Mermaid (เวอร์ชันปรับปรุงล่าสุด)
def create_mermaid_flowchart(df: pd.DataFrame):
    """
    สร้างโค้ด Mermaid สำหรับวาด Flowchart แบบรวมกลุ่ม, มีสี, และสร้างกล่อง 3E อัตโนมัติ
    """
    # เพิ่ม "วัตถุประสงค์" เข้ามาในลำดับ
    sequence = ["วัตถุประสงค์", "Input", "Activity", "Output", "Outcome", "Impact"]
    
    mermaid_string = "graph LR\n"
    
    # กำหนดสีสำหรับแต่ละประเภท
    styles = {
        "วัตถุประสงค์": "fill:#E6E6FA,stroke:#333,stroke-width:2px",
        "Input": "fill:#a9def9,stroke:#333,stroke-width:2px",
        "Activity": "fill:#e4c1f9,stroke:#333,stroke-width:2px",
        "Output": "fill:#fcf6bd,stroke:#333,stroke-width:2px",
        "Outcome": "fill:#d0f4de,stroke:#333,stroke-width:2px",
        "Impact": "fill:#ff99c8,stroke:#333,stroke-width:2px",
        "E_Box": "fill:#FFDAB9,stroke:#e67300,stroke-width:2px,color:#e67300",
    }
    
    # เขียน classDef สำหรับ style ใน Mermaid
    for key, value in styles.items():
        mermaid_string += f"  classDef {key}Style {value}\n"

    nodes_exist = []
    for item_type in sequence:
        items_df = df[df['type'] == item_type]
        if not items_df.empty:
            header = f'<strong>{item_type}</strong>'
            description_lines = []
            for _, row in items_df.iterrows():
                desc = str(row.get('description', '') or '')
                metric = str(row.get('metric', '') or '')
                target = str(row.get('target', '') or '')
                unit = str(row.get('unit', '') or '')
                number = target if target else metric
                
                line_parts = [f"• {desc}"]
                if number: line_parts.append(number)
                if unit: line_parts.append(unit)
                description_lines.append(" ".join(part for part in line_parts if part))
            
            content = f"{header}<br/>" + "<br/>".join(description_lines)
            mermaid_string += f'  {item_type}["{content}"]\n'
            mermaid_string += f'  class {item_type} {item_type}Style\n' # กำหนด style ให้ node
            nodes_exist.append(item_type)
    
    # เชื่อมโยง Node หลัก
    if len(nodes_exist) > 1:
        main_flow = " --> ".join(nodes_exist)
        mermaid_string += f"  {main_flow}\n"

    # --- สร้างและเชื่อมโยง 3E อัตโนมัติ ---
    if len(nodes_exist) >= 3:
        mermaid_string += "\n  %% 3E Boxes Autogenerated\n"
        # 1. ประหยัด (Economy) -> Input
        if "Input" in nodes_exist:
            mermaid_string += '  Economy["ประหยัด (Economy)"]\n'
            mermaid_string += "  Economy --> Input\n"
            mermaid_string += "  class Economy E_BoxStyle\n"
        
        # 2. ประสิทธิภาพ (Efficiency) -> อยู่ระหว่าง Input กับ Output
        if "Input" in nodes_exist and "Output" in nodes_exist:
            mermaid_string += '  Efficiency["ประสิทธิภาพ (Efficiency)"]\n'
            mermaid_string += "  Input --> Efficiency --> Output\n"
            mermaid_string += "  class Efficiency E_BoxStyle\n"

        # 3. ประสิทธิผล (Effectiveness)
        effectiveness_connections = []
        if "วัตถุประสงค์" in nodes_exist: effectiveness_connections.append("วัตถุประสงค์")
        if "Output" in nodes_exist: effectiveness_connections.append("Output")

        if effectiveness_connections:
            mermaid_string += '  Effectiveness["ประสิทธิผล (Effectiveness)"]\n'
            for node in effectiveness_connections:
                mermaid_string += f"  {node} --> Effectiveness\n"
            
            if "Outcome" in nodes_exist: mermaid_string += "  Effectiveness --> Outcome\n"
            if "Impact" in nodes_exist: mermaid_string += "  Effectiveness --> Impact\n"
            mermaid_string += "  class Effectiveness E_BoxStyle\n"

    return mermaid_string
#! <<< END: สิ้นสุดฟังก์ชัน Mermaid

# (ฟังก์ชันอื่นๆ ที่ไม่เกี่ยวข้องกับการเปลี่ยนแปลงนี้จะถูกข้ามไปเพื่อความกระชับ)
# ... init_state(), next_id(), etc. ...

init_state()
plan = st.session_state["plan"]
logic_df = st.session_state["logic_items"]
# ... (ส่วนกำหนดตัวแปรอื่นๆ เหมือนเดิม) ...

st.title("🧭 Planning Studio – Performance Audit")

# ... (ส่วน expander คำแนะนำ และ markdown style เหมือนเดิม) ...

tab_plan, tab_logic, tab_method, tab_kpi, tab_risk, tab_issue, tab_preview, tab_assist, tab_chatbot = st.tabs(["1. ระบุ แผน & 6W2H", "2. ระบุ Logic Model", "3. ระบุ Methods", "4. ระบุ KPIs", "5. ระบุ Risks", "6. 🔍ค้นหาข้อตรวจพบที่ผ่านมา", "7. 📋สรุปข้อมูล (Preview)", "8. ✨PA Assistant แนะนำประเด็น", "9. 💬 PA Chat"]) 

with tab_plan:
    # ... (โค้ดใน tab_plan เหมือนเดิมทั้งหมด) ...
    pass

#! <<< START: แก้ไข Layout และ Logic ใน Tab นี้ทั้งหมด
with tab_logic:
    st.subheader("ระบุ Logic Model: Input → Activities → Output → Outcome → Impact")
    
    # --- 1. ส่วนเพิ่ม/แก้ไขข้อมูล ---
    with st.expander("➕ เพิ่ม/แก้ไขรายการใน Logic Model", expanded=True):
        
        # กำหนดค่าเริ่มต้นของ DataFrame ถ้ายังไม่มี
        if 'logic_items' not in st.session_state:
            st.session_state.logic_items = pd.DataFrame(columns=["item_id","plan_id","type","description","metric","unit","target","source"])

        edited_df = st.data_editor(
            st.session_state.logic_items,
            column_config={
                "type": st.column_config.SelectboxColumn(
                    "ประเภท",
                    options=["วัตถุประสงค์", "Input", "Activity", "Output", "Outcome", "Impact"],
                    required=True,
                ),
                "description": st.column_config.TextColumn("คำอธิบาย/รายละเอียด", required=True),
                "metric": st.column_config.TextColumn("ตัวชี้วัด/metric"),
                "target": st.column_config.TextColumn("เป้าหมาย"),
                "unit": st.column_config.TextColumn("หน่วย"),
                "source": st.column_config.TextColumn("แหล่งข้อมูล"),
                "item_id": st.column_config.TextColumn("ID", disabled=True),
                "plan_id": st.column_config.TextColumn("Plan ID", disabled=True),
            },
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic", # ให้ผู้ใช้เพิ่ม/ลบแถวได้เองจาก UI ของ data_editor
            key="logic_editor"
        )

        # --- ส่วนจัดการข้อมูลหลังการแก้ไข ---
        if edited_df is not None:
             # สร้าง ID สำหรับแถวใหม่ที่ยังไม่มี ID
            for i, row in edited_df.iterrows():
                if pd.isna(row.get('item_id')) or row.get('item_id') == '':
                    temp_df = edited_df.dropna(subset=['item_id'])
                    edited_df.loc[i, 'item_id'] = next_id("LG", temp_df, "item_id")
                if pd.isna(row.get('plan_id')) or row.get('plan_id') == '':
                     edited_df.loc[i, 'plan_id'] = plan["plan_id"]
            
            st.session_state.logic_items = edited_df

    # --- 2. ส่วนแสดง Flowchart ---
    st.subheader("📊 Flowchart Logic Model")
    with st.container(border=True):
        if not st.session_state.logic_items.empty:
            try:
                # เรียกใช้ Mermaid เสมอ (มุมมองรวมกลุ่มเป็นค่าหลัก)
                mermaid_chart = create_mermaid_flowchart(st.session_state.logic_items)
                st_mermaid(mermaid_chart, height="600px") # เพิ่มความสูงเผื่อ 3E boxes
            except Exception as e:
                st.error(f"ไม่สามารถสร้าง Flowchart ได้: {e}")
        else:
            st.info("กรุณาเพิ่มข้อมูลในตารางด้านบนเพื่อสร้าง Flowchart")

#! <<< END: สิ้นสุดการแก้ไข Tab Logic

with tab_method:
    # ... (โค้ดใน tab_method เหมือนเดิมทั้งหมด) ...
    pass
with tab_kpi:
    # ... (โค้ดใน tab_kpi เหมือนเดิมทั้งหมด) ...
    pass
with tab_risk:
    # ... (โค้ดใน tab_risk เหมือนเดิมทั้งหมด) ...
    pass
with tab_issue:
    # ... (โค้ดใน tab_issue เหมือนเดิมทั้งหมด) ...
    pass
with tab_preview:
    # ... (โค้ดใน tab_preview เหมือนเดิมทั้งหมด) ...
    pass
with tab_assist:
    # ... (โค้ดใน tab_assist เหมือนเดิมทั้งหมด) ...
    pass
with tab_chatbot:
    # ... (โค้dใน tab_chatbot เหมือนเดิมทั้งหมด) ...
    pass
