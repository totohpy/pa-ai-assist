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
from streamlit_agraph import agraph, Node, Edge, Config #! <<< เพิ่ม Library ใหม่

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
    ss.setdefault("plan", {"plan_id": "PLN-" + datetime.now().strftime("%y%m%d-%H%M%S"),"plan_title": ""})
    logic_cols = ["item_id","plan_id","type","description","metric","unit","target","source"]
    ss.setdefault("logic_items", pd.DataFrame(columns=logic_cols))
    # ... (ส่วนที่เหลือของ init_state เหมือนเดิม) ...

def next_id(prefix, df, col):
    if df.empty: return f"{prefix}-001"
    nums = [int(str(x).split("-")[-1]) for x in df[col] if str(x).split("-")[-1].isdigit()]
    n = max(nums) + 1 if nums else 1
    return f"{prefix}-{n:03d}"

#! <<< START: ฟังก์ชันใหม่สำหรับ Interactive Flowchart
def create_interactive_flowchart(df: pd.DataFrame):
    nodes = []
    edges = []
    
    styles = {
        "Objective": "#E6E6FA", "Input": "#a9def9", "Activity": "#e4c1f9",
        "Output": "#fcf6bd", "Outcome": "#d0f4de", "Impact": "#ff99c8",
        "E_Box": "#FFDAB9"
    }
    sequence = ["Objective", "Input", "Activity", "Output", "Outcome", "Impact"]
    
    nodes_exist = []
    # สร้าง Node หลัก
    for item_type in sequence:
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
            
            # ใช้ \n สำหรับขึ้นบรรทัดใหม่ใน agraph
            label = f"{header}\n\n" + "\n".join(description_lines)
            
            nodes.append(Node(id=item_type, label=label, color=styles.get(item_type), shape="box", font={'face': 'Kanit'}))
            nodes_exist.append(item_type)

    # เชื่อม Node หลัก
    if len(nodes_exist) > 1:
        for i in range(len(nodes_exist)-1):
            edges.append(Edge(source=nodes_exist[i], target=nodes_exist[i+1]))

    # สร้างและเชื่อมโยง 3E
    if len(nodes_exist) >= 3:
        # สร้าง Nodes
        nodes.append(Node(id='Economy', label='ประหยัด\n(Economy)', color=styles["E_Box"], shape='box', font={'face': 'Kanit'}))
        nodes.append(Node(id='Efficiency', label='ประสิทธิภาพ\n(Efficiency)', color=styles["E_Box"], shape='box', font={'face': 'Kanit'}))
        nodes.append(Node(id='Effectiveness', label='ประสิทธิผล\n(Effectiveness)', color=styles["E_Box"], shape='box', font={'face': 'Kanit'}))

        # สร้าง Edges
        if "Input" in nodes_exist: edges.append(Edge(source='Economy', target='Input', dashes=True))
        if "Input" in nodes_exist and "Output" in nodes_exist:
            edges.append(Edge(source='Input', target='Efficiency', dashes=True))
            edges.append(Edge(source='Efficiency', target='Output', dashes=True))
        if "Objective" in nodes_exist: edges.append(Edge(source='Objective', target='Effectiveness', dashes=True))
        if "Output" in nodes_exist: edges.append(Edge(source='Output', target='Effectiveness', dashes=True))
        if "Outcome" in nodes_exist: edges.append(Edge(source='Effectiveness', target='Outcome', dashes=True))
        if "Impact" in nodes_exist: edges.append(Edge(source='Effectiveness', target='Impact', dashes=True))

    # กำหนดค่า Config ของกราฟ
    config = Config(width='100%', 
                    height=600, # ตั้งค่าความสูงเริ่มต้น
                    directed=True, 
                    physics=False, # ปิด physics เพื่อให้จัดตาม layout ที่กำหนด
                    hierarchical={ # เปิดใช้งาน layout แบบลำดับชั้น
                        "enabled": True,
                        "direction": "LR", # LR = Left to Right
                        "sortMethod": "directed"
                    })

    return nodes, edges, config
#! <<< END: สิ้นสุดฟังก์ชัน Interactive

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
    # ... (ส่วนนี้เหมือนเดิม)
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

    st.subheader("📊 Flowchart Logic Model (ลากจัดวางได้)")
    with st.container(border=True):
        if not st.session_state.logic_items.empty:
            try:
                #! <<< แก้ไข: เปลี่ยนมาเรียกใช้ agraph
                nodes, edges, config = create_interactive_flowchart(st.session_state.logic_items)
                agraph(nodes=nodes, edges=edges, config=config)
            except Exception as e:
                st.error(f"ไม่สามารถสร้าง Flowchart ได้: {e}")
        else:
            st.info("กรุณาเพิ่มข้อมูลในฟอร์มด้านบนเพื่อสร้าง Flowchart")

# ... (โค้ดของ Tab ที่เหลือเหมือนเดิมทั้งหมด) ...
