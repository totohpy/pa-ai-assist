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
    # ... (ส่วนที่เหลือของ init_state เหมือนเดิม) ...

def next_id(prefix, df, col):
    if df.empty: return f"{prefix}-001"
    nums = [int(str(x).split("-")[-1]) for x in df[col] if str(x).split("-")[-1].isdigit()]
    n = max(nums) + 1 if nums else 1
    return f"{prefix}-{n:03d}"

def create_graphviz_flowchart_with_3e(df: pd.DataFrame):
    """
    สร้าง Flowchart ด้วย Graphviz + HTML-like Label และเพิ่มกล่อง 3E
    """
    dot = graphviz.Digraph(comment='Logic Model with 3Es')
    dot.attr('graph', rankdir='LR', splines='ortho', bgcolor='transparent', compound='true', fontname='Kanit')
    dot.attr('node', shape='plain', fontname='Kanit')
    dot.attr('edge', fontname='Kanit')

    #! <<< START: แก้ไข - เพิ่ม Dummy Node ที่มองไม่เห็น
    # โหนดนี้จะช่วยแก้ปัญหาที่โหนดแรกสุดมีขนาดใหญ่ผิดปกติ
    dot.node('start_node', label='', style='invis', width='0', height='0')
    #! <<< END: สิ้นสุดการแก้ไข

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
            header_html = f'<TR><TD BORDER="0" ALIGN="CENTER" BGCOLOR="{styles.get(item_type)}"><B>{item_type}</B></TD></TR>'
            rows_html = []
            for _, row in items_df.iterrows():
                desc = str(row.get('description', '') or ''); metric = str(row.get('metric', '') or '')
                target = str(row.get('target', '') or ''); unit = str(row.get('unit', '') or '')
                number = target if target else metric
                line_parts = [f"• {desc}"]
                if number: line_parts.append(number)
                if unit: line_parts.append(unit)
                line = " ".join(part for part in line_parts if part)
                rows_html.append(f'<TR><TD BORDER="0" ALIGN="LEFT">{line}</TD></TR>')
            
            full_html = f'''<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="5" STYLE="ROUNDED" BGCOLOR="{styles.get(item_type)}">{header_html}{''.join(rows_html)}</TABLE>>'''
            dot.node(name=item_type, label=full_html)
            nodes_exist.append(item_type)

    # เชื่อม Node หลัก
    if len(nodes_exist) > 1:
        dot.edges([(nodes_exist[i], nodes_exist[i+1]) for i in range(len(nodes_exist)-1)])

    # สร้างและเชื่อมโยง 3E (เมื่อมี Node หลัก 3 โหนดขึ้นไป)
    if len(nodes_exist) >= 3:
        dot.node('Economy', label='ประหยัด (Economy)', shape='box', style='rounded,filled', fillcolor=styles["E_Box"])
        dot.node('Efficiency', label='ประสิทธิภาพ (Efficiency)', shape='box', style='rounded,filled', fillcolor=styles["E_Box"])
        dot.node('Effectiveness', label='ประสิทธิผล (Effectiveness)', shape='box', style='rounded,filled', fillcolor=styles["E_Box"])

        # เชื่อมโยงเส้น
        if "Input" in nodes_exist:
            dot.edge('Economy', 'Input', style='dashed')
        if "Input" in nodes_exist and "Output" in nodes_exist:
            dot.edge('Input', 'Efficiency', style='dashed')
            dot.edge('Efficiency', 'Output', style='dashed')
        if "Objective" in nodes_exist:
            dot.edge('Objective', 'Effectiveness', style='dashed')
        if "Output" in nodes_exist:
            dot.edge('Output', 'Effectiveness', style='dashed')
        if "Outcome" in nodes_exist:
            dot.edge('Effectiveness', 'Outcome', style='dashed')
        if "Impact" in nodes_exist:
            dot.edge('Effectiveness', 'Impact', style='dashed')
            
    return dot

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
    # ... (โค้ดส่วนนี้เหมือนเดิม)
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
                graphviz_chart = create_graphviz_flowchart_with_3e(st.session_state.logic_items)
                st.graphviz_chart(graphviz_chart, use_container_width=True)
            except Exception as e:
                st.error(f"ไม่สามารถสร้าง Flowchart ได้: {e}")
        else:
            st.info("กรุณาเพิ่มข้อมูลในฟอร์มด้านบนเพื่อสร้าง Flowchart")

# ... (โค้ดของ Tab ที่เหลือเหมือนเดิมทั้งหมด) ...
