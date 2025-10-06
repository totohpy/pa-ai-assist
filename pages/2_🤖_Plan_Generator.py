import streamlit as st
from fpdf import FPDF
from datetime import datetime
import json
import re
from openai import OpenAI
import os

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Plan Generator")
st.title("🤖 AI Plan Generator")
st.markdown("เครื่องมือช่วยสร้างแผนและแนวการตรวจสอบ พร้อมระบบ AI ช่วยร่างเนื้อหา")

# --- Initialize Session State ---
def init_plan_state():
    ss = st.session_state
    if "plan_gen_data" not in ss:
        ss.plan_gen_data = {
            "general_info": {"office": "", "topic": "", "agency": "", "ministry": ""},
            "objectives": [],
            "estimates": {"cost": "", "effort": ""},
            "signatures": {
                "maker": {"name": "", "position": "", "date": None, "comment": ""},
                "reviewer": {"name": "", "position": "", "date": None, "comment": ""},
                "approver": {"name": "", "position": "", "date": None, "comment": ""},
            },
            "ui_feedback_message": None # For showing success/error messages reliably
        }
init_plan_state()

# --- Helper Functions ---
def add_objective():
    new_obj = {"id": f"obj_{len(st.session_state.plan_gen_data['objectives']) + 1}", "text": "", "issues": []}
    st.session_state.plan_gen_data["objectives"].append(new_obj)
    st.session_state.ui_feedback_message = None # Clear message on action

def remove_objective(obj_index):
    st.session_state.plan_gen_data["objectives"].pop(obj_index)
    st.session_state.ui_feedback_message = None

def add_issue(obj_index, parent_issue_path=None):
    obj = st.session_state.plan_gen_data["objectives"][obj_index]
    target_list = obj["issues"]
    
    if parent_issue_path:
        current_level = obj
        for issue_index in parent_issue_path:
            current_level = current_level["issues"][issue_index]
        target_list = current_level["issues"]

    new_issue = {
        "id": f"issue_{obj_index}_{len(target_list) + 1}",
        "text": "",
        "details": {"criteria": "", "info_needed": "", "source": "", "collection_method": "", "analysis_method": ""},
        "issues": []
    }
    target_list.append(new_issue)
    st.session_state.ui_feedback_message = None

# --- AI Function ---
def call_typhoon_api(context_text):
    try:
        api_key = st.secrets["api_key"]
        client = OpenAI(api_key=api_key, base_url="https://api.opentyphoon.ai/v1")
        full_prompt = f"""คุณคือผู้เชี่ยวชาญด้านการตรวจสอบภาครัฐ (State Auditor) 
หน้าที่ของคุณคือช่วยร่างแผนและแนวการตรวจสอบตามข้อมูลที่ได้รับ 
จงสร้างเนื้อหาสำหรับแนวการตรวจสอบโดยละเอียดสำหรับประเด็นสุดท้ายที่อยู่ใน context ต่อไปนี้:
--- CONTEXT START ---
{context_text}
--- CONTEXT END ---
**คำสั่ง:**
ตอบกลับเป็น JSON object **เท่านั้น** ห้ามมีข้อความอื่นนำหน้าหรือตามหลัง JSON โดยเด็ดขาด 
ใช้ key ต่อไปนี้: "criteria", "info_needed", "source", "collection_method", "analysis_method" 
เนื้อหาต้องเป็นภาษาไทยที่กระชับและชัดเจน
"""
        messages = [{"role": "user", "content": full_prompt}]
        response = client.chat.completions.create(
            model="typhoon-v2.1-12b-instruct", messages=messages, temperature=0.5
        )
        generated_text = response.choices[0].message.content
        match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        st.session_state.ui_feedback_message = ("error", f"AI ไม่ได้ส่งข้อมูลกลับมาในรูปแบบ JSON ที่คาดหวัง:\n{generated_text}")
        return None
    except Exception as e:
        st.session_state.ui_feedback_message = ("error", f"เกิดข้อผิดพลาดในการเรียก Typhoon AI: {e}")
        return None

# --- PDF Generation Function ---
class PDF(FPDF):
    def header(self):
        self.set_font('Sarabun', 'B', 14)
        self.cell(0, 10, 'แผนและแนวการตรวจสอบ', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Sarabun', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def write_thai(self, text, style=''):
        self.set_font('Sarabun', style, self.font_size)
        self.multi_cell(0, 7, str(text))

def generate_pdf():
    # --- Font File Paths ---
    FONT_REGULAR = 'Sarabun-Regular.ttf'
    FONT_BOLD = 'Sarabun-Bold.ttf'
    FONT_ITALIC = 'Sarabun-Italic.ttf'
    
    # --- Check if all font files exist ---
    for font_file in [FONT_REGULAR, FONT_BOLD, FONT_ITALIC]:
        if not os.path.exists(font_file):
            st.error(f"ไม่พบไฟล์ฟอนต์ที่จำเป็น: '{font_file}'! กรุณาตรวจสอบให้แน่ใจว่าไฟล์อยู่ในตำแหน่งบนสุดของโปรเจกต์")
            return None
            
    pdf = PDF(orientation='L', unit='mm', format='A4')
    
    # --- Add all font styles to FPDF ---
    try:
        pdf.add_font('Sarabun', '', FONT_REGULAR, uni=True)
        pdf.add_font('Sarabun', 'B', FONT_BOLD, uni=True)
        pdf.add_font('Sarabun', 'I', FONT_ITALIC, uni=True)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดร้ายแรงในการโหลดฟอนต์: {e}")
        return None

    pdf.add_page()
    plan_data = st.session_state.plan_gen_data
    
    pdf.set_font('Sarabun', 'B', 12)
    pdf.write_thai(f"เรื่องที่ตรวจสอบ: {plan_data['general_info']['topic']}", style='')
    pdf.write_thai(f"หน่วยงาน: {plan_data['general_info']['agency']} กระทรวง: {plan_data['general_info']['ministry']}", style='')
    pdf.write_thai(f"สำนักงาน: {plan_data['general_info']['office']}", style='')
    pdf.ln(5)

    pdf.write_thai('วัตถุประสงค์และประเด็นการตรวจสอบ', style='B')
    
    def write_issues_to_pdf(issues_list, prefix_num, indent_level=1):
        for i, issue in enumerate(issues_list):
            current_prefix = f"{prefix_num}.{i+1}"
            pdf.set_font_size(11)
            pdf.write_thai(f"{' ' * (indent_level*4)}ประเด็น {current_prefix}: {issue['text']}", style='B')
            
            if not issue['issues']:
                pdf.set_font_size(10)
                details = issue['details']
                indent_str = ' ' * ((indent_level*4)+4)
                pdf.write_thai(f"{indent_str}เกณฑ์: {details.get('criteria', '')}")
                pdf.write_thai(f"{indent_str}ข้อมูลที่ต้องการ: {details.get('info_needed', '')}")
                pdf.write_thai(f"{indent_str}แหล่งข้อมูล: {details.get('source', '')}")
                pdf.write_thai(f"{indent_str}วิธีรวบรวม: {details.get('collection_method', '')}")
                pdf.write_thai(f"{indent_str}วิธีวิเคราะห์: {details.get('analysis_method', '')}")
            
            if issue['issues']:
                write_issues_to_pdf(issue['issues'], current_prefix, indent_level + 1)

    for i, obj in enumerate(plan_data['objectives']):
        pdf.set_font_size(11)
        pdf.write_thai(f"\nวัตถุประสงค์ที่ {i+1}: {obj['text']}", style='B')
        write_issues_to_pdf(obj['issues'], str(i+1))
    
    pdf.ln(10)
    pdf.write_thai('ประมาณการและผู้จัดทำ', style='B')
    pdf.write_thai(f"ประมาณการค่าใช้จ่าย: {plan_data['estimates']['cost']}")
    pdf.write_thai(f"ประมาณการคน/วัน: {plan_data['estimates']['effort']}")
    
    pdf.ln(10)
    sig_data = plan_data['signatures']
    col_width = pdf.w / 3.2 
    line_height = 7
    
    pdf.set_font('Sarabun', 'B', 11)
    pdf.cell(col_width, line_height, 'ผู้จัดทำ', border=1, align='C')
    pdf.cell(col_width, line_height, 'ผู้สอบทาน', border=1, align='C')
    pdf.cell(col_width, line_height, 'ผู้อนุมัติ (รผต. / ผอ. สำนัก)', border=1, align='C')
    pdf.ln(line_height)
    
    pdf.set_font('Sarabun', '', 10)
    y_before = pdf.get_y()
    date_format = lambda d: d.strftime('%d/%m/%Y') if d else ''
    
    content1 = f"ลงชื่อ: {sig_data['maker']['name']}\nตำแหน่ง: {sig_data['maker']['position']}\nวันที่: {date_format(sig_data['maker']['date'])}\nความเห็น: {sig_data['maker']['comment']}"
    content2 = f"ลงชื่อ: {sig_data['reviewer']['name']}\nตำแหน่ง: {sig_data['reviewer']['position']}\nวันที่: {date_format(sig_data['reviewer']['date'])}\nความเห็น: {sig_data['reviewer']['comment']}"
    content3 = f"ลงชื่อ: {sig_data['approver']['name']}\nตำแหน่ง: {sig_data['approver']['position']}\nวันที่: {date_format(sig_data['approver']['date'])}\nความเห็น: {sig_data['approver']['comment']}"

    pdf.multi_cell(col_width, line_height, content1, border=1)
    y1 = pdf.get_y()
    pdf.set_xy(pdf.get_x() + col_width, y_before)
    
    pdf.multi_cell(col_width, line_height, content2, border=1)
    y2 = pdf.get_y()
    pdf.set_xy(pdf.get_x() + col_width * 2, y_before)

    pdf.multi_cell(col_width, line_height, content3, border=1)
    pdf.set_y(max(y1, y2, pdf.get_y()))

    return pdf.output(dest='S').encode('latin-1')

# --- UI Rendering ---
# Display feedback message if it exists
if st.session_state.ui_feedback_message:
    msg_type, msg_content = st.session_state.ui_feedback_message
    if msg_type == "success":
        st.success(msg_content)
    else:
        st.error(msg_content)
    st.session_state.ui_feedback_message = None # Clear after displaying

with st.form("general_info_form"):
    st.subheader("1. ข้อมูลทั่วไป")
    c1, c2 = st.columns(2)
    st.session_state.plan_gen_data["general_info"]["office"] = c1.text_input("สำนักงาน/จังหวัด/กลุ่ม", st.session_state.plan_gen_data["general_info"]["office"])
    st.session_state.plan_gen_data["general_info"]["topic"] = c1.text_input("เรื่องที่ตรวจสอบ", st.session_state.plan_gen_data["general_info"]["topic"])
    st.session_state.plan_gen_data["general_info"]["agency"] = c2.text_input("หน่วยงาน", st.session_state.plan_gen_data["general_info"]["agency"])
    st.session_state.plan_gen_data["general_info"]["ministry"] = c2.text_input("กระทรวง", st.session_state.plan_gen_data["general_info"]["ministry"])
    st.form_submit_button("บันทึกข้อมูลทั่วไป", use_container_width=True)

st.subheader("2. วัตถุประสงค์และประเด็นการตรวจสอบ")
for i, obj in enumerate(st.session_state.plan_gen_data["objectives"]):
    with st.container(border=True):
        c1, c2 = st.columns([5, 1])
        obj['text'] = c1.text_area(f"วัตถุประสงค์ที่ {i+1}", obj['text'], key=f"obj_text_{i}")
        c2.button("🗑️ ลบ", key=f"del_obj_{i}", on_click=remove_objective, args=(i,), use_container_width=True)

        def display_issues(issues_list, obj_index, path):
            for j, issue in enumerate(issues_list):
                current_path = path + [j]
                level = len(current_path)
                prefix_parts = [str(obj_index + 1)] + [str(p + 1) for p in current_path]
                prefix = ".".join(prefix_parts)
                unique_key_suffix = f"{obj_index}_{'_'.join(map(str, current_path))}"

                with st.container():
                    st.markdown(f"<div style='margin-left: {level * 20}px;'>", unsafe_allow_html=True)
                    issue['text'] = st.text_area(f"ประเด็นการตรวจสอบที่ {prefix}", issue['text'], key=f"issue_text_{unique_key_suffix}")

                    if not issue['issues']:
                        with st.expander("รายละเอียดแนวการตรวจสอบ (AI)"):
                            if st.button(f"🤖 ให้ AI ช่วยร่าง (ประเด็น {prefix})", key=f"ai_btn_{unique_key_suffix}"):
                                with st.spinner("AI กำลังประมวลผล..."):
                                    context = f"เรื่องที่ตรวจสอบ: {st.session_state.plan_gen_data['general_info']['topic']}\n"
                                    context += f"วัตถุประสงค์: {obj['text']}\n"
                                    context += f"ประเด็นการตรวจสอบ: {issue['text']}"
                                    ai_result = call_typhoon_api(context)
                                    if ai_result and all(k in ai_result for k in issue['details'].keys()):
                                        issue['details'] = ai_result
                                        st.session_state.ui_feedback_message = ("success", f"AI สร้างเนื้อหาสำหรับประเด็น {prefix} เรียบร้อยแล้ว")
                                        st.rerun() # Rerun to show new data and success message
                                    
                            issue['details']['criteria'] = st.text_area("เกณฑ์การตรวจสอบ", issue['details'].get('criteria', ''), key=f"crit_{unique_key_suffix}")
                            issue['details']['info_needed'] = st.text_area("ข้อมูลที่ต้องการ", issue['details'].get('info_needed', ''), key=f"info_{unique_key_suffix}")
                            issue['details']['source'] = st.text_area("แหล่งข้อมูล", issue['details'].get('source', ''), key=f"src_{unique_key_suffix}")
                            issue['details']['collection_method'] = st.text_area("วิธีการรวบรวมหลักฐาน", issue['details'].get('collection_method', ''), key=f"coll_{unique_key_suffix}")
                            issue['details']['analysis_method'] = st.text_area("วิธีการวิเคราะห์หลักฐาน", issue['details'].get('analysis_method', ''), key=f"anal_{unique_key_suffix}")
                    
                    st.button(f"➕ เพิ่มประเด็นย่อย (สำหรับ {prefix})", key=f"add_sub_issue_{unique_key_suffix}", on_click=add_issue, args=(obj_index, current_path))
                    display_issues(issue['issues'], obj_index, current_path)
                    st.markdown("</div>", unsafe_allow_html=True)

        display_issues(obj['issues'], i, [])
        st.button("➕ เพิ่มประเด็นการตรวจสอบหลัก", key=f"add_issue_{i}", on_click=add_issue, args=(i, None))

st.button("➕ เพิ่มวัตถุประสงค์", on_click=add_objective, type="primary")

with st.form("estimates_signatures_form"):
    st.subheader("3. ประมาณการและผู้จัดทำ")
    sig_data = st.session_state.plan_gen_data["signatures"]
    st.session_state.plan_gen_data["estimates"]["cost"] = st.text_area("ประมาณการค่าใช้จ่ายในการตรวจสอบ", st.session_state.plan_gen_data["estimates"]["cost"], key="cost_estimate")
    st.session_state.plan_gen_data["estimates"]["effort"] = st.text_area("ประมาณการคน/วันที่ใช้ในการตรวจสอบ", st.session_state.plan_gen_data["estimates"]["effort"], key="effort_estimate")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**ผู้จัดทำ**")
        sig_data["maker"]["name"] = st.text_input("ลงชื่อ", key="maker_name")
        sig_data["maker"]["position"] = st.text_input("ตำแหน่ง", key="maker_pos")
        sig_data["maker"]["date"] = st.date_input("วันที่", value=None, key="maker_date")
        sig_data["maker"]["comment"] = st.text_area("ความเห็นเพิ่มเติม", key="maker_comment")
    with c2:
        st.markdown("**ผู้สอบทาน**")
        sig_data["reviewer"]["name"] = st.text_input("ลงชื่อ", key="reviewer_name")
        sig_data["reviewer"]["position"] = st.text_input("ตำแหน่ง", key="reviewer_pos")
        sig_data["reviewer"]["date"] = st.date_input("วันที่", value=None, key="reviewer_date")
        sig_data["reviewer"]["comment"] = st.text_area("ความเห็นเพิ่มเติม", key="reviewer_comment")
    with c3:
        st.markdown("**ผู้อนุมัติ (รผต. / ผอ. สำนัก)**")
        sig_data["approver"]["name"] = st.text_input("ลงชื่อ", key="approver_name")
        sig_data["approver"]["position"] = st.text_input("ตำแหน่ง", key="approver_pos")
        sig_data["approver"]["date"] = st.date_input("วันที่", value=None, key="approver_date")
        sig_data["approver"]["comment"] = st.text_area("ความเห็นเพิ่มเติม", key="approver_comment")
    st.form_submit_button("บันทึกข้อมูลผู้จัดทำ", use_container_width=True)

st.divider()
st.subheader("สร้างเอกสาร")
if st.button("📄 สร้างเอกสาร PDF (แนวนอน)", type="primary", use_container_width=True):
    with st.spinner("กำลังสร้างไฟล์ PDF..."):
        pdf_bytes = generate_pdf()
        if pdf_bytes:
            st.download_button(
                label="✅ ดาวน์โหลด PDF สำเร็จ!",
                data=pdf_bytes,
                file_name=f"แผนการตรวจสอบ_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

