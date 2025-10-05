import streamlit as st
import google.generativeai as genai
from fpdf import FPDF
from datetime import datetime
import json

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
            }
        }
init_plan_state()

# --- Helper Functions ---
def add_objective():
    new_obj = {"id": f"obj_{len(st.session_state.plan_gen_data['objectives']) + 1}", "text": "", "issues": []}
    st.session_state.plan_gen_data["objectives"].append(new_obj)

def remove_objective(obj_index):
    st.session_state.plan_gen_data["objectives"].pop(obj_index)

def add_issue(obj_index, parent_issue_path=None):
    obj = st.session_state.plan_gen_data["objectives"][obj_index]
    target_list = obj["issues"]
    
    # Navigate to the correct sub-issue list if a path is provided
    if parent_issue_path:
        current_level = obj
        for issue_index in parent_issue_path:
            current_level = current_level["issues"][issue_index]
        target_list = current_level["issues"]

    new_issue = {
        "id": f"issue_{obj_index}_{len(target_list) + 1}",
        "text": "",
        "details": {"criteria": "", "info_needed": "", "source": "", "collection_method": "", "analysis_method": ""},
        "issues": [] # For sub-issues
    }
    target_list.append(new_issue)

# --- AI Function ---
def call_gemini_api(context_text):
    try:
        # Load API Key from Streamlit Secrets
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel('gemini-1.5-flash')
        
        system_prompt = """คุณคือผู้เชี่ยวชาญด้านการตรวจสอบภาครัฐ (State Auditor) มีหน้าที่ในการช่วยร่างแผนและแนวการตรวจสอบตามข้อมูลที่ได้รับ จงสร้างเนื้อหาสำหรับแนวการตรวจสอบโดยละเอียดสำหรับประเด็นสุดท้ายเท่านั้น ตอบกลับเป็น JSON object ที่ถูกต้องสมบูรณ์ โดยใช้ key ต่อไปนี้: "criteria", "info_needed", "source", "collection_method", "analysis_method" เนื้อหาต้องเป็นภาษาไทยที่กระชับและชัดเจน"""
        
        response = model.generate_content(
            [system_prompt, context_text],
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการเรียก AI: {e}")
        return None

# --- PDF Generation Function ---
class PDF(FPDF):
    def header(self):
        self.add_font('Sarabun', '', 'Sarabun-Regular.ttf', uni=True)
        self.set_font('Sarabun', 'B', 14)
        self.cell(0, 10, 'แผนและแนวการตรวจสอบ', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Sarabun', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf():
    # ... (PDF generation logic will be implemented here) ...
    # This part is complex and requires careful layouting. 
    # For now, we will create a placeholder.
    pdf = PDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font('Sarabun', '', 12)
    pdf.cell(0, 10, "ส่วนนี้กำลังอยู่ในระหว่างการพัฒนาฟังก์ชันสร้าง PDF", 0, 1)
    
    # Add content from session state
    plan_data = st.session_state.plan_gen_data
    pdf.multi_cell(0, 10, f"เรื่องที่ตรวจสอบ: {plan_data['general_info']['topic']}")
    
    for obj in plan_data['objectives']:
        pdf.multi_cell(0,10, f"\nวัตถุประสงค์: {obj['text']}")
        for issue in obj['issues']:
             pdf.multi_cell(0,10, f"  - ประเด็น: {issue['text']}")


    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return pdf_bytes

# --- UI Rendering ---

# 1. General Info
with st.form("general_info_form"):
    st.subheader("1. ข้อมูลทั่วไป")
    c1, c2 = st.columns(2)
    st.session_state.plan_gen_data["general_info"]["office"] = c1.text_input("สำนักงาน/จังหวัด/กลุ่ม", st.session_state.plan_gen_data["general_info"]["office"])
    st.session_state.plan_gen_data["general_info"]["topic"] = c1.text_input("เรื่องที่ตรวจสอบ", st.session_state.plan_gen_data["general_info"]["topic"])
    st.session_state.plan_gen_data["general_info"]["agency"] = c2.text_input("หน่วยงาน", st.session_state.plan_gen_data["general_info"]["agency"])
    st.session_state.plan_gen_data["general_info"]["ministry"] = c2.text_input("กระทรวง", st.session_state.plan_gen_data["general_info"]["ministry"])
    st.form_submit_button("บันทึกข้อมูลทั่วไป", use_container_width=True)


# 2. Objectives and Issues
st.subheader("2. วัตถุประสงค์และประเด็นการตรวจสอบ")
for i, obj in enumerate(st.session_state.plan_gen_data["objectives"]):
    with st.container(border=True):
        c1, c2 = st.columns([5, 1])
        obj['text'] = c1.text_area(f"วัตถุประสงค์ที่ {i+1}", obj['text'], key=f"obj_text_{i}")
        c2.button("🗑️ ลบ", key=f"del_obj_{i}", on_click=remove_objective, args=(i,), use_container_width=True)

        # Recursive function to display issues
        def display_issues(issues_list, obj_index, path):
            for j, issue in enumerate(issues_list):
                current_path = path + [j]
                level = len(current_path)
                prefix = ".".join(map(str, [k + 1 for k in current_path]))

                with st.container():
                    st.markdown(f"<div style='margin-left: {level * 20}px;'>", unsafe_allow_html=True)
                    issue['text'] = st.text_area(f"ประเด็นการตรวจสอบที่ {prefix}", issue['text'], key=f"issue_text_{obj_index}_{j}_{path}")

                    # If there are sub-issues, don't show AI/details
                    if not issue['issues']:
                        with st.expander("รายละเอียดแนวการตรวจสอบ (AI)"):
                            if st.button(f"🤖 ให้ AI ช่วยร่าง (ประเด็น {prefix})", key=f"ai_btn_{obj_index}_{j}_{path}"):
                                with st.spinner("AI กำลังประมวลผล..."):
                                    # Build context for AI
                                    context = f"เรื่องที่ตรวจสอบ: {st.session_state.plan_gen_data['general_info']['topic']}\n"
                                    context += f"วัตถุประสงค์: {obj['text']}\n"
                                    context += f"ประเด็นการตรวจสอบ: {issue['text']}"
                                    ai_result = call_gemini_api(context)
                                    if ai_result:
                                        issue['details'] = ai_result
                                        st.success("AI สร้างเนื้อหาเรียบร้อยแล้ว")

                            issue['details']['criteria'] = st.text_area("เกณฑ์การตรวจสอบ", issue['details']['criteria'], key=f"crit_{obj_index}_{j}_{path}")
                            issue['details']['info_needed'] = st.text_area("ข้อมูลที่ต้องการ", issue['details']['info_needed'], key=f"info_{obj_index}_{j}_{path}")
                            issue['details']['source'] = st.text_area("แหล่งข้อมูล", issue['details']['source'], key=f"src_{obj_index}_{j}_{path}")
                            issue['details']['collection_method'] = st.text_area("วิธีการรวบรวมหลักฐาน", issue['details']['collection_method'], key=f"coll_{obj_index}_{j}_{path}")
                            issue['details']['analysis_method'] = st.text_area("วิธีการวิเคราะห์หลักฐาน", issue['details']['analysis_method'], key=f"anal_{obj_index}_{j}_{path}")
                    
                    st.button(f"➕ เพิ่มประเด็นย่อย (สำหรับ {prefix})", key=f"add_sub_issue_{obj_index}_{j}_{path}", on_click=add_issue, args=(obj_index, current_path))
                    display_issues(issue['issues'], obj_index, current_path)
                    st.markdown("</div>", unsafe_allow_html=True)

        display_issues(obj['issues'], i, [])
        st.button("➕ เพิ่มประเด็นการตรวจสอบหลัก", key=f"add_issue_{i}", on_click=add_issue, args=(i, None))

st.button("➕ เพิ่มวัตถุประสงค์", on_click=add_objective, type="primary")

# 3. Estimates and Signatures
with st.form("estimates_signatures_form"):
    st.subheader("3. ประมาณการและผู้จัดทำ")
    sig_data = st.session_state.plan_gen_data["signatures"]

    st.text_area("ประมาณการค่าใช้จ่ายในการตรวจสอบ", key="cost_estimate")
    st.text_area("ประมาณการคน/วันที่ใช้ในการตรวจสอบ", key="effort_estimate")

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

# 4. Generate PDF
st.divider()
st.subheader("สร้างเอกสาร")
if st.button("📄 สร้างเอกสาร PDF (แนวนอน)", type="primary", use_container_width=True):
    with st.spinner("กำลังสร้างไฟล์ PDF..."):
        pdf_bytes = generate_pdf()
        st.download_button(
            label="✅ ดาวน์โหลด PDF สำเร็จ!",
            data=pdf_bytes,
            file_name=f"แผนการตรวจสอบ_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
