import streamlit as st
from datetime import datetime
import json
import re
from openai import OpenAI
import os
import html

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Plan Generator")
st.title("🤖 AI Plan Generator")
st.markdown("เครื่องมือช่วยสร้างแผนและแนวการตรวจสอบ พร้อมระบบ AI ช่วยร่างเนื้อหา")

# --- Custom CSS for Styling ---
st.markdown("""
<style>
/* General expander button text */
div[data-testid="stExpander"] div[role="button"] p {
    font-size: 1.1rem;
}

/* Custom style for the AI section expander */
.ai-expander .st-emotion-cache-ff2938 {
    background-color: #e7f3ff; /* Light blue background */
    border: 1px solid #007bff; /* Blue border */
    border-radius: 0.5rem;
}
.ai-expander .st-emotion-cache-ff2938:hover {
    background-color: #d0e8ff; /* Slightly darker blue on hover */
}
.ai-expander .st-emotion-cache-ff2938 p {
    color: #004085; /* Darker blue text */
    font-weight: bold;
}

/* Custom style for the AI action button */
.ai-button-container .stButton > button {
    background-color: #d4edda; /* Light green background */
    color: #155724; /* Dark green text */
    border: 1px solid #c3e6cb;
    font-weight: bold;
    border-radius: 0.5rem;
}
.ai-button-container .stButton > button:hover {
    background-color: #c3e6cb;
    color: #155724;
    border-color: #b1dfbb;
}
</style>
""", unsafe_allow_html=True)


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
    if "ui_feedback_message" not in ss:
        ss.ui_feedback_message = None
init_plan_state()

# --- Helper Functions ---
def add_objective():
    new_obj = {"id": f"obj_{len(st.session_state.plan_gen_data['objectives']) + 1}", "text": "", "issues": []}
    st.session_state.plan_gen_data["objectives"].append(new_obj)
    st.session_state.ui_feedback_message = None

def remove_objective(obj_index):
    st.session_state.plan_gen_data["objectives"].pop(obj_index)
    st.session_state.ui_feedback_message = None

def add_issue(obj_index, parent_issue_path=None):
    obj = st.session_state.plan_gen_data["objectives"][obj_index]
    target_container = obj
    if parent_issue_path:
        for index in parent_issue_path:
            target_container = target_container["issues"][index]
    
    new_issue = {
        "id": f"issue_{obj_index}_{len(target_container['issues']) + 1}",
        "text": "",
        "details": {"criteria": "", "info_needed": "", "source": "", "collection_method": "", "analysis_method": ""},
        "issues": []
    }
    target_container["issues"].append(new_issue)
    st.session_state.ui_feedback_message = None

# --- AI Function ---
def run_ai_for_issue(obj_index, path, prefix):
    """Callback function to run AI and update session state."""
    st.session_state.ui_feedback_message = None # Clear previous messages
    try:
        if "api_key" not in st.secrets:
            st.session_state.ui_feedback_message = ("error", "ไม่พบ API Key ใน Streamlit Secrets")
            return

        api_key = st.secrets["api_key"]
        client = OpenAI(api_key=api_key, base_url="https://api.opentyphoon.ai/v1")

        # Navigate to the correct issue in the session state
        obj = st.session_state.plan_gen_data["objectives"][obj_index]
        target_issue = obj
        for index in path:
            target_issue = target_issue["issues"][index]

        context = f"เรื่องที่ตรวจสอบ: {st.session_state.plan_gen_data['general_info']['topic']}\nวัตถุประสงค์: {obj.get('text', '')}\nประเด็นการตรวจสอบ: {target_issue.get('text', '')}"
        
        full_prompt = f"""คุณคือผู้เชี่ยวชาญด้านการตรวจสอบภาครัฐ (State Auditor) 
วิเคราะห์ context ต่อไปนี้:
--- CONTEXT START ---
{context}
--- CONTEXT END ---
**คำสั่ง:**
สร้างเนื้อหาสำหรับแนวการตรวจสอบโดยละเอียด แล้วตอบกลับโดยใช้ XML tags ต่อไปนี้เท่านั้น:
<CRITERIA>...</CRITERIA>
<INFO_NEEDED>...</INFO_NEEDED>
<SOURCE>...</SOURCE>
<COLLECTION_METHOD>...</COLLECTION_METHOD>
<ANALYSIS_METHOD>...</ANALYSIS_METHOD>
"""
        
        messages = [{"role": "user", "content": full_prompt}]
        response = client.chat.completions.create(
            model="typhoon-v2.1-12b-instruct", messages=messages, temperature=0.5
        )
        generated_text = response.choices[0].message.content
        
        def extract_tag_content(tag, text):
            match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
            content = match.group(1).strip() if match else ""
            items = [item.strip() for item in content.split('\n') if item.strip()]
            return "\n".join([f"- {item.lstrip('- ')}" for item in items])

        details = {
            "criteria": extract_tag_content("CRITERIA", generated_text),
            "info_needed": extract_tag_content("INFO_NEEDED", generated_text),
            "source": extract_tag_content("SOURCE", generated_text),
            "collection_method": extract_tag_content("COLLECTION_METHOD", generated_text),
            "analysis_method": extract_tag_content("ANALYSIS_METHOD", generated_text),
        }

        if any(details.values()):
            target_issue['details'].update(details)
            st.session_state.ui_feedback_message = ("success", f"AI สร้างเนื้อหาสำหรับประเด็น {prefix} เรียบร้อยแล้ว")
        else:
            st.session_state.ui_feedback_message = ("error", f"AI ไม่สามารถแยกแยะข้อมูลจากข้อความที่สร้างขึ้นได้:\n{generated_text}")

    except Exception as e:
        st.session_state.ui_feedback_message = ("error", f"เกิดข้อผิดพลาดในการเรียก Typhoon AI: {e}")


# --- UI Rendering ---
if st.session_state.get("ui_feedback_message"):
    msg_type, msg_content = st.session_state.ui_feedback_message
    if msg_type == "success": st.success(msg_content)
    else: st.error(msg_content)
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
        st.session_state.plan_gen_data["objectives"][i]['text'] = c1.text_area(f"วัตถุประสงค์ที่ {i+1}", obj.get('text', ''), key=f"obj_text_{i}")
        c2.button("🗑️ ลบ", key=f"del_obj_{i}", on_click=remove_objective, args=(i,), use_container_width=True)

        def display_issues(issues_list, obj_index, path):
            for j, issue in enumerate(issues_list):
                current_path = path + [j]
                prefix = ".".join([str(obj_index + 1)] + [str(p + 1) for p in current_path])
                key_suffix = f"{obj_index}_{'_'.join(map(str, current_path))}"

                with st.container():
                    st.markdown(f"<div style='margin-left: {len(current_path) * 20}px;'>", unsafe_allow_html=True)
                    
                    target_issue = issue
                    target_issue['text'] = st.text_area(f"ประเด็นการตรวจสอบที่ {prefix}", value=target_issue.get('text', ''), key=f"issue_text_{key_suffix}")

                    if not target_issue.get('issues'):
                        st.markdown('<div class="ai-expander">', unsafe_allow_html=True)
                        with st.expander("เพิ่มรายละเอียดแนวการตรวจสอบ (ให้ AI ช่วย)"):
                            st.markdown('<div class="ai-button-container">', unsafe_allow_html=True)
                            
                            st.button(f"🤖 ให้ AI ช่วยร่าง (ประเด็น {prefix})", key=f"ai_btn_{key_suffix}", 
                                on_click=run_ai_for_issue, args=(obj_index, current_path, prefix))

                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            details = target_issue.get('details', {})
                            target_issue['details']['criteria'] = st.text_area("เกณฑ์การตรวจสอบ", value=details.get('criteria', ''), key=f"crit_{key_suffix}")
                            target_issue['details']['info_needed'] = st.text_area("ข้อมูลที่ต้องการ", value=details.get('info_needed', ''), key=f"info_{key_suffix}")
                            target_issue['details']['source'] = st.text_area("แหล่งข้อมูล", value=details.get('source', ''), key=f"src_{key_suffix}")
                            target_issue['details']['collection_method'] = st.text_area("วิธีการรวบรวมหลักฐาน", value=details.get('collection_method', ''), key=f"coll_{key_suffix}")
                            target_issue['details']['analysis_method'] = st.text_area("วิธีการวิเคราะห์หลักฐาน", value=details.get('analysis_method', ''), key=f"anal_{key_suffix}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    st.button(f"➕ เพิ่มประเด็นย่อย (สำหรับ {prefix})", key=f"add_sub_issue_{key_suffix}", on_click=add_issue, args=(obj_index, current_path))
                    
                    if target_issue.get('issues'): display_issues(target_issue['issues'], obj_index, current_path)
                    st.markdown("</div>", unsafe_allow_html=True)

        if obj.get('issues'): display_issues(obj['issues'], i, [])
        st.button("➕ เพิ่มประเด็นการตรวจสอบหลัก", key=f"add_issue_{i}", on_click=add_issue, args=(i, None))

st.button("➕ เพิ่มวัตถุประสงค์", on_click=add_objective, type="primary")

with st.form("estimates_signatures_form"):
    st.subheader("3. ประมาณการและผู้จัดทำ")
    st.session_state.plan_gen_data["estimates"]["cost"] = st.text_area("ประมาณการค่าใช้จ่ายในการตรวจสอบ", st.session_state.plan_gen_data["estimates"]["cost"])
    st.session_state.plan_gen_data["estimates"]["effort"] = st.text_area("ประมาณการคน/วันที่ใช้ในการตรวจสอบ", st.session_state.plan_gen_data["estimates"]["effort"])
    
    c1, c2, c3 = st.columns(3)
    sig_data = st.session_state.plan_gen_data["signatures"]
    
    with c1:
        st.markdown("**ผู้จัดทำ**")
        sig_data["maker"]["name"] = st.text_input("ลงชื่อ", value=sig_data["maker"].get("name", ""), key="maker_name")
        sig_data["maker"]["position"] = st.text_input("ตำแหน่ง", value=sig_data["maker"].get("position", ""), key="maker_pos")
        sig_data["maker"]["date"] = st.date_input("วันที่", value=sig_data["maker"].get("date"), key="maker_date")
        sig_data["maker"]["comment"] = st.text_area("ความเห็นเพิ่มเติม", value=sig_data["maker"].get("comment", ""), key="maker_comment")
    with c2:
        st.markdown("**ผู้สอบทาน**")
        sig_data["reviewer"]["name"] = st.text_input("ลงชื่อ", value=sig_data["reviewer"].get("name", ""), key="reviewer_name")
        sig_data["reviewer"]["position"] = st.text_input("ตำแหน่ง", value=sig_data["reviewer"].get("position", ""), key="reviewer_pos")
        sig_data["reviewer"]["date"] = st.date_input("วันที่", value=sig_data["reviewer"].get("date"), key="reviewer_date")
        sig_data["reviewer"]["comment"] = st.text_area("ความเห็นเพิ่มเติม", value=sig_data["reviewer"].get("comment", ""), key="reviewer_comment")
    with c3:
        st.markdown("**ผู้อนุมัติ (รผต. / ผอ. สำนัก)**")
        sig_data["approver"]["name"] = st.text_input("ลงชื่อ", value=sig_data["approver"].get("name", ""), key="approver_name")
        sig_data["approver"]["position"] = st.text_input("ตำแหน่ง", value=sig_data["approver"].get("position", ""), key="approver_pos")
        sig_data["approver"]["date"] = st.date_input("วันที่", value=sig_data["approver"].get("date"), key="approver_date")
        sig_data["approver"]["comment"] = st.text_area("ความเห็นเพิ่มเติม", value=sig_data["approver"].get("comment", ""), key="approver_comment")
        
    st.form_submit_button("บันทึกข้อมูลผู้จัดทำ", use_container_width=True)

# --- Document Generation Section (Commented out for now) ---
# st.divider()
# st.subheader("สร้างเอกสาร")
# if 'show_report' not in st.session_state:
#     st.session_state.show_report = False

# if st.button("📄 แสดง/ซ่อนตัวอย่างเอกสาร (HTML)", type="primary", use_container_width=True):
#     st.session_state.show_report = not st.session_state.show_report

# if st.session_state.show_report:
#     with st.spinner("กำลังสร้างตัวอย่างเอกสาร..."):
#         # You would need a function here to generate the HTML report
#         # report_html = generate_html_report() 
#         # with st.expander("แสดงตัวอย่างเอกสาร", expanded=True):
#         #     st.components.v1.html(report_html, height=800, scrolling=True)
#         st.info("ส่วนของการสร้างเอกสารถูกปิดใช้งานชั่วคราว")

