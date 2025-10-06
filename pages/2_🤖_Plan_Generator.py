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
def call_typhoon_api(context_text):
    try:
        if "api_key" not in st.secrets:
            return None, ("error", "ไม่พบ API Key ใน Streamlit Secrets")
        
        api_key = st.secrets["api_key"]
        client = OpenAI(api_key=api_key, base_url="https://api.opentyphoon.ai/v1")

        full_prompt = f"""คุณคือผู้เชี่ยวชาญด้านการตรวจสอบภาครัฐ (State Auditor)... จงสร้างเนื้อหาสำหรับ... context ต่อไปนี้:\n--- CONTEXT START ---\n{context_text}\n--- CONTEXT END ---\n**คำสั่ง:**\nตอบกลับเป็น JSON object **เท่านั้น**..."""
        
        messages = [{"role": "user", "content": full_prompt}]
        response = client.chat.completions.create(
            model="typhoon-v2.1-12b-instruct", messages=messages, temperature=0.5
        )
        generated_text = response.choices[0].message.content
        
        match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if match:
            parsed_json = json.loads(match.group(0))
            formatted_json = {}
            for key, value in parsed_json.items():
                if isinstance(value, list):
                    formatted_json[key] = "\n".join([f"- {item}" for item in value])
                elif isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                    try:
                        list_value = eval(value)
                        formatted_json[key] = "\n".join([f"- {item}" for item in list_value]) if isinstance(list_value, list) else value
                    except:
                        cleaned_value = value.strip("[]'\" ")
                        items = [item.strip() for item in cleaned_value.replace("'", "").replace('"', '').split(',')]
                        formatted_json[key] = "\n".join([f"- {item}" for item in items])
                else:
                    formatted_json[key] = value
            return formatted_json, None
        return None, ("error", f"AI ไม่ได้ส่งข้อมูลกลับมาในรูปแบบ JSON ที่คาดหวัง:\n{generated_text}")

    except Exception as e:
        return None, ("error", f"เกิดข้อผิดพลาดในการเรียก Typhoon AI: {e}")

# --- HTML Generation Function ---
def generate_html_report():
    plan_data = st.session_state.plan_gen_data

    def escape(text):
        raw_text = str(text) if text is not None else ""
        return html.escape(raw_text).replace("\n", "<br>")

    def render_issues_html(issues_list, prefix_num):
        if not issues_list: return ""
        html_out = ""
        for i, issue in enumerate(issues_list):
            current_prefix = f"{prefix_num}.{i+1}"
            html_out += f"<h4>ประเด็นการตรวจสอบที่ {current_prefix}: {escape(issue.get('text', ''))}</h4>"
            
            if not issue.get('issues'):
                details = issue.get('details', {})
                html_out += "<table class='details-table'>"
                html_out += "<thead><tr><th>เกณฑ์การตรวจสอบ</th><th>ข้อมูลที่ต้องการ</th><th>แหล่งข้อมูล</th><th>วิธีการรวบรวมหลักฐาน</th><th>วิธีการวิเคราะห์หลักฐาน</th></tr></thead>"
                html_out += "<tbody><tr>"
                html_out += f"<td>{escape(details.get('criteria', ''))}</td>"
                html_out += f"<td>{escape(details.get('info_needed', ''))}</td>"
                html_out += f"<td>{escape(details.get('source', ''))}</td>"
                html_out += f"<td>{escape(details.get('collection_method', ''))}</td>"
                html_out += f"<td>{escape(details.get('analysis_method', ''))}</td>"
                html_out += "</tr></tbody></table>"
            
            if issue.get('issues'):
                html_out += render_issues_html(issue['issues'], current_prefix)
        return html_out

    objectives_html = ""
    for i, obj in enumerate(plan_data['objectives']):
        objectives_html += f"<div class='objective-block'><h3>วัตถุประสงค์การตรวจสอบที่ {i+1}: {escape(obj.get('text', ''))}</h3>"
        objectives_html += render_issues_html(obj.get('issues', []), str(i+1))
        objectives_html += "</div>"
        
    sig = plan_data['signatures']
    date_format = lambda d: d.strftime('%d/%m/%Y') if d else ''
    
    maker_text = f"ลงชื่อ: {sig['maker']['name']}\nตำแหน่ง: {sig['maker']['position']}\nวันที่: {date_format(sig['maker']['date'])}\nความเห็น: {sig['maker']['comment']}"
    reviewer_text = f"ลงชื่อ: {sig['reviewer']['name']}\nตำแหน่ง: {sig['reviewer']['position']}\nวันที่: {date_format(sig['reviewer']['date'])}\nความเห็น: {sig['reviewer']['comment']}"
    approver_text = f"ลงชื่อ: {sig['approver']['name']}\nตำแหน่ง: {sig['approver']['position']}\nวันที่: {date_format(sig['approver']['date'])}\nความเห็น: {sig['approver']['comment']}"

    report_html = f"""
    <!DOCTYPE html><html lang="th"><head><meta charset="UTF-8"><title>แผนและแนวการตรวจสอบ</title>
    <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Sarabun', sans-serif; background-color: #f0f2f5; margin: 0; padding: 0; -webkit-print-color-adjust: exact; }}
        .page {{ background: white; width: 29.7cm; min-height: 21cm; padding: 2cm; margin: 1cm auto; border: 1px #D3D3D3 solid; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); box-sizing: border-box; position: relative; }}
        h1, h3, h4 {{ margin-top: 0; font-weight: 700; }}
        .header-info p {{ margin: 4px 0; }}
        .details-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; margin-bottom: 20px; table-layout: auto; }}
        .details-table th, .details-table td {{ border: 1px solid #999; padding: 8px; text-align: left; vertical-align: top; word-wrap: break-word; }}
        .details-table th {{ background-color: #f2f2f2; font-weight: bold; }}
        .signature-table {{ width: 100%; border-collapse: collapse; margin-top: 25px; table-layout: fixed; }}
        .signature-table th, .signature-table td {{ width: 33.33%; border: 1px solid #000; padding: 8px; text-align: left; vertical-align: top; word-wrap: break-word; }}
        .signature-table th {{ text-align: center; font-weight: bold; }} .signature-table td {{ height: 120px; }}
        .print-button {{ position: absolute; top: 20px; right: 20px; padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-family: 'Sarabun', sans-serif; font-size: 16px; }}
        
        @media print {{
            body, .page {{ margin: 0; padding: 0; box-shadow: none; border: none; background: white; }}
            .print-button {{ display: none !important; }}
            #root > div:first-child, .stApp > header, .stApp .main > div:first-child, .stButton, .stDownloadButton, .stSpinner {{ display: none !important; }}
            .main .block-container {{ padding: 0 !important; margin: 0 !important; max-width: 100% !important; }}
            @page {{ size: A4 landscape; margin: 1.5cm; }}
        }}
    </style></head><body><div class="page">
        <button class="print-button" onclick="window.print()">🖨️ พิมพ์เอกสาร</button>
        <h1 style="text-align: center; font-weight: 700;">แผนและแนวการตรวจสอบ</h1>
        <div class="header-info">
            <p><strong>เรื่องที่ตรวจสอบ:</strong> {escape(plan_data['general_info']['topic'])}</p>
            <p><strong>หน่วยงาน:</strong> {escape(plan_data['general_info']['agency'])} &nbsp;&nbsp;<strong>กระทรวง:</strong> {escape(plan_data['general_info']['ministry'])}</p>
            <p><strong>สำนักงาน:</strong> {escape(plan_data['general_info']['office'])}</p>
        </div>
        {objectives_html}
        <p style="margin-top: 25px;"><strong>ประมาณการค่าใช้จ่ายในการตรวจสอบ:</strong> {escape(plan_data['estimates']['cost'])}</p>
        <p><strong>ประมาณการคน/วันที่ใช้ในการตรวจสอบ:</strong> {escape(plan_data['estimates']['effort'])}</p>
        <table class="signature-table"><thead><tr><th>ผู้จัดทำ</th><th>ผู้สอบทาน</th><th>ผู้อนุมัติ (รผต. / ผอ. สำนัก)</th></tr></thead>
            <tbody><tr>
                <td>{escape(maker_text)}</td>
                <td>{escape(reviewer_text)}</td>
                <td>{escape(approver_text)}</td>
            </tr></tbody></table>
        </div></body></html>
    """
    return report_html

# --- UI Rendering ---
if st.session_state.get("ui_feedback_message"):
    msg_type, msg_content = st.session_state.ui_feedback_message
    if msg_type == "success": st.success(msg_content)
    else: st.error(msg_content)
    st.session_state.ui_feedback_message = None

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
                            
                            if st.button(f"🤖 ให้ AI ช่วยร่าง (ประเด็น {prefix})", key=f"ai_btn_{key_suffix}"):
                                ai_result, error = call_typhoon_api(f"เรื่องที่ตรวจสอบ: {st.session_state.plan_gen_data['general_info']['topic']}\nวัตถุประสงค์: {obj.get('text', '')}\nประเด็นการตรวจสอบ: {target_issue.get('text', '')}")
                                if error:
                                    st.session_state.ui_feedback_message = error
                                else:
                                    target_issue['details'].update(ai_result)
                                    st.session_state.ui_feedback_message = ("success", f"AI สร้างเนื้อหาสำหรับประเด็น {prefix} เรียบร้อยแล้ว")
                                st.rerun()

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
if 'show_report' not in st.session_state:
    st.session_state.show_report = False

if st.button("📄 แสดง/ซ่อนตัวอย่างเอกสาร (HTML)", type="primary", use_container_width=True):
    st.session_state.show_report = not st.session_state.show_report

if st.session_state.show_report:
    with st.spinner("กำลังสร้างตัวอย่างเอกสาร..."):
        report_html = generate_html_report()
        with st.expander("แสดงตัวอย่างเอกสาร (คลิกปุ่ม 'พิมพ์เอกสาร' ด้านในเพื่อ Print/Save as PDF)", expanded=True):
            st.components.v1.html(report_html, height=800, scrolling=True)

