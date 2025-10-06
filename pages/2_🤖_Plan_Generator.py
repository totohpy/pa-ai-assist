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

# --- AI Function (Adopted from pa_assistant) ---
def call_typhoon_api(context_text):
    try:
        # Ensure the global API key is loaded from secrets
        if "api_key" not in st.secrets:
            st.session_state.ui_feedback_message = ("error", "ไม่พบ API Key ใน Streamlit Secrets")
            return None
        
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
            model="typhoon-v2.1-12b-instruct", 
            messages=messages, 
            temperature=0.5
        )
        
        generated_text = response.choices[0].message.content
        
        # More robust JSON parsing
        match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                st.session_state.ui_feedback_message = ("error", f"AI ไม่ได้ส่งข้อมูลกลับมาในรูปแบบ JSON ที่ถูกต้อง")
                return None
        else:
            st.session_state.ui_feedback_message = ("error", f"AI ไม่ได้ส่งข้อมูลกลับมาในรูปแบบ JSON ที่คาดหวัง:\n{generated_text}")
            return None

    except Exception as e:
        st.session_state.ui_feedback_message = ("error", f"เกิดข้อผิดพลาดในการเรียก Typhoon AI: {e}")
        return None

# --- HTML Generation Function ---
def generate_html_report():
    plan_data = st.session_state.plan_gen_data

    def escape(text):
        return html.escape(str(text) if text is not None else "")

    def render_issues_html(issues_list, prefix_num):
        if not issues_list: return ""
        html_out = "<ul>"
        for i, issue in enumerate(issues_list):
            current_prefix = f"{prefix_num}.{i+1}"
            html_out += f"<li><strong>ประเด็น {current_prefix}:</strong> {escape(issue.get('text', ''))}"
            
            if not issue.get('issues'):
                details = issue.get('details', {})
                html_out += "<div class='issue-details'>"
                html_out += f"<p><strong>เกณฑ์:</strong> {escape(details.get('criteria', ''))}</p>"
                html_out += f"<p><strong>ข้อมูลที่ต้องการ:</strong> {escape(details.get('info_needed', ''))}</p>"
                html_out += f"<p><strong>แหล่งข้อมูล:</strong> {escape(details.get('source', ''))}</p>"
                html_out += f"<p><strong>วิธีรวบรวม:</strong> {escape(details.get('collection_method', ''))}</p>"
                html_out += f"<p><strong>วิธีวิเคราะห์:</strong> {escape(details.get('analysis_method', ''))}</p>"
                html_out += "</div>"
            
            if issue.get('issues'):
                html_out += render_issues_html(issue['issues'], current_prefix)
            
            html_out += "</li>"
        html_out += "</ul>"
        return html_out

    objectives_html = ""
    for i, obj in enumerate(plan_data['objectives']):
        objectives_html += f"<div class='objective-block'><h3>วัตถุประสงค์ที่ {i+1}: {escape(obj.get('text', ''))}</h3>"
        objectives_html += render_issues_html(obj.get('issues', []), str(i+1))
        objectives_html += "</div>"
    
    sig = plan_data['signatures']
    date_format = lambda d: d.strftime('%d/%m/%Y') if d else ''

    report_html = f"""
    <!DOCTYPE html><html lang="th"><head><meta charset="UTF-8"><title>แผนและแนวการตรวจสอบ</title>
    <link href="https://fonts.googleapis.com/css2?family=Sarabun:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Sarabun', sans-serif; background-color: #ffffff; margin: 0; padding: 0; -webkit-print-color-adjust: exact; }}
        .page {{ background: white; width: 29.7cm; min-height: 21cm; padding: 2cm; margin: 1cm auto; border: 1px #D3D3D3 solid; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); box-sizing: border-box; }}
        h1, h3 {{ margin-top: 0; }} .header-info p {{ margin: 4px 0; }}
        .section-title {{ font-weight: bold; border-bottom: 1px solid #999; padding-bottom: 4px; margin: 20px 0 10px 0; }}
        ul {{ list-style-type: none; padding-left: 25px; }} li {{ margin-bottom: 12px; }}
        .issue-details {{ padding-left: 20px; font-size: 0.95em; color: #333; }} .issue-details p {{ margin: 3px 0; }}
        .signature-table {{ width: 100%; border-collapse: collapse; margin-top: 25px; table-layout: fixed; }}
        .signature-table th, .signature-table td {{ border: 1px solid #000; padding: 8px; text-align: left; vertical-align: top; word-wrap: break-word; }}
        .signature-table th {{ text-align: center; font-weight: bold; }} .signature-table td {{ height: 120px; }}
        @media print {{ body, .page {{ margin: 0; box-shadow: none; border: none; }} @page {{ size: A4 landscape; margin: 2cm; }} }}
    </style></head><body><div class="page">
        <h1 style="text-align: center;">แผนและแนวการตรวจสอบ</h1>
        <div class="header-info">
            <p><strong>เรื่องที่ตรวจสอบ:</strong> {escape(plan_data['general_info']['topic'])}</p>
            <p><strong>หน่วยงาน:</strong> {escape(plan_data['general_info']['agency'])} &nbsp;&nbsp;<strong>กระทรวง:</strong> {escape(plan_data['general_info']['ministry'])}</p>
            <p><strong>สำนักงาน:</strong> {escape(plan_data['general_info']['office'])}</p>
        </div>
        <div class="section-title">วัตถุประสงค์และประเด็นการตรวจสอบ</div>{objectives_html}
        <div class="section-title">ประมาณการและผู้จัดทำ</div>
        <p><strong>ประมาณการค่าใช้จ่าย:</strong> {escape(plan_data['estimates']['cost'])}</p>
        <p><strong>ประมาณการคน/วัน:</strong> {escape(plan_data['estimates']['effort'])}</p>
        <table class="signature-table"><thead><tr><th>ผู้จัดทำ</th><th>ผู้สอบทาน</th><th>ผู้อนุมัติ (รผต. / ผอ. สำนัก)</th></tr></thead>
            <tbody><tr>
                <td><strong>ลงชื่อ:</strong> {escape(sig['maker']['name'])}<br><strong>ตำแหน่ง:</strong> {escape(sig['maker']['position'])}<br><strong>วันที่:</strong> {date_format(sig['maker']['date'])}<br><strong>ความเห็น:</strong> {escape(sig['maker']['comment'])}</td>
                <td><strong>ลงชื่อ:</strong> {escape(sig['reviewer']['name'])}<br><strong>ตำแหน่ง:</strong> {escape(sig['reviewer']['position'])}<br><strong>วันที่:</strong> {date_format(sig['reviewer']['date'])}<br><strong>ความเห็น:</strong> {escape(sig['reviewer']['comment'])}</td>
                <td><strong>ลงชื่อ:</strong> {escape(sig['approver']['name'])}<br><strong>ตำแหน่ง:</strong> {escape(sig['approver']['position'])}<br><strong>วันที่:</strong> {date_format(sig['approver']['date'])}<br><strong>ความเห็น:</strong> {escape(sig['approver']['comment'])}</td>
            </tr></tbody></table></div></body></html>
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
                    
                    target_container = st.session_state.plan_gen_data["objectives"][obj_index]
                    for index in path: target_container = target_container["issues"][index]
                    
                    target_container["issues"][j]['text'] = st.text_area(f"ประเด็นการตรวจสอบที่ {prefix}", value=issue.get('text', ''), key=f"issue_text_{key_suffix}")

                    if not issue.get('issues'):
                        with st.expander("รายละเอียดแนวการตรวจสอบ (AI)"):
                            if st.button(f"🤖 ให้ AI ช่วยร่าง (ประเด็น {prefix})", key=f"ai_btn_{key_suffix}"):
                                with st.spinner("AI กำลังประมวลผล..."):
                                    context = f"เรื่องที่ตรวจสอบ: {st.session_state.plan_gen_data['general_info']['topic']}\nวัตถุประสงค์: {obj.get('text', '')}\nประเด็นการตรวจสอบ: {issue.get('text', '')}"
                                    ai_result = call_typhoon_api(context)
                                    if ai_result:
                                        target_container["issues"][j]['details'] = ai_result
                                        st.session_state.ui_feedback_message = ("success", f"AI สร้างเนื้อหาสำหรับประเด็น {prefix} เรียบร้อยแล้ว")
                                        st.rerun()
                            
                            details = issue.get('details', {})
                            target_container["issues"][j]['details']['criteria'] = st.text_area("เกณฑ์การตรวจสอบ", value=details.get('criteria', ''), key=f"crit_{key_suffix}")
                            target_container["issues"][j]['details']['info_needed'] = st.text_area("ข้อมูลที่ต้องการ", value=details.get('info_needed', ''), key=f"info_{key_suffix}")
                            target_container["issues"][j]['details']['source'] = st.text_area("แหล่งข้อมูล", value=details.get('source', ''), key=f"src_{key_suffix}")
                            target_container["issues"][j]['details']['collection_method'] = st.text_area("วิธีการรวบรวมหลักฐาน", value=details.get('collection_method', ''), key=f"coll_{key_suffix}")
                            target_container["issues"][j]['details']['analysis_method'] = st.text_area("วิธีการวิเคราะห์หลักฐาน", value=details.get('analysis_method', ''), key=f"anal_{key_suffix}")
                    
                    st.button(f"➕ เพิ่มประเด็นย่อย (สำหรับ {prefix})", key=f"add_sub_issue_{key_suffix}", on_click=add_issue, args=(obj_index, current_path))
                    
                    if issue.get('issues'): display_issues(issue['issues'], obj_index, current_path)
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
        with st.expander("แสดงตัวอย่างเอกสาร (คลิกขวาเพื่อ Print/Save as PDF)", expanded=True):
            st.components.v1.html(report_html, height=800, scrolling=True)
            st.info("คุณสามารถคลิกขวาบนพื้นที่เอกสารด้านบน แล้วเลือก 'Print...' จากนั้นเลือก 'Save as PDF' เพื่อบันทึกไฟล์")

