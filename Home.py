import streamlit as st
from style import load_css

# --- Page Configuration ---
st.set_page_config(
    page_title="PA Planning Studio",
    page_icon="🧭",
    layout="wide"
)

# --- Load CSS ---
load_css()

# --- Sidebar Content ---
with st.sidebar:
    # --- ย้าย Tabs มาไว้ในนี้ ---
    st.title("เมนูหลัก")
    
    tab1, tab2, tab3 = st.tabs([
        "🏳️ Design Assistant",
        "🧾 Plan Generator",
        "💬 PA Assistant Chat"
    ])

    with tab1:
        st.write("### Audit Design Assistant")
        st.caption("แนะนำประเด็นตรวจสอบที่น่าสนใจ")
        if st.button("ไปที่หน้า Design Assistant", key="btn_design"):
            st.switch_page("pages/2_Design_Assistant.py")

    with tab2:
        st.write("### Audit Plan Generator")
        st.caption("ช่วยร่างแผนและแนวการตรวจสอบ")
        if st.button("ไปที่หน้า Plan Generator", key="btn_plan"):
            st.switch_page("pages/3_Plan_Generator.py")
    
    with tab3:
        st.write("### PA Assistant Chat")
        st.caption("ผู้ช่วยอัจฉริยะ ถาม-ตอบ")
        if st.button("ไปที่หน้า PA Assistant Chat", key="btn_chat"):
            st.switch_page("pages/4_PA_Assistant_Chat.py")
            
    # --- Footer remains at the bottom ---
    st.markdown("""
        <div class="sidebar-footer">
            <p>
                <span style="color: grey;">By PAO1 </span><br>
                <span style="font-size: 16px; letter-spacing: 0.5px;">
                    <span style="color: red; font-weight: bold;">A</span>udit 
                    <span style="color: red; font-weight: bold;">I</span>ntelligence
                    <span style="color: red; font-weight: bold;">T</span>eam
                </span>
            </p>
        </div>
    """, unsafe_allow_html=True)


# --- Homepage Layout ---
st.title("🧭 Planning Studio – Performance Audit")
st.markdown(
    "<h3 class='subtitle'>⚒ Achieve More, Faster. Your Intelligent Efficiency Tools ᯓ★</h3>",
    unsafe_allow_html=True
)
st.info("กรุณาเลือกเมนูจากแถบด้านข้างเพื่อเริ่มต้นใช้งาน")
