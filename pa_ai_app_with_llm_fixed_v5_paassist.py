# ----------------- START: Custom CSS for Styling and Responsiveness (ปรับปรุงแท็บและมือถือ) -----------------
st.markdown("""
<style>
/* 1. GLOBAL FONT/BACKGROUND ADJUSTMENTS */
body {
    font-family: 'Kanit', sans-serif;
}

/* 2. STYLE TABS AS COLORED BUTTONS (Custom Tabs) */

/* A. Container for all tabs: REMOVE the default bottom bar and set spacing */
div[data-baseweb="tab-list"] {
    border-bottom: none !important; /* *** กำจัดเส้นแนวนอนใต้แท็บเริ่มต้น (สำคัญ) *** */
    margin-bottom: 15px;
    flex-wrap: wrap; 
    gap: 10px; /* เพิ่มช่องว่างระหว่างปุ่ม */
}

/* B. Base style for ALL tab buttons */
button[data-baseweb="tab"] {
    /* Override Streamlit's default tab styling */
    border-bottom: none !important; /* กำจัดเส้นใต้ปุ่ม */
    
    /* Apply button-like styling */
    border: 1px solid #007bff; /* ขอบสีน้ำเงิน */
    border-radius: 8px; /* มุมโค้งมน */
    padding: 10px 15px;
    transition: all 0.3s;
    font-weight: bold;
    color: #007bff !important; /* สีตัวอักษรเริ่มต้น */
    background-color: white !important; /* พื้นหลังสีขาว */
    box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
    
    /* C. กำจัดส่วนขยายที่ Streamlit ใช้ในการทำไฮไลต์ใต้แท็บ (สำคัญมาก) */
    &::after {
        content: none !important;
    }
}

/* D. Style for the ACTIVE tab (when selected) */
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #007bff !important; /* พื้นหลังสีน้ำเงินเข้ม */
    color: white !important; /* ตัวอักษรสีขาว */
    border: 1px solid #007bff;
    box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
}

/* 3. MOBILE RESPONSIVENESS ADJUSTMENTS */
@media (max-width: 768px) {
    .st-emotion-cache-18ni2cb, .st-emotion-cache-1jm69l4, [data-testid="stColumn"] {
        width: 100% !important;
        margin-bottom: 1rem;
    }
}

/* 4. STYLE HEADERS */
h4 {
    color: #007bff !important;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)
# ----------------- END: Custom CSS -----------------
