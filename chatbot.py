import streamlit as st
import json
import fitz  # For PDF handling
from openai import OpenAI
import numpy as np

# --- Page Setup ---
st.set_page_config(page_title="E-Composer（ESG Environmental Composer）", layout="wide")
st.title("🌱 Production Sustainability Chatbot")
st.write("Let’s assess your sustainability efforts! Upload an ESG report (sidebar) or answer a few questions.")

# --- OpenAI Setup ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except:
    st.warning("AI features (report analysis) disabled. Use manual input.")
    OPENAI_AVAILABLE = False

# --- Session State ---
if "chat" not in st.session_state:
    st.session_state["chat"] = {
        "history": [],
        "mode": None,  # "upload" or "manual"
        "data": {
            "company": "",
            "industry": "",
            # Theme 1: Resource Efficiency (aligns with 12.2)
            "resource_use": {
                "renewable_energy_pct": 0,
                "water_reuse_pct": 0,
                "energy_saving_tech": 0,  # Number of technologies
                "extra_resource": ""      # User's additional info
            },
            # Theme 2: Waste & Materials (aligns with 12.3)
            "materials_waste": {
                "recycled_materials_pct": 0,
                "waste_reduction_pct": 0,  # vs previous year
                "eco_certified_products": False,
                "extra_materials": ""
            },
            # Theme 3: Circular Practices (aligns with 12.5)
            "circular_practices": {
                "product_takeback_pct": 0,  # % of product lines
                "sustainable_packaging_pct": 0,
                "supplier_sustainability_pct": 0,  # % of suppliers
                "extra_circular": ""
            },
            "total_score": 0,
            "breakdown": {},
            "recommendations": []
        },
        "manual_round": 1,  # 4 rounds (intro + 3 themes)
        "completed": False,
        "pdf_text": ""
    }

# --- Scoring Framework (Hidden SDG Alignment) ---
THEMES = [
    {
        "name": "Resource Use Efficiency",
        "weight": 0.3,
        "metrics": [
            {"key": "renewable_energy_pct", "calc": lambda x: 10 if x >=50 else 5 if x >=30 else 0},
            {"key": "water_reuse_pct", "calc": lambda x: 10 if x >=70 else 5 if x >=40 else 0},
            {"key": "energy_saving_tech", "calc": lambda x: min(10, x * 5)}  # 0→0, 1→5, 2+→10
        ],
        "max_score": 30
    },
    {
        "name": "Materials & Waste Management",
        "weight": 0.3,
        "metrics": [
            {"key": "recycled_materials_pct", "calc": lambda x: 10 if x >=40 else 5 if x >=20 else 0},
            {"key": "waste_reduction_pct", "calc": lambda x: 10 if x >=30 else 5 if x >=15 else 0},
            {"key": "eco_certified_products", "calc": lambda x: 10 if x else 0}
        ],
        "max_score": 30
    },
    {
        "name": "Circular Economy Practices",
        "weight": 0.4,
        "metrics": [
            {"key": "product_takeback_pct", "calc": lambda x: 10 if x >=50 else 5 if x >=20 else 0},
            {"key": "sustainable_packaging_pct", "calc": lambda x: 10 if x >=80 else 5 if x >=50 else 0},
            {"key": "supplier_sustainability_pct", "calc": lambda x: 10 if x >=60 else 5 if x >=30 else 0}
        ],
        "max_score": 30
    }
]

# --- Core Functions ---
def extract_pdf_text(uploaded_file):
    """Extract text from large PDFs (capped at 150k chars)"""
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)[:150000]
        return text
    except:
        st.error("Could not read PDF. Try manual input.")
        return ""

def ai_extract_data(text, industry):
    """Extract metrics from PDF without SDG jargon"""
    if not OPENAI_AVAILABLE:
        return None

    prompt = f"""analyse environmental or production related data from this report (industry: {industry}), feel free to estimate the percentages and information for the complete text below if there is no direct data.
    Return JSON with:
    - company: name (string)
    - resource_use: {{renewable_energy_pct (0-100), water_reuse_pct (0-100), energy_saving_tech (0+), extra_resource (string)}}
    - materials_waste: {{recycled_materials_pct (0-100), waste_reduction_pct (0-100), eco_certified_products (true/false), extra_materials (string)}}
    - circular_practices: {{product_takeback_pct (0-100), sustainable_packaging_pct (0-100), supplier_sustainability_pct (0-100), extra_circular (string)}}
    Use 0/False/"" for missing data. NO extra text."""

    try:
        response = client.chat.completions.create(
            model=""gpt-5"",
            messages=[{"role": "user", "content": f"{prompt}\nReport: {text}"}],
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content.strip())
    except:
        return None

def calculate_score():
    """Calculate total score from collected data"""
    data = st.session_state["chat"]["data"]
    breakdown = {}
    total = 0

    for theme in THEMES:
        theme_data = data[theme["name"].lower().replace(" ", "_")]
        score = sum(metric["calc"](theme_data[metric["key"]]) for metric in theme["metrics"])
        weighted = round(score * theme["weight"], 1)
        breakdown[theme["name"]] = {"score": weighted, "max": theme["max_score"] * theme["weight"]}
        total += weighted

    data["breakdown"] = breakdown
    data["total_score"] = min(100, round(total, 1))

def generate_recommendations():
    """Generate natural recommendations, introduce SDG terms here"""
    data = st.session_state["chat"]["data"]
    if not OPENAI_AVAILABLE:
        return [
            "Increase renewable energy use to reduce reliance on fossil fuels.",
            "Expand recycled material adoption to lower waste.",
            "Develop product take-back programs to extend product lifecycles."
        ]

    prompt = f"""Recommend 3 sustainability actions for {data['company']} ({data['industry']}).
    Scores: {data['breakdown']}
    Context: {data['resource_use']['extra_resource']}; {data['materials_waste']['extra_materials']}; {data['circular_practices']['extra_circular']}
    Link to UN Sustainable Development Goal 12 (responsible consumption/production) targets.
    Use simple language, no jargon. Include measurable goals."""

    response = client.chat.completions.create(
        model=""gpt-5"",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return [line.strip() for line in response.choices[0].message.content.split("\n") if line.strip()]

# --- Chat Helpers ---
def add_msg(role, content):
    st.session_state["chat"]["history"].append({"role": role, "content": content})

def display_chat():
    for msg in st.session_state["chat"]["history"]:
        with st.chat_message(msg["role"], avatar="🤖" if role == "assistant" else "👤"):
            st.write(msg["content"])

# --- Main Logic ---
chat = st.session_state["chat"]

# PDF Upload (Sidebar)
st.sidebar.subheader("Upload ESG Report (PDF)")
uploaded_pdf = st.sidebar.file_uploader("Select PDF", type="pdf")

if uploaded_pdf and chat["mode"] != "upload_industry":
    with st.spinner("Extracting report data..."):
        chat["pdf_text"] = extract_pdf_text(uploaded_pdf)
        if chat["pdf_text"]:
            chat["mode"] = "upload_industry"
            add_msg("user", "I uploaded our sustainability report.")
            add_msg("assistant", "Thanks! To better analyze, what industry is your company in?")
        st.rerun()

# Upload Flow: Ask Industry
if chat["mode"] == "upload_industry" and not chat["completed"]:
    display_chat()
    industry = st.chat_input("Your industry (e.g., Manufacturing, Retail):")
    if industry:
        add_msg("user", industry)
        with st.spinner("Analyzing report..."):
            extracted = ai_extract_data(chat["pdf_text"], industry)
            if extracted:
                chat["data"] = {**chat["data"],** extracted, "industry": industry}
                add_msg("assistant", f"Got it, {chat['data']['company']}! Anything else to share about your resource use, materials, or circular practices?")
                chat["mode"] = "upload_final"
            else:
                add_msg("assistant", "Let's try manual input instead. What's your company name?")
                chat["mode"] = "manual"
        st.rerun()

# Upload Flow: Final Input
if chat["mode"] == "upload_final" and not chat["completed"]:
    display_chat()
    extra = st.chat_input("Add any extra details (or 'done'):")
    if extra:
        add_msg("user", extra if extra != "done" else "No extra details.")
        chat["data"]["circular_practices"]["extra_circular"] += f" {extra}"
        calculate_score()
        chat["data"]["recommendations"] = generate_recommendations()
        chat["completed"] = True
        st.rerun()

# Manual Flow: 4 Rounds (Themes)
if chat["mode"] == "manual" and not chat["completed"]:
    display_chat()

    # Round 1: Intro (Company + Industry)
    if chat["manual_round"] == 1:
        company = st.chat_input("Let's start with your company name:")
        if company:
            add_msg("user", company)
            chat["data"]["company"] = company
            chat["manual_round"] = 2
            add_msg("assistant", f"Nice to meet you, {company}! What industry are you in?")
            st.rerun()

    # Round 2: Resource Use (12.2 theme)
    elif chat["manual_round"] == 2:
        if chat["data"]["industry"] == "":  # Collect industry first
            industry = st.chat_input("Your industry:")
            if industry:
                add_msg("user", industry)
                chat["data"]["industry"] = industry
                add_msg("assistant", """Let's talk about resource use:
- What % of your energy comes from renewable sources (e.g., solar, wind)?
- What % of water do you reuse or recycle?
- How many energy-saving technologies do you use (e.g., LED, solar panels)?
Feel free to add anything else about resource efficiency!""")
                st.rerun()
        else:  # Collect resource metrics
            res_info = st.chat_input("Share your resource use details:")
            if res_info:
                add_msg("user", res_info)
                # Parse numbers (simplified; in practice, use NLP)
                import re
                chat["data"]["resource_use"]["renewable_energy_pct"] = int(re.findall(r"(\d+)%.*energy", res_info)[0]) if re.findall(r"(\d+)%.*energy", res_info) else 0
                chat["data"]["resource_use"]["water_reuse_pct"] = int(re.findall(r"(\d+)%.*water", res_info)[0]) if re.findall(r"(\d+)%.*water", res_info) else 0
                chat["data"]["resource_use"]["energy_saving_tech"] = int(re.findall(r"(\d+).*energy-saving tech", res_info)[0]) if re.findall(r"(\d+).*energy-saving tech", res_info) else 0
                chat["data"]["resource_use"]["extra_resource"] = res_info
                chat["manual_round"] = 3
                add_msg("assistant", """Great! Next, let's discuss materials and waste:
- What % of your materials are recycled or upcycled?
- By what % have you reduced waste compared to last year?
- Do you have eco-certified products? (yes/no)
Add any other details about materials/waste!""")
                st.rerun()

    # Round 3: Materials & Waste (12.3 theme)
    elif chat["manual_round"] == 3:
        mat_info = st.chat_input("Share your materials/waste details:")
        if mat_info:
            add_msg("user", mat_info)
            import re
            chat["data"]["materials_waste"]["recycled_materials_pct"] = int(re.findall(r"(\d+)%.*recycled materials", mat_info)[0]) if re.findall(r"(\d+)%.*recycled materials", mat_info) else 0
            chat["data"]["materials_waste"]["waste_reduction_pct"] = int(re.findall(r"(\d+)%.*reduce waste", mat_info)[0]) if re.findall(r"(\d+)%.*reduce waste", mat_info) else 0
            chat["data"]["materials_waste"]["eco_certified_products"] = "yes" in mat_info.lower()
            chat["data"]["materials_waste"]["extra_materials"] = mat_info
            chat["manual_round"] = 4
            add_msg("assistant", """Thanks! Finally, let's cover circular practices:
- What % of your product lines have take-back/recycling programs?
- What % of your packaging is sustainable (recyclable/compostable)?
- What % of your suppliers follow sustainable practices?
Add any other circular economy efforts!""")
            st.rerun()

    # Round 4: Circular Practices (12.5 theme) + Open End
    elif chat["manual_round"] == 4:
        circ_info = st.chat_input("Share your circular practices details:")
        if circ_info:
            add_msg("user", circ_info)
            import re
            chat["data"]["circular_practices"]["product_takeback_pct"] = int(re.findall(r"(\d+)%.*product take-back", circ_info)[0]) if re.findall(r"(\d+)%.*product take-back", circ_info) else 0
            chat["data"]["circular_practices"]["sustainable_packaging_pct"] = int(re.findall(r"(\d+)%.*packaging", circ_info)[0]) if re.findall(r"(\d+)%.*packaging", circ_info) else 0
            chat["data"]["circular_practices"]["supplier_sustainability_pct"] = int(re.findall(r"(\d+)%.*suppliers", circ_info)[0]) if re.findall(r"(\d+)%.*suppliers", circ_info) else 0
            chat["data"]["circular_practices"]["extra_circular"] = circ_info
            calculate_score()
            chat["data"]["recommendations"] = generate_recommendations()
            chat["completed"] = True
            st.rerun()

# Show Results
if chat["completed"]:
    display_chat()
    data = chat["data"]
    with st.chat_message("assistant"):
        st.subheader(f"Your Sustainability Performance: {data['total_score']}/100")
        st.write("### Performance Breakdown")
        for theme, stats in data["breakdown"].items():
            st.write(f"- {theme}: {stats['score']}/{stats['max']}")
        st.write("### Recommendations")
        for i, rec in enumerate(data["recommendations"], 1):
            st.write(f"{i}. {rec}")
        st.download_button("Download Report", json.dumps(data, indent=2), "sustainability_report.txt")
        if st.button("Restart"):
            st.session_state["chat"] = {**st.session_state["chat"], "history": [], "mode": None, "completed": False, "manual_round": 1}
            st.rerun()

# Initial Prompt (No Mode)
if chat["mode"] is None and not chat["completed"] and not uploaded_pdf:
    add_msg("assistant", "Hi! I’m here to help assess your sustainability efforts. You can upload an ESG report (sidebar) or type 'start' to answer questions.")
    display_chat()
    if st.chat_input("Type 'start' to begin:") == "start":
        add_msg("user", "Let's start with questions.")
        chat["mode"] = "manual"
        add_msg("assistant", "Great! What's your company name?")
        st.rerun()
