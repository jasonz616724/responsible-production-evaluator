import streamlit as st
import json
import fitz  # PyMuPDF for PDF text extraction
from openai import OpenAI, APIError, Timeout
import re

# --- Page Configuration ---
st.set_page_config(page_title="Sustainability Chatbot", layout="wide")
st.title("🌱 Production Sustainability Evaluator")
st.write("Assess your sustainability efforts—upload an ESG report (sidebar) or answer a few questions!")

# --- OpenAI Client Setup ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except KeyError:
    st.warning("⚠️ OpenAI API key not found. AI features (report analysis) disabled.")
    OPENAI_AVAILABLE = False
except Exception as e:
    st.error(f"⚠️ OpenAI initialization failed: {str(e)}")
    OPENAI_AVAILABLE = False

# --- Session State Initialization ---
if "chat" not in st.session_state:
    st.session_state["chat"] = {
        "history": [],  # Stores conversation messages
        "mode": None,   # "upload" or "manual"
        "data": {
            "company": "",
            "industry": "",
            # Resource efficiency metrics (aligned with 12.2)
            "resource_use": {
                "renewable_energy_pct": 0,
                "water_reuse_pct": 0,
                "energy_saving_tech": 0,
                "extra_resource": ""
            },
            # Materials/waste metrics (aligned with 12.3)
            "materials_waste": {
                "recycled_materials_pct": 0,
                "waste_reduction_pct": 0,
                "eco_certified_products": False,
                "extra_materials": ""
            },
            # Circular economy metrics (aligned with 12.5)
            "circular_practices": {
                "product_takeback_pct": 0,
                "sustainable_packaging_pct": 0,
                "supplier_sustainability_pct": 0,
                "extra_circular": ""
            },
            "total_score": 0,
            "breakdown": {},
            "recommendations": []
        },
        "manual_round": 1,  # 4 rounds total
        "completed": False,
        "pdf_text": ""  # Stored PDF text for extraction
    }

# --- Scoring Framework (Hidden SDG Alignment) ---
THEMES = [
    {
        "name": "Resource Use Efficiency",
        "weight": 0.3,
        "metrics": [
            {"key": "renewable_energy_pct", "calc": lambda x: 10 if x >=50 else 5 if x >=30 else 0},
            {"key": "water_reuse_pct", "calc": lambda x: 10 if x >=70 else 5 if x >=40 else 0},
            {"key": "energy_saving_tech", "calc": lambda x: min(10, x * 5)}
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
    """Extract text from PDF (supports large files)"""
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            full_text = "\n".join(page.get_text() for page in doc)
            return full_text[:200000]  # Limit to 200k chars for GPT-5
    except Exception as e:
        st.error(f"⚠️ PDF extraction failed: {str(e)} (Is the file password-protected?)")
        return ""

def ai_extract_data(text, industry):
    """Extract sustainability data using GPT-5"""
    if not OPENAI_AVAILABLE:
        return None

    prompt = f"""analyse environmental or production related data from this report (industry: {industry}), feel free to estimate the percentages and information for the complete text below if there is no direct data.
    Return ONLY a valid JSON object with these fields:
    {{
        "company": "Company name (string, 'Unknown' if missing)",
        "resource_use": {{
            "renewable_energy_pct": number (0-100, % renewable energy; 0 if missing),
            "water_reuse_pct": number (0-100, % water reused; 0 if missing),
            "energy_saving_tech": number (count of energy-saving technologies; 0 if missing),
            "extra_resource": "Additional resource details (string, empty if none)"
        }},
        "materials_waste": {{
            "recycled_materials_pct": number (0-100, % recycled materials; 0 if missing),
            "waste_reduction_pct": number (0-100, % waste reduced vs last year; 0 if missing),
            "eco_certified_products": boolean (true/false for eco-certifications),
            "extra_materials": "Additional materials/waste details (string, empty if none)"
        }},
        "circular_practices": {{
            "product_takeback_pct": number (0-100, % product lines with take-back; 0 if missing),
            "sustainable_packaging_pct": number (0-100, % sustainable packaging; 0 if missing),
            "supplier_sustainability_pct": number (0-100, % sustainable suppliers; 0 if missing),
            "extra_circular": "Additional circular economy details (string, empty if none)"
        }}
    }}"""

    try:
        response = client.chat.completions.create(
            model="gpt-5",  # Use GPT-5
            messages=[
                {"role": "system", "content": "You are a precise data extractor. Return only JSON."},
                {"role": "user", "content": f"{prompt}\n\nReport Text:\n{text}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=60
        )
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        st.error("⚠️ Could not parse GPT-5's response. Using manual input.")
        return None
    except (APIError, Timeout) as e:
        st.error(f"⚠️ GPT-5 error: {str(e)}. Try manual input.")
        return None
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {str(e)}. Using manual input.")
        return None

def calculate_score():
    """Calculate total sustainability score"""
    data = st.session_state["chat"]["data"]
    breakdown = {}
    total = 0

    for theme in THEMES:
        theme_key = theme["name"].lower().replace(" ", "_")
        theme_data = data[theme_key]
        score = sum(metric["calc"](theme_data[metric["key"]]) for metric in theme["metrics"])
        weighted_score = round(score * theme["weight"], 1)
        breakdown[theme["name"]] = {
            "score": weighted_score,
            "max": round(theme["max_score"] * theme["weight"], 1)
        }
        total += weighted_score

    data["breakdown"] = breakdown
    data["total_score"] = min(100, round(total, 1))

def generate_recommendations():
    """Generate sustainability recommendations"""
    data = st.session_state["chat"]["data"]
    if not OPENAI_AVAILABLE:
        return [
            "Increase renewable energy adoption to reduce environmental impact.",
            "Expand use of recycled materials to minimize waste.",
            "Develop product take-back programs to support circular economy."
        ]

    prompt = f"""Generate 3 actionable sustainability recommendations for {data['company']} ({data['industry']}).
    Use the following performance data: {data['breakdown']}
    Consider extra context: {data['resource_use']['extra_resource']}; {data['materials_waste']['extra_materials']}; {data['circular_practices']['extra_circular']}
    Link recommendations to responsible production goals. Keep language simple."""

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return [line.strip() for line in response.choices[0].message.content.split("\n") if line.strip()]
    except Exception:
        return ["Prioritize energy efficiency upgrades.", "Reduce waste through material recycling.", "Strengthen supplier sustainability standards."]

# --- Chat UI Helpers ---
def add_msg(role, content):
    """Add message to chat history"""
    st.session_state["chat"]["history"].append({"role": role, "content": content})

def display_chat():
    """Display chat history with avatars"""
    for msg in st.session_state["chat"]["history"]:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])

# --- Main Logic ---
chat = st.session_state["chat"]

# PDF Upload (Sidebar)
st.sidebar.subheader("📄 Upload ESG Report")
uploaded_pdf = st.sidebar.file_uploader("Select a PDF report", type="pdf")

# Handle new PDF upload
if uploaded_pdf and chat["mode"] != "upload_industry":
    with st.spinner("Extracting text from PDF..."):
        chat["pdf_text"] = extract_pdf_text(uploaded_pdf)
        if chat["pdf_text"]:
            chat["mode"] = "upload_industry"
            add_msg("user", "I uploaded our sustainability report.")
            add_msg("assistant", "Thanks! To better analyze, what industry is your company in? (e.g., Manufacturing, Retail)")
        st.rerun()

# Upload flow: Ask for industry
if chat["mode"] == "upload_industry" and not chat["completed"]:
    display_chat()
    industry = st.chat_input("Your industry:")
    if industry:
        add_msg("user", industry)
        with st.spinner("Analyzing report with GPT-5..."):
            extracted_data = ai_extract_data(chat["pdf_text"], industry)
            if extracted_data:
                chat["data"] = {**chat["data"],** extracted_data, "industry": industry}
                add_msg("assistant", f"Got it, {chat['data']['company']}! Any extra details about your sustainability efforts to share?")
                chat["mode"] = "upload_final"
            else:
                add_msg("assistant", "Let's try manual input instead. What's your company name?")
                chat["mode"] = "manual"
        st.rerun()

# Upload flow: Final input
if chat["mode"] == "upload_final" and not chat["completed"]:
    display_chat()
    extra = st.chat_input("Add extra details (or 'done'):")
    if extra:
        add_msg("user", extra if extra != "done" else "No extra details.")
        chat["data"]["circular_practices"]["extra_circular"] += f" {extra}"
        calculate_score()
        chat["data"]["recommendations"] = generate_recommendations()
        chat["completed"] = True
        st.rerun()

# Manual input flow
if chat["mode"] == "manual" and not chat["completed"]:
    display_chat()

    # Round 1: Company name
    if chat["manual_round"] == 1:
        company = st.chat_input("Let's start with your company name:")
        if company:
            add_msg("user", company)
            chat["data"]["company"] = company
            chat["manual_round"] = 2
            add_msg("assistant", f"Nice to meet you, {company}! What industry are you in?")
            st.rerun()

    # Round 2: Industry + Resource use
    elif chat["manual_round"] == 2:
        if not chat["data"]["industry"]:
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
        else:
            res_info = st.chat_input("Share your resource use details:")
            if res_info:
                add_msg("user", res_info)
                # Simple parsing with regex
                chat["data"]["resource_use"]["renewable_energy_pct"] = int(re.findall(r"(\d+)%.*energy", res_info)[0]) if re.findall(r"(\d+)%.*energy", res_info) else 0
                chat["data"]["resource_use"]["water_reuse_pct"] = int(re.findall(r"(\d+)%.*water", res_info)[0]) if re.findall(r"(\d+)%.*water", res_info) else 0
                chat["data"]["resource_use"]["energy_saving_tech"] = int(re.findall(r"(\d+).*energy-saving", res_info)[0]) if re.findall(r"(\d+).*energy-saving", res_info) else 0
                chat["data"]["resource_use"]["extra_resource"] = res_info
                chat["manual_round"] = 3
                add_msg("assistant", """Great! Next, materials and waste:
- What % of your materials are recycled or upcycled?
- By what % have you reduced waste compared to last year?
- Do you have eco-certified products? (yes/no)
Add any other details!""")
                st.rerun()

    # Round 3: Materials & waste
    elif chat["manual_round"] == 3:
        mat_info = st.chat_input("Share your materials/waste details:")
        if mat_info:
            add_msg("user", mat_info)
            chat["data"]["materials_waste"]["recycled_materials_pct"] = int(re.findall(r"(\d+)%.*recycled", mat_info)[0]) if re.findall(r"(\d+)%.*recycled", mat_info) else 0
            chat["data"]["materials_waste"]["waste_reduction_pct"] = int(re.findall(r"(\d+)%.*reduce waste", mat_info)[0]) if re.findall(r"(\d+)%.*reduce waste", mat_info) else 0
            chat["data"]["materials_waste"]["eco_certified_products"] = "yes" in mat_info.lower()
            chat["data"]["materials_waste"]["extra_materials"] = mat_info
            chat["manual_round"] = 4
            add_msg("assistant", """Thanks! Finally, circular practices:
- What % of your product lines have take-back/recycling programs?
- What % of your packaging is sustainable (recyclable/compostable)?
- What % of your suppliers follow sustainable practices?
Add any other circular efforts!""")
            st.rerun()

    # Round 4: Circular practices
    elif chat["manual_round"] == 4:
        circ_info = st.chat_input("Share your circular practices details:")
        if circ_info:
            add_msg("user", circ_info)
            chat["data"]["circular_practices"]["product_takeback_pct"] = int(re.findall(r"(\d+)%.*take-back", circ_info)[0]) if re.findall(r"(\d+)%.*take-back", circ_info) else 0
            chat["data"]["circular_practices"]["sustainable_packaging_pct"] = int(re.findall(r"(\d+)%.*packaging", circ_info)[0]) if re.findall(r"(\d+)%.*packaging", circ_info) else 0
            chat["data"]["circular_practices"]["supplier_sustainability_pct"] = int(re.findall(r"(\d+)%.*suppliers", circ_info)[0]) if re.findall(r"(\d+)%.*suppliers", circ_info) else 0
            chat["data"]["circular_practices"]["extra_circular"] = circ_info
            calculate_score()
            chat["data"]["recommendations"] = generate_recommendations()
            chat["completed"] = True
            st.rerun()

# Display results
if chat["completed"]:
    display_chat()
    data = st.session_state["chat"]["data"]
    with st.chat_message("assistant", avatar="🤖"):
        st.subheader(f"Your Sustainability Score: {data['total_score']}/100")
        
        st.write("### Performance Breakdown")
        for theme, stats in data["breakdown"].items():
            st.write(f"- {theme}: {stats['score']}/{stats['max']}")
        
        st.write("### Recommendations")
        for i, rec in enumerate(data["recommendations"], 1):
            st.write(f"{i}. {rec}")
        
        # Download report
        report = f"""Sustainability Report for {data['company']}
Industry: {data['industry']}
Total Score: {data['total_score']}/100

Breakdown:
{json.dumps(data['breakdown'], indent=2)}

Recommendations:
{chr(10).join([f"{i}. {r}" for i, r in enumerate(data['recommendations'], 1)])}
"""
        st.download_button(
            "Download Full Report",
            report,
            f"{data['company']}_sustainability_report.txt",
            "text/plain"
        )
        
        if st.button("Assess Another Company"):
            st.session_state["chat"] = {
                "history": [],
                "mode": None,
                "data": {
                    "company": "",
                    "industry": "",
                    "resource_use": {"renewable_energy_pct": 0, "water_reuse_pct": 0, "energy_saving_tech": 0, "extra_resource": ""},
                    "materials_waste": {"recycled_materials_pct": 0, "waste_reduction_pct": 0, "eco_certified_products": False, "extra_materials": ""},
                    "circular_practices": {"product_takeback_pct": 0, "sustainable_packaging_pct": 0, "supplier_sustainability_pct": 0, "extra_circular": ""},
                    "total_score": 0,
                    "breakdown": {},
                    "recommendations": []
                },
                "manual_round": 1,
                "completed": False,
                "pdf_text": ""
            }
            st.rerun()

# Initial prompt (no mode selected)
if chat["mode"] is None and not chat["completed"] and not uploaded_pdf:
    add_msg("assistant", "Hi! I can assess your sustainability efforts. Upload an ESG report (sidebar) or type 'start' to answer questions.")
    display_chat()
    if st.chat_input("Type 'start' to begin:") == "start":
        add_msg("user", "Let's start with questions.")
        chat["mode"] = "manual"
        add_msg("assistant", "Great! What's your company name?")
        st.rerun()
