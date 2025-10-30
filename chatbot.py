import streamlit as st
import json
import fitz  # PyMuPDF for PDF handling
from openai import OpenAI
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="SDG 12 Chatbot", layout="wide")
st.title("🌱 SDG 12 Production Responsibility Chatbot")
st.write("I can help assess your SDG 12 performance! Upload an ESG report (preferred) or I'll guide you through questions.")

# --- Initialize OpenAI Client ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except KeyError:
    st.warning("⚠️ OPENAI_API_KEY not found. AI features (report extraction) disabled.")
    OPENAI_AVAILABLE = False
except Exception as e:
    st.error(f"⚠️ OpenAI error: {str(e)}")
    OPENAI_AVAILABLE = False

# --- Session State Initialization ---
if "chat" not in st.session_state:
    st.session_state["chat"] = {
        "history": [],  # Stores conversation history
        "mode": None,  # "upload" or "manual"
        "data": {
            "company_name": "",
            "industry": "",
            "resource_efficiency": {"renewable_energy_pct": 0, "water_reuse_pct": 0, "energy_tech_count": 0},
            "sustainable_production": {"recycled_material_pct": 0, "waste_intensity_pct": 0, "eco_design_cert": False},
            "circular_economy": {"takeback_program_pct": 0, "packaging_sustainable_pct": 0, "certified_supplier_pct": 0},
            "additional_info": "",  # For final round
            "total_score": 0,
            "dimension_scores": {},
            "recommendations": []
        },
        "manual_round": 1,  # Tracks 6 manual input rounds
        "completed": False
    }

# --- Scoring Framework ---
DIMENSIONS = [
    {
        "id": "resource_efficiency",
        "name": "Resource Efficiency (SDG 12.2)",
        "weight": 0.3,
        "actions": [
            {"name": "renewable_energy_pct", "calc": lambda x: 10 if x >=50 else 5 if x >=30 else 0},
            {"name": "water_reuse_pct", "calc": lambda x: 10 if x >=70 else 5 if x >=40 else 0},
            {"name": "energy_tech_count", "calc": lambda x: min(10, x * 5)}
        ],
        "max_subtotal": 30
    },
    {
        "id": "sustainable_production",
        "name": "Sustainable Production (SDG 12.3)",
        "weight": 0.3,
        "actions": [
            {"name": "recycled_material_pct", "calc": lambda x: 10 if x >=40 else 5 if x >=20 else 0},
            {"name": "waste_intensity_pct", "calc": lambda x: 10 if x <=20 else 5 if x <=40 else 0},
            {"name": "eco_design_cert", "calc": lambda x: 10 if x else 0}
        ],
        "max_subtotal": 30
    },
    {
        "id": "circular_economy",
        "name": "Circular Economy (SDG 12.5)",
        "weight": 0.4,
        "actions": [
            {"name": "takeback_program_pct", "calc": lambda x: 10 if x >=50 else 5 if x >=20 else 0},
            {"name": "packaging_sustainable_pct", "calc": lambda x: 10 if x >=80 else 5 if x >=50 else 0},
            {"name": "certified_supplier_pct", "calc": lambda x: 10 if x >=60 else 5 if x >=30 else 0}
        ],
        "max_subtotal": 30
    }
]

# --- Core Functions ---
def extract_text_from_pdf(uploaded_file):
    """Extract text from large PDFs (up to 100k+ words)"""
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            # Return first 150k characters to respect token limits
            return text[:150000]
    except Exception as e:
        st.error(f"⚠️ PDF extraction failed: {str(e)}")
        return ""

def ai_extract_esg_data(text, industry):
    """Extract SDG 12 metrics from ESG report text"""
    if not OPENAI_AVAILABLE:
        return None

    prompt = f"""Extract SDG 12 production metrics from this ESG report (Industry: {industry}).
    Return ONLY a JSON object with these fields (use 0 for missing data, true/false for booleans):
    {{
        "company_name": "Company name (string)",
        "resource_efficiency": {{
            "renewable_energy_pct": number (0-100),
            "water_reuse_pct": number (0-100),
            "energy_tech_count": number (0+)
        }},
        "sustainable_production": {{
            "recycled_material_pct": number (0-100),
            "waste_intensity_pct": number (0+),
            "eco_design_cert": boolean
        }},
        "circular_economy": {{
            "takeback_program_pct": number (0-100),
            "packaging_sustainable_pct": number (0-100),
            "certified_supplier_pct": number (0-100)
        }}
    }}
    Report text: {text}"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",  # Handles large context
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"⚠️ Data extraction failed: {str(e)}")
        return None

def calculate_score():
    """Calculate SDG 12 score from collected data"""
    data = st.session_state["chat"]["data"]
    dimension_scores = {}
    total_score = 0

    for dim in DIMENSIONS:
        dim_data = data[dim["id"]]
        subtotal = sum([action["calc"](dim_data[action["name"]]) for action in dim["actions"]])
        weighted_score = round(subtotal * dim["weight"], 1)
        dimension_scores[dim["id"]] = {
            "name": dim["name"],
            "score": weighted_score,
            "max": round(dim["max_subtotal"] * dim["weight"], 1)
        }
        total_score += weighted_score

    data["dimension_scores"] = dimension_scores
    data["total_score"] = min(100, round(total_score, 1))
    st.session_state["chat"]["data"] = data

def generate_recommendations():
    """Generate tailored recommendations"""
    data = st.session_state["chat"]["data"]
    if not OPENAI_AVAILABLE:
        return [
            "1. Increase renewable energy adoption to improve SDG 12.2 performance.",
            "2. Enhance circular economy practices (e.g., product take-back programs).",
            "3. Reduce waste intensity through process optimization."
        ]

    prompt = f"""Generate 3 SDG 12 recommendations for {data['company_name']} ({data['industry']}).
    Scores: {data['dimension_scores']}
    Additional info: {data['additional_info']}
    Link each to a specific SDG 12 target and include measurable actions."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return [line.strip() for line in response.choices[0].message.content.split("\n") if line.strip()]

def format_feedback():
    """Format final feedback document"""
    data = st.session_state["chat"]["data"]
    feedback = f"SDG 12 Performance Report for {data['company_name']}\n"
    feedback += f"Industry: {data['industry']}\n"
    feedback += f"Total Score: {data['total_score']}/100\n\n"

    feedback += "Dimension Scores:\n"
    for dim in data["dimension_scores"].values():
        feedback += f"- {dim['name']}: {dim['score']}/{dim['max']}\n"

    feedback += "\nRecommendations:\n"
    for i, rec in enumerate(data["recommendations"], 1):
        feedback += f"{i}. {rec}\n"

    if data["additional_info"]:
        feedback += f"\nAdditional Context Provided: {data['additional_info']}"

    return feedback

# --- Chat Interface ---
def add_message(role, content):
    """Add message to chat history"""
    st.session_state["chat"]["history"].append({"role": role, "content": content})

def display_chat():
    """Display chat history"""
    for msg in st.session_state["chat"]["history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# --- Main Logic ---
chat = st.session_state["chat"]

# Initial state: Ask for upload or manual input
if not chat["mode"] and not chat["completed"]:
    add_message("assistant", "Would you like to upload an ESG report (I'll extract data automatically) or provide information manually?")
    display_chat()

    # Upload option (default)
    uploaded_file = st.chat_input("Upload your ESG report (PDF) or type 'manual' to enter info yourself")
    if uploaded_file is not None:
        if hasattr(uploaded_file, "name") and uploaded_file.name.endswith(".pdf"):
            add_message("user", "I uploaded my ESG report.")
            with st.spinner("Extracting data from your report..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
                if pdf_text:
                    # Get industry first for better extraction
                    add_message("assistant", "Thanks! What industry is your company in? (e.g., Manufacturing, Retail)")
                    chat["mode"] = "upload_waiting_industry"
                else:
                    add_message("assistant", "I couldn't extract text from the PDF. Let's try manual input instead?")
                    chat["mode"] = None
            st.rerun()
        elif uploaded_file.lower() == "manual":
            chat["mode"] = "manual"
            add_message("user", "I'll provide information manually.")
            add_message("assistant", "Great! Let's start with your company name?")
            st.rerun()

# Handle upload mode (waiting for industry)
elif chat["mode"] == "upload_waiting_industry" and not chat["completed"]:
    display_chat()
    industry = st.chat_input("Please enter your industry:")
    if industry:
        add_message("user", f"Our industry is {industry}.")
        with st.spinner("Analyzing your report..."):
            pdf_text = extract_text_from_pdf(st.session_state.get("uploaded_file"))  # Retrieve from state
            extracted_data = ai_extract_esg_data(pdf_text, industry)
            if extracted_data:
                chat["data"]["company_name"] = extracted_data.get("company_name", "Unknown")
                chat["data"]["industry"] = industry
                chat["data"]["resource_efficiency"] = extracted_data.get("resource_efficiency", {})
                chat["data"]["sustainable_production"] = extracted_data.get("sustainable_production", {})
                chat["data"]["circular_economy"] = extracted_data.get("circular_economy", {})
                
                add_message("assistant", f"I've extracted data for {chat['data']['company_name']}. Anything else you'd like to add about your sustainability efforts?")
                chat["mode"] = "upload_final"
            else:
                add_message("assistant", "I couldn't extract valid data. Let's switch to manual input. What's your company name?")
                chat["mode"] = "manual"
        st.rerun()

# Upload mode final step (additional info)
elif chat["mode"] == "upload_final" and not chat["completed"]:
    display_chat()
    additional_info = st.chat_input("Enter any additional details (or 'done' to finish):")
    if additional_info:
        add_message("user", additional_info if additional_info.lower() != "done" else "No additional info.")
        chat["data"]["additional_info"] = additional_info if additional_info.lower() != "done" else ""
        calculate_score()
        chat["data"]["recommendations"] = generate_recommendations()
        chat["completed"] = True
        st.rerun()

# Manual input mode (6 rounds)
elif chat["mode"] == "manual" and not chat["completed"]:
    display_chat()
    
    # Round 1: Company name
    if chat["manual_round"] == 1:
        company_name = st.chat_input("What's your company name?")
        if company_name:
            add_message("user", company_name)
            chat["data"]["company_name"] = company_name
            chat["manual_round"] = 2
            add_message("assistant", "Thanks! What industry are you in? (e.g., Manufacturing, Retail)")
            st.rerun()
    
    # Round 2: Industry
    elif chat["manual_round"] == 2:
        industry = st.chat_input("Your industry:")
        if industry:
            add_message("user", industry)
            chat["data"]["industry"] = industry
            chat["manual_round"] = 3
            add_message("assistant", "Great. Let's talk about resources: What percentage of your energy comes from renewable sources? (e.g., 40 for 40%)")
            st.rerun()
    
    # Round 3: Resource efficiency
    elif chat["manual_round"] == 3:
        energy_pct = st.chat_input("Renewable energy percentage:")
        if energy_pct and energy_pct.isdigit():
            add_message("user", f"{energy_pct}%")
            chat["data"]["resource_efficiency"]["renewable_energy_pct"] = int(energy_pct)
            chat["manual_round"] = 4
            add_message("assistant", "Got it. What percentage of water do you reuse/recycle?")
            st.rerun()
    
    # Round 4: More resource metrics
    elif chat["manual_round"] == 4:
        water_pct = st.chat_input("Water reuse percentage:")
        if water_pct and water_pct.isdigit():
            add_message("user", f"{water_pct}%")
            chat["data"]["resource_efficiency"]["water_reuse_pct"] = int(water_pct)
            chat["manual_round"] = 5
            add_message("assistant", "Thanks! How many energy-efficient technologies do you use? (e.g., solar panels, heat pumps)")
            st.rerun()
    
    # Round 5: Production metrics
    elif chat["manual_round"] == 5:
        tech_count = st.chat_input("Number of energy-efficient technologies:")
        if tech_count and tech_count.isdigit():
            add_message("user", tech_count)
            chat["data"]["resource_efficiency"]["energy_tech_count"] = int(tech_count)
            chat["manual_round"] = 6
            add_message("assistant", "Last question set: What percentage of your materials are recycled?")
            st.rerun()
    
    # Round 6: Final metrics + open ended
    elif chat["manual_round"] == 6:
        recycled_pct = st.chat_input("Recycled materials percentage:")
        if recycled_pct and recycled_pct.isdigit():
            add_message("user", f"{recycled_pct}%")
            chat["data"]["sustainable_production"]["recycled_material_pct"] = int(recycled_pct)
            
            # Collect remaining metrics in follow-up (conversational flow)
            add_message("assistant", "Thanks! Any other sustainability efforts you'd like to mention? (e.g., waste reduction, packaging initiatives)")
            chat["manual_round"] = 7
            st.rerun()
    
    # Open-ended final input
    elif chat["manual_round"] == 7:
        additional_info = st.chat_input("Your additional sustainability details:")
        if additional_info:
            add_message("user", additional_info)
            chat["data"]["additional_info"] = additional_info
            calculate_score()
            chat["data"]["recommendations"] = generate_recommendations()
            chat["completed"] = True
            st.rerun()

# Display results
if chat["completed"]:
    display_chat()
    data = chat["data"]
    
    with st.chat_message("assistant"):
        st.subheader(f"Your SDG 12 Performance Summary: {data['total_score']}/100")
        
        st.write("### Dimension Breakdown")
        for dim in data["dimension_scores"].values():
            st.write(f"- {dim['name']}: {dim['score']}/{dim['max']}")
        
        st.write("### Recommendations")
        for i, rec in enumerate(data["recommendations"], 1):
            st.write(f"{i}. {rec}")
        
        # Download report
        feedback_text = format_feedback()
        st.download_button(
            "Download Full Report",
            feedback_text,
            f"{data['company_name']}_SDG12_Report.txt",
            "text/plain"
        )
        
        # Restart option
        if st.button("Assess Another Company"):
            st.session_state["chat"] = {
                "history": [],
                "mode": None,
                "data": {
                    "company_name": "",
                    "industry": "",
                    "resource_efficiency": {"renewable_energy_pct": 0, "water_reuse_pct": 0, "energy_tech_count": 0},
                    "sustainable_production": {"recycled_material_pct": 0, "waste_intensity_pct": 0, "eco_design_cert": False},
                    "circular_economy": {"takeback_program_pct": 0, "packaging_sustainable_pct": 0, "certified_supplier_pct": 0},
                    "additional_info": "",
                    "total_score": 0,
                    "dimension_scores": {},
                    "recommendations": []
                },
                "manual_round": 1,
                "completed": False
            }
            st.rerun()
