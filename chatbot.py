import streamlit as st
import json
import fitz
from openai import OpenAI
import re

# --- Page Setup ---
st.set_page_config(page_title="Sustainability Evaluator", layout="centered")
st.title("🌱 Production Sustainability Check")
st.write("Upload an ESG report or answer questions to assess your efforts.")

# --- OpenAI Setup (Simplified) ---
OPENAI_AVAILABLE = False
client = None
try:
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))
    OPENAI_AVAILABLE = True
except:
    st.warning("AI features disabled (no API key). Use manual input.")

# --- Session State (Simplified) ---
if "state" not in st.session_state:
    st.session_state.state = {
        "company": "",
        "industry": "",
        "step": "start",  # start → industry → questions → results
        "data": {
            "resource_use": {"renewable": 0, "water_reuse": 0, "energy_tech": 0, "extra": ""},
            "materials": {"recycled_pct": 0, "waste_reduction": 0, "eco_cert": False, "extra": ""},
            "circular": {"takeback": 0, "packaging": 0, "suppliers": 0, "extra": ""}
        },
        "score": 0,
        "recommendations": []
    }

# --- Scoring (Simplified) ---
def calculate_score(data):
    resource_score = (min(data["resource_use"]["renewable"], 100)/10 +
                     min(data["resource_use"]["water_reuse"], 100)/10 +
                     min(data["resource_use"]["energy_tech"], 2)*5) * 0.3
    
    materials_score = (min(data["materials"]["recycled_pct"], 100)/10 +
                      min(data["materials"]["waste_reduction"], 100)/10 +
                      10 if data["materials"]["eco_cert"] else 0) * 0.3
    
    circular_score = (min(data["circular"]["takeback"], 100)/10 +
                     min(data["circular"]["packaging"], 100)/10 +
                     min(data["circular"]["suppliers"], 100)/10) * 0.4
    
    return round(resource_score + materials_score + circular_score, 1)

# --- PDF Extraction (Simplified) ---
def extract_pdf_text(uploaded_file):
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)[:150000]
    except:
        st.error("Failed to read PDF. Use manual input.")
        return ""

# --- AI Data Extraction (Simplified with Estimation) ---
def ai_extract(text, industry):
    if not OPENAI_AVAILABLE:
        return None

    prompt = f"""Analyze this {industry} company's ESG report. 
    Extract/provide estimates for:
    - Renewable energy % (0-100)
    - Water reuse % (0-100)
    - Number of energy-saving technologies
    - Recycled materials % (0-100)
    - Waste reduction % vs last year (0-100)
    - Eco-certified products? (yes/no)
    - Product take-back program % (0-100)
    - Sustainable packaging % (0-100)
    - Sustainable suppliers % (0-100)
    Explain estimates briefly.
    Return as JSON with keys: renewable, water_reuse, energy_tech, recycled_pct, waste_reduction, eco_cert (bool), takeback, packaging, suppliers, extra."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"{prompt}\n\nReport: {text}"}],
            temperature=0.3,
            timeout=30
        )
        return json.loads(response.choices[0].message.content)
    except:
        return None

# --- Main Workflow ---
state = st.session_state.state

# Step 1: Start (Upload or Manual)
if state["step"] == "start":
    st.subheader("How would you like to proceed?")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_pdf = st.file_uploader("Upload ESG Report (PDF)", type="pdf")
        if uploaded_pdf:
            text = extract_pdf_text(uploaded_pdf)
            if text:
                state["step"] = "industry"
                st.success("PDF uploaded! Now tell us your industry.")
                st.rerun()
    with col2:
        if st.button("Enter Data Manually"):
            state["step"] = "industry"
            st.rerun()

# Step 2: Enter Industry
elif state["step"] == "industry":
    st.subheader("Your Industry")
    industry = st.text_input("What industry is your company in? (e.g., Manufacturing, Retail)")
    if industry:
        state["industry"] = industry
        state["step"] = "questions"
        st.rerun()

# Step 3: Enter Data (Manual or AI)
elif state["step"] == "questions":
    st.subheader(f"Tell us about {state['company'] or 'your company'}'s efforts")
    
    # If AI extraction is possible
    if OPENAI_AVAILABLE and "pdf_text" in state:
        with st.spinner("Analyzing report..."):
            ai_data = ai_extract(state["pdf_text"], state["industry"])
            if ai_data:
                state["data"] = {
                    "resource_use": {
                        "renewable": ai_data.get("renewable", 0),
                        "water_reuse": ai_data.get("water_reuse", 0),
                        "energy_tech": ai_data.get("energy_tech", 0),
                        "extra": ai_data.get("extra", "")
                    },
                    "materials": {
                        "recycled_pct": ai_data.get("recycled_pct", 0),
                        "waste_reduction": ai_data.get("waste_reduction", 0),
                        "eco_cert": ai_data.get("eco_cert", False),
                        "extra": ""
                    },
                    "circular": {
                        "takeback": ai_data.get("takeback", 0),
                        "packaging": ai_data.get("packaging", 0),
                        "suppliers": ai_data.get("suppliers", 0),
                        "extra": ""
                    }
                }
                state["step"] = "results"
                st.rerun()

    # Manual input form
    with st.form("sustainability_form"):
        st.subheader("Resource Use")
        ren = st.slider("Renewable energy %", 0, 100, 0)
        water = st.slider("Water reuse %", 0, 100, 0)
        tech = st.number_input("Energy-saving technologies (count)", 0, 10, 0)

        st.subheader("Materials & Waste")
        recycled = st.slider("Recycled materials %", 0, 100, 0)
        waste = st.slider("Waste reduction % (vs last year)", 0, 100, 0)
        eco = st.checkbox("Eco-certified products")

        st.subheader("Circular Practices")
        takeback = st.slider("Product take-back %", 0, 100, 0)
        packaging = st.slider("Sustainable packaging %", 0, 100, 0)
        suppliers = st.slider("Sustainable suppliers %", 0, 100, 0)

        if st.form_submit_button("Calculate Score"):
            state["data"] = {
                "resource_use": {"renewable": ren, "water_reuse": water, "energy_tech": tech, "extra": ""},
                "materials": {"recycled_pct": recycled, "waste_reduction": waste, "eco_cert": eco, "extra": ""},
                "circular": {"takeback": takeback, "packaging": packaging, "suppliers": suppliers, "extra": ""}
            }
            state["score"] = calculate_score(state["data"])
            state["step"] = "results"
            st.rerun()

# Step 4: Show Results
elif state["step"] == "results":
    st.subheader(f"Sustainability Score: {state['score']}/100")
    
    st.write("### Breakdown")
    st.write(f"- Resource Use: {round(state['data']['resource_use']['renewable']*0.03 + state['data']['resource_use']['water_reuse']*0.03 + state['data']['resource_use']['energy_tech']*0.15, 1)}/30")
    st.write(f"- Materials: {round(state['data']['materials']['recycled_pct']*0.03 + state['data']['materials']['waste_reduction']*0.03 + (10 if state['data']['materials']['eco_cert'] else 0)*0.3, 1)}/30")
    st.write(f"- Circular Practices: {round(state['data']['circular']['takeback']*0.04 + state['data']['circular']['packaging']*0.04 + state['data']['circular']['suppliers']*0.04, 1)}/40")

    if OPENAI_AVAILABLE:
        try:
            prompt = f"Give 3 sustainability recommendations for a {state['industry']} company with these stats: {state['data']}"
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            st.write("### Recommendations")
            st.write(response.choices[0].message.content)
        except:
            st.write("### Recommendations")
            st.write("1. Increase renewable energy adoption\n2. Reduce waste through recycling\n3. Expand sustainable packaging")
    else:
        st.write("### Recommendations")
        st.write("1. Increase renewable energy adoption\n2. Reduce waste through recycling\n3. Expand sustainable packaging")

    if st.button("Start Over"):
        st.session_state.state = {
            "company": "",
            "industry": "",
            "step": "start",
            "data": {
                "resource_use": {"renewable": 0, "water_reuse": 0, "energy_tech": 0, "extra": ""},
                "materials": {"recycled_pct": 0, "waste_reduction": 0, "eco_cert": False, "extra": ""},
                "circular": {"takeback": 0, "packaging": 0, "suppliers": 0, "extra": ""}
            },
            "score": 0,
            "recommendations": []
        }
        st.rerun()
