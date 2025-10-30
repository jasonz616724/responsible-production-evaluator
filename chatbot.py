import streamlit as st
import json
import fitz  # For PDF extraction
from openai import OpenAI
import re

# --- Page Setup ---
st.set_page_config(page_title="Sustainability Evaluator", layout="centered")
st.title("🌱 Production Sustainability Check")
st.write("Upload an ESG report or enter data manually to assess your efforts.")

# --- OpenAI Setup (GPT-3.5-Turbo) ---
OPENAI_AVAILABLE = False
client = None
try:
    # Use GPT-4 (state-of-the-art model)
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", ""))
    OPENAI_AVAILABLE = True
except Exception as e:
    st.warning(f"AI features disabled (check API key). Error: {str(e)[:50]}")

# --- Session State (Simplified) ---
if "state" not in st.session_state:
    st.session_state.state = {
        "company": "",
        "industry": "",
        "step": "start",  # start → input → results
        "pdf_text": "",
        "data": {
            "resource": {"renewable_pct": 0, "water_reuse_pct": 0, "energy_tech": 0},
            "materials": {"recycled_pct": 0, "waste_reduction_pct": 0, "eco_cert": False},
            "circular": {"takeback_pct": 0, "packaging_pct": 0, "suppliers_pct": 0}
        },
        "score": 0
    }

# --- Core Functions ---
def extract_pdf_text(uploaded_file):
    """Robust PDF text extraction, with better diagnostics"""
    try:
        pdf_bytes = uploaded_file.read()
        if not pdf_bytes or len(pdf_bytes) < 10:
            st.error("Uploaded file is empty or too small to be a PDF.")
            return ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)[:100000]  # Limit for GPT-4
            if not text.strip():
                st.error("No text could be extracted from the PDF. It may be scanned or image-only.")
            return text
    except Exception as e:
        st.error(f"PDF error: {str(e)[:100]}")
        return ""

def ai_extract_esg(text, industry):
    """Extract ESG data with GPT-4 (with estimation)"""
    if not OPENAI_AVAILABLE:
        return None

    prompt = f"""Extract environmental data for a {industry} company from this text.
    If data is missing, ESTIMATE using context. Return ONLY JSON with:
    {{
        "resource": {{
            "renewable_pct": 0-100 (renewable energy %),
            "water_reuse_pct": 0-100 (water reuse %),
            "energy_tech": number (energy-saving technologies count)
        }},
        "materials": {{
            "recycled_pct": 0-100 (recycled materials %),
            "waste_reduction_pct": 0-100 (waste reduction vs last year %),
            "eco_cert": true/false (eco-certified products)
        }},
        "circular": {{
            "takeback_pct": 0-100 (product take-back %),
            "packaging_pct": 0-100 (sustainable packaging %),
            "suppliers_pct": 0-100 (sustainable suppliers %)
        }}
    }}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Upgraded to GPT-4
            messages=[{"role": "user", "content": f"{prompt}\n\nText: {text}"}],
            temperature=0.3,
            timeout=20
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        st.error(f"AI extraction failed: {str(e)[:50]}")
        return None

def calculate_score(data):
    """Simple scoring logic"""
    resource_score = (data["resource"]["renewable_pct"] * 0.1 +
                     data["resource"]["water_reuse_pct"] * 0.1 +
                     min(data["resource"]["energy_tech"], 5) * 2) * 0.3
    
    materials_score = (data["materials"]["recycled_pct"] * 0.1 +
                      data["materials"]["waste_reduction_pct"] * 0.1 +
                      10 if data["materials"]["eco_cert"] else 0) * 0.3
    
    circular_score = (data["circular"]["takeback_pct"] * 0.1 +
                     data["circular"]["packaging_pct"] * 0.1 +
                     data["circular"]["suppliers_pct"] * 0.1) * 0.4
    
    return min(100, round(resource_score + materials_score + circular_score, 1))

# --- Workflow Logic ---
state = st.session_state.state

# Step 1: Start (Upload PDF or Manual Input)
if state["step"] == "start":
    st.subheader("Choose Input Method")
    
    # PDF Upload Option
    st.subheader("Option 1: Upload ESG Report (PDF)")
    uploaded_pdf = st.file_uploader("Select PDF", type="pdf")
    if uploaded_pdf:
        state["pdf_text"] = extract_pdf_text(uploaded_pdf)
        if state["pdf_text"]:
            state["step"] = "input"
            st.success("PDF loaded! Enter industry below.")
            st.rerun()
    
    # Manual Input Option
    st.subheader("Option 2: Enter Data Manually")
    if st.button("Start Manual Input"):
        state["step"] = "input"
        st.rerun()

# Step 2: Input Data (Industry + AI/Manual)
elif state["step"] == "input":
    st.subheader("Company Details")
    state["company"] = st.text_input("Company Name", state["company"])
    state["industry"] = st.text_input("Industry (e.g., Manufacturing)", state["industry"])

    # If PDF was uploaded, try AI extraction with GPT-4
    if state["pdf_text"] and OPENAI_AVAILABLE and state["industry"]:
        if st.button("Analyze PDF with AI (GPT-4)"):
            with st.spinner("Analyzing with GPT-4..."):
                extracted = ai_extract_esg(state["pdf_text"], state["industry"])
                if extracted:
                    state["data"] = extracted
                    state["score"] = calculate_score(state["data"])
                    state["step"] = "results"
                    st.rerun()

    # Manual Input Form
    st.subheader("Enter Data Manually (if no PDF)")
    with st.form("manual_input"):
        st.write("### Resource Use")
        ren_pct = st.slider("Renewable Energy %", 0, 100, 0)
        water_pct = st.slider("Water Reuse %", 0, 100, 0)
        energy_tech = st.number_input("Energy-Saving Technologies (count)", 0, 10, 0)

        st.write("### Materials & Waste")
        recycled_pct = st.slider("Recycled Materials %", 0, 100, 0)
        waste_pct = st.slider("Waste Reduction % (vs last year)", 0, 100, 0)
        eco_cert = st.checkbox("Has Eco-Certified Products")

        st.write("### Circular Practices")
        takeback_pct = st.slider("Product Take-Back %", 0, 100, 0)
        packaging_pct = st.slider("Sustainable Packaging %", 0, 100, 0)
        suppliers_pct = st.slider("Sustainable Suppliers %", 0, 100, 0)

        if st.form_submit_button("Calculate Score"):
            state["data"] = {
                "resource": {"renewable_pct": ren_pct, "water_reuse_pct": water_pct, "energy_tech": energy_tech},
                "materials": {"recycled_pct": recycled_pct, "waste_reduction_pct": waste_pct, "eco_cert": eco_cert},
                "circular": {"takeback_pct": takeback_pct, "packaging_pct": packaging_pct, "suppliers_pct": suppliers_pct}
            }
            state["score"] = calculate_score(state["data"])
            state["step"] = "results"
            st.rerun()

# Step 3: Show Results
elif state["step"] == "results":
    st.subheader(f"{state['company'] or 'Your Company'}: Sustainability Score = {state['score']}/100")

    st.write("### Performance Breakdown")
    st.write(f"- Resource Use: {round((state['data']['resource']['renewable_pct'] * 0.1 + state['data']['resource']['water_reuse_pct'] * 0.1 + min(state['data']['resource']['energy_tech'], 5)*2) * 0.3, 1)}/30")
    st.write(f"- Materials & Waste: {round((state['data']['materials']['recycled_pct'] * 0.1 + state['data']['materials']['waste_reduction_pct'] * 0.1 + (10 if state['data']['materials']['eco_cert'] else 0)) * 0.3, 1)}/30")
    st.write(f"- Circular Practices: {round((state['data']['circular']['takeback_pct'] * 0.1 + state['data']['circular']['packaging_pct'] * 0.1 + state['data']['circular']['suppliers_pct'] * 0.1) * 0.4, 1)}/40")

    # Generate recommendations with GPT-3.5 if available
    if OPENAI_AVAILABLE and state["industry"]:
        try:
            st.write("### Recommendations")
            prompt = f"Give 3 specific sustainability recommendations for a {state['industry']} company with these metrics: {state['data']}. Keep them simple."
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.write("### Recommendations")
            st.write("1. Increase renewable energy adoption\n2. Use more recycled materials\n3. Expand product take-back programs")
    else:
        st.write("### Recommendations")
        st.write("1. Increase renewable energy adoption\n2. Use more recycled materials\n3. Expand product take-back programs")

    if st.button("Start Over"):
        st.session_state.state = {
            "company": "",
            "industry": "",
            "step": "start",
            "pdf_text": "",
            "data": {
                "resource": {"renewable_pct": 0, "water_reuse_pct": 0, "energy_tech": 0},
                "materials": {"recycled_pct": 0, "waste_reduction_pct": 0, "eco_cert": False},
                "circular": {"takeback_pct": 0, "packaging_pct": 0, "suppliers_pct": 0}
            },
            "score": 0
        }
        st.rerun()
