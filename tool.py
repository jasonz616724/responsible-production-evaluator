import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pdfkit
import tempfile
import os
import fitz
import json
from openai import OpenAI
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="SDG 12 Production Responsibility Evaluator", layout="wide")
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- Initialize OpenAI Client ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except KeyError:
    st.warning("⚠️ OPENAI_API_KEY not found in Streamlit Secrets. AI features disabled.")
    OPENAI_AVAILABLE = False
except Exception as e:
    st.error(f"⚠️ OpenAI client error: {str(e)}")
    OPENAI_AVAILABLE = False

# --- Session State Initialization ---
if "production_data" not in st.session_state:
    st.session_state["production_data"] = {
        "Company Name": "EcoManufacture Inc.",
        "Industry": "Manufacturing",  # Options: Manufacturing, Tourism, Hospitality, Infrastructure, Other
        "extracted_pdf_text": "",
        # Core Dimensions Data
        "resource_efficiency": {"renewable_energy_pct": 30, "energy_tech_count": 2, "water_reuse_pct": 40, "ai_score": 0},
        "sustainable_production": {"recycled_material_pct": 20, "waste_intensity_pct": 30, "eco_design_cert": False, "ai_score": 0},
        "chemical_waste": {"hazardous_reduction_pct": 20, "waste_recycling_pct": 50, "chemical_compliance": True, "ai_score": 0},
        "circular_economy": {"takeback_program_pct": 20, "packaging_sustainable_pct": 50, "certified_supplier_pct": 30, "ai_score": 0},
        "sustainable_procurement": {"procurement_criteria_count": 2, "sustainable_budget_pct": 10, "procurement_tracking": False, "ai_score": 0},
        "life_cycle_thinking": {"lca_product_pct": 20, "consumer_comm": False, "lca_improvements": 1, "ai_score": 0},
        "waste_management": {"food_waste_reduction_pct": 20, "segregation_rate_pct": 60, "circular_partnerships": False, "ai_score": 0},
        "tourism_infrastructure": {"sustainable_material_pct": 30, "eco_tourism_pct": 0, "energy_water_efficiency": False, "ai_score": 0},
        # Final Outputs
        "total_score": 0,
        "dimension_scores": {},
        "ai_recommendations": [],
        "mock_esg_excerpt": ""
    }

if "rerun_trigger" not in st.session_state:
    st.session_state["rerun_trigger"] = False

# --- Constants (Scoring Framework) ---
# Dimension Definitions (aligned with SDG 12, adjusted for feasibility)
DIMENSIONS = [
    {
        "id": "resource_efficiency",
        "name": "Resource Efficiency (SDG 12.2)",
        "weight": 0.25,
        "actions": [
            {"name": "renewable_energy_pct", "desc": "Renewable energy percentage", "calc": lambda x: 10 if x >=50 else 5 if x >=30 else 0},
            {"name": "energy_tech_count", "desc": "Number of energy-efficient tech categories", "calc": lambda x: min(10, x * 5)},
            {"name": "water_reuse_pct", "desc": "Water reuse rate percentage", "calc": lambda x: 10 if x >=70 else 5 if x >=40 else 0}
        ],
        "ai_criteria": "Evaluate 1) energy intensity trends vs. industry peers, 2) water consumption efficiency (e.g., leak prevention). Return a score 0-5 (no extra text).",
        "max_subtotal": 30
    },
    {
        "id": "sustainable_production",
        "name": "Sustainable Production (SDG 12.3)",
        "weight": 0.20,
        "actions": [
            {"name": "recycled_material_pct", "desc": "Recycled/upcycled material percentage", "calc": lambda x: 10 if x >=40 else 5 if x >=20 else 0},
            {"name": "waste_intensity_pct", "desc": "Waste intensity vs. industry average (%)", "calc": lambda x: 10 if x <=20 else 5 if x <=40 else 0},
            {"name": "eco_design_cert", "desc": "Eco-design certification (Yes/No)", "calc": lambda x: 10 if x else 0}
        ],
        "ai_criteria": "Evaluate 1) material efficiency (yield improvement), 2) product end-of-life recyclability potential. Return a score 0-5 (no extra text).",
        "max_subtotal": 30
    },
    {
        "id": "chemical_waste",
        "name": "Chemicals & Waste Management (SDG 12.4)",
        "weight": 0.18,
        "actions": [
            {"name": "hazardous_reduction_pct", "desc": "Hazardous chemical reduction vs. baseline (%)", "calc": lambda x: 7 if x >=50 else 3 if x >=20 else 0},
            {"name": "waste_recycling_pct", "desc": "Production waste recycling rate (%)", "calc": lambda x: 7 if x >=80 else 3 if x >=50 else 0},
            {"name": "chemical_compliance", "desc": "Compliance with REACH/Stockholm Convention", "calc": lambda x: 6 if x else 0}
        ],
        "ai_criteria": "Evaluate 1) chemical spill prevention protocols, 2) hazardous waste treatment effectiveness. Return a score 0-3 (no extra text).",
        "max_subtotal": 20
    },
    {
        "id": "circular_economy",
        "name": "Circular Economy Integration (SDG 12.5)",
        "weight": 0.10,
        "actions": [
            {"name": "takeback_program_pct", "desc": "Product lines covered by take-back programs (%)", "calc": lambda x: 4 if x >=50 else 2 if x >=20 else 0},
            {"name": "packaging_sustainable_pct", "desc": "Renewable/compostable packaging percentage", "calc": lambda x: 3 if x >=80 else 1 if x >=50 else 0},
            {"name": "certified_supplier_pct", "desc": "Suppliers with sustainability certifications (%)", "calc": lambda x: 3 if x >=60 else 1 if x >=30 else 0}
        ],
        "ai_criteria": "Evaluate 1) tier 2 supplier sustainability practices, 2) product-as-a-service model adoption. Return a score 0-2 (no extra text).",
        "max_subtotal": 10
    },
    {
        "id": "sustainable_procurement",
        "name": "Sustainable Procurement (SDG 12.7)",
        "weight": 0.08,
        "actions": [
            {"name": "procurement_criteria_count", "desc": "Sustainability criteria in procurement policy", "calc": lambda x: 3 if x >=3 else 1 if x >=1 else 0},
            {"name": "sustainable_budget_pct", "desc": "Procurement budget for sustainable goods (%)", "calc": lambda x: 3 if x >=30 else 1 if x >=10 else 0},
            {"name": "procurement_tracking", "desc": "Tracking of sustainable procurement performance", "calc": lambda x: 2 if x else 0}
        ],
        "ai_criteria": "Evaluate 1) supplier diversity in sustainable procurement, 2) alignment with ISO 20400. Return a score 0-2 (no extra text).",
        "max_subtotal": 8
    },
    {
        "id": "life_cycle_thinking",
        "name": "Life-Cycle Thinking (SDG 12.8)",
        "weight": 0.05,
        "actions": [
            {"name": "lca_product_pct", "desc": "Product lines with Life-Cycle Assessment (%)", "calc": lambda x: 2 if x >=50 else 1 if x >=20 else 0},
            {"name": "consumer_comm", "desc": "Sustainability info communicated to consumers", "calc": lambda x: 2 if x else 0},
            {"name": "lca_improvements", "desc": "Product improvements from LCA insights", "calc": lambda x: 1 if x >=2 else 0.5 if x >=1 else 0}
        ],
        "ai_criteria": "Evaluate 1) inclusion of scope 3 emissions in LCA, 2) consumer engagement with eco-labels. Return a score 0-1 (no extra text).",
        "max_subtotal": 5
    },
    {
        "id": "waste_management",
        "name": "Waste Generation & Management (SDG 12.5.1)",
        "weight": 0.07,
        "actions": [
            {"name": "food_waste_reduction_pct", "desc": "Food/by-product waste reduction vs. baseline (%)", "calc": lambda x: 3 if x >=40 else 1 if x >=20 else 0},
            {"name": "segregation_rate_pct", "desc": "Waste segregation rate (%)", "calc": lambda x: 3 if x >=90 else 1 if x >=60 else 0},
            {"name": "circular_partnerships", "desc": "Partnerships with circular waste startups", "calc": lambda x: 1 if x else 0}
        ],
        "ai_criteria": "Evaluate 1) purity of recycled waste streams, 2) revenue from by-product utilization. Return a score 0-2 (no extra text).",
        "max_subtotal": 7
    },
    {
        "id": "tourism_infrastructure",
        "name": "Sustainable Tourism & Infrastructure (SDG 12.9)",
        "weight": 0.07,
        "actions": [
            {"name": "sustainable_material_pct", "desc": "Sustainable building materials percentage", "calc": lambda x: 3 if x >=50 else 1 if x >=20 else 0},
            {"name": "eco_tourism_pct", "desc": "Eco-tourism practices coverage (%)", "calc": lambda x: 3 if x >=60 else 1 if x >=30 else 0},
            {"name": "energy_water_efficiency", "desc": "Energy/water use ≤30% of industry average", "calc": lambda x: 1 if x else 0}
        ],
        "ai_criteria": "Evaluate 1) community impact of projects, 2) durability of infrastructure materials. Return a score 0-2 (no extra text).",
        "max_subtotal": 7,
        "industries": ["Tourism", "Hospitality", "Infrastructure"]  # Only show for these industries
    }
]

# --- Core AI Functions ---
def get_ai_response(prompt, system_msg="You are an ESG expert. Be concise."):
    if not OPENAI_AVAILABLE:
        return "❌ AI requires OPENAI_API_KEY in secrets."
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            temperature=0.4,
            timeout=20
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"⚠️ AI error: {str(e)}")
        return None

def ai_extract_esg_data(pdf_text):
    """Extract production-related data from ESG report text."""
    if not pdf_text:
        return {}
    
    prompt = f"""Extract the following data from this ESG report excerpt (return ONLY valid JSON, no extra text):
    {{
        "company_name": "Company name (string)",
        "industry": "Industry (Manufacturing/Tourism/Hospitality/Infrastructure/Other)",
        "resource_efficiency": {{
            "renewable_energy_pct": "Renewable energy percentage (number, 0-100)",
            "energy_tech_count": "Number of energy-efficient tech categories (number, 0+)",
            "water_reuse_pct": "Water reuse rate (number, 0-100)"
        }},
        "sustainable_production": {{
            "recycled_material_pct": "Recycled raw material percentage (number, 0-100)",
            "waste_intensity_pct": "Waste intensity vs industry average (number, 0+)",
            "eco_design_cert": "Eco-design certification (true/false)"
        }},
        "chemical_waste": {{
            "hazardous_reduction_pct": "Hazardous chemical reduction vs baseline (number, 0-100)",
            "waste_recycling_pct": "Production waste recycling rate (number, 0-100)",
            "chemical_compliance": "Compliance with REACH/Stockholm Convention (true/false)"
        }},
        "circular_economy": {{
            "takeback_program_pct": "Product lines with take-back programs (number, 0-100)",
            "packaging_sustainable_pct": "Sustainable packaging percentage (number, 0-100)",
            "certified_supplier_pct": "Certified sustainable suppliers (number, 0-100)"
        }},
        "sustainable_procurement": {{
            "procurement_criteria_count": "Sustainability criteria in procurement (number, 0+)",
            "sustainable_budget_pct": "Procurement budget for sustainable goods (number, 0-100)",
            "procurement_tracking": "Tracking sustainable procurement (true/false)"
        }},
        "life_cycle_thinking": {{
            "lca_product_pct": "Products with Life-Cycle Assessment (number, 0-100)",
            "consumer_comm": "Sustainability info for consumers (true/false)",
            "lca_improvements": "Product improvements from LCA (number, 0+)"
        }},
        "waste_management": {{
            "food_waste_reduction_pct": "Food/by-product waste reduction (number, 0-100)",
            "segregation_rate_pct": "Waste segregation rate (number, 0-100)",
            "circular_partnerships": "Partnerships with circular startups (true/false)"
        }},
        "tourism_infrastructure": {{
            "sustainable_material_pct": "Sustainable building materials (number, 0-100)",
            "eco_tourism_pct": "Eco-tourism practice coverage (number, 0-100)",
            "energy_water_efficiency": "Energy/water use ≤30% industry average (true/false)"
        }}
    }}
    Use null for missing data. Do NOT add explanations.
    PDF Text: {pdf_text[:3000]}"""
    
    response = get_ai_response(prompt, system_msg="You are a data extractor. Return ONLY valid JSON (no extra text).")
    if not response:
        return {}
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        st.warning(f"⚠️ AI PDF parse failed (invalid JSON). Raw response: {response[:100]}...")
        return {}

def ai_evaluate_unlisted_criteria(dimension):
    """Evaluate AI-only criteria for a dimension."""
    data = st.session_state["production_data"][dimension["id"]]
    prompt = f"""Evaluate the following for {dimension['name']}:
    - Company Data: {data}
    - Evaluation Criteria: {dimension['ai_criteria']}
    Return ONLY the numeric score (no text, no units)."""
    
    response = get_ai_response(prompt, system_msg="You are an ESG analyst. Return ONLY a number.")
    if not response:
        return 0
    
    try:
        return max(0, min(dimension.get("max_ai_score", 5), float(response.strip())))
    except (ValueError, TypeError):
        st.warning(f"⚠️ Invalid AI score for {dimension['name']}. Defaulting to 0.")
        return 0

def ai_generate_recommendations():
    """Generate tailored improvement recommendations based on scores."""
    data = st.session_state["production_data"]
    scores = data["dimension_scores"]  # Format: {dim_id: {"weighted_score": X, ...}, ...}
    company_industry = data["Industry"]
    
    # Fix 1: Only include relevant dimensions (skip tourism for non-relevant industries)
    # Fix 2: Compare the "weighted_score" value (not the full score object) to the threshold
    low_dimensions = []
    for dim in DIMENSIONS:
        # Skip tourism dimension if industry is not in its target list
        if dim["id"] == "tourism_infrastructure" and company_industry not in dim["industries"]:
            continue
        
        # Get the weighted score for the dimension (default to 0 if missing)
        dim_score = scores.get(dim["id"], {}).get("weighted_score", 0)
        # Calculate 50% of the maximum possible weighted score for the dimension
        max_weighted_threshold = (dim["max_subtotal"] * dim["weight"]) * 0.5
        
        # Add to low dimensions if score is below threshold
        if dim_score < max_weighted_threshold:
            low_dimensions.append(dim["name"])
    
    # If no low dimensions (unlikely), use a fallback list
    if not low_dimensions:
        low_dimensions = ["Resource Efficiency (SDG 12.2)", "Circular Economy Integration (SDG 12.5)"]
    
    prompt = f"""Generate 3 specific, actionable sustainability recommendations for {data['Company Name']} (Industry: {data['Industry']}).
    Focus on their low-performing areas: {low_dimensions}.
    Each recommendation must:
    1. Reference a specific SDG 12 target (e.g., SDG 12.2 for resource efficiency).
    2. Include a measurable goal (e.g., "Increase renewable energy to 50% by 2026").
    3. Explain why it matters (e.g., "Reduces carbon footprint by 30%").
    Do NOT use bullet points—number each recommendation (1., 2., 3.)."""
    
    response = get_ai_response(prompt, system_msg="You are a sustainability consultant. Be specific and actionable.")
    # Return cleaned recommendations (handle empty/error responses)
    if not response:
        return [
            "1. Increase renewable energy adoption to 50% by 2026 (SDG 12.2) – Reduces reliance on fossil fuels and cuts carbon emissions by 30%.",
            "2. Expand product take-back programs to cover 50% of product lines (SDG 12.5) – Enhances circularity and reduces end-of-life waste by 25%.",
            "3. Implement eco-design certifications for all core products (SDG 12.3) – Improves product recyclability and aligns with global sustainability standards."
        ]
    # Split response into numbered items and clean whitespace
    return [line.strip() for line in response.split("\n") if line.strip() and line.strip()[0].isdigit()]
def ai_generate_mock_esg():
    """Generate a 300-500 word mock ESG excerpt for the company."""
    data = st.session_state["production_data"]
    scores = data["dimension_scores"]
    total_score = data["total_score"]
    
    prompt = f"""Write a 300-500 word mock ESG report excerpt for {data['Company Name']} (Industry: {data['Industry']}).
    Include:
    1. Introduction to their production sustainability efforts.
    2. Key metrics: renewable energy use ({data['resource_efficiency']['renewable_energy_pct']}%), recycled materials ({data['sustainable_production']['recycled_material_pct']}%), waste recycling ({data['chemical_waste']['waste_recycling_pct']}%).
    3. Strengths (high-scoring dimensions: {[dim['name'] for dim in DIMENSIONS if scores.get(dim['id'], 0) > (dim['max_subtotal'] * dim['weight']) * 0.7]}).
    4. Areas for improvement (low-scoring dimensions: {[dim['name'] for dim in DIMENSIONS if scores.get(dim['id'], 0) < (dim['max_subtotal'] * dim['weight']) * 0.5]}).
    5. Future goals aligned with SDG 12.
    Use a formal, professional tone (like real ESG reports). Do NOT use bullet points."""
    
    return get_ai_response(prompt, system_msg="You are an ESG report writer. Write in a formal, concise style.") or """EcoManufacture Inc. (Manufacturing) is committed to advancing SDG 12 (Responsible Consumption and Production) through its production operations. In 2024, the company sourced 30% of its energy from renewable sources (solar and wind), utilized 20% recycled raw materials in production, and achieved a 50% production waste recycling rate—reflecting progress toward circularity.

Strengths include waste management (segregation rate of 60%) and sustainable procurement (2 sustainability criteria in procurement policies), which have reduced operational environmental impact by 15% year-over-year. The company also complies with international chemical management standards (REACH), minimizing hazardous material risks.

Key areas for improvement include resource efficiency (water reuse rate of 40% below the 70% target) and circular economy integration (only 20% of product lines have take-back programs). Life-cycle thinking efforts are also nascent, with LCAs conducted for just 20% of product lines.

Looking ahead, EcoManufacture aims to increase renewable energy to 50% by 2026 (SDG 12.2), expand take-back programs to 50% of products (SDG 12.5), and achieve 70% water reuse (SDG 12.2)—aligning with its long-term vision of carbon-neutral production by 2030."""

# --- Helper Functions ---
def extract_pdf_text(uploaded_file):
    """Extract text from uploaded PDF."""
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf_doc:
            return "\n\n".join([page.get_text().strip() for page in pdf_doc])
    except Exception as e:
        st.error(f"⚠️ PDF extraction failed: {str(e)}")
        return ""

def calculate_dimension_scores():
    """Calculate scores for each dimension (including AI unlisted criteria)."""
    data = st.session_state["production_data"]
    dimension_scores = {}
    total_score = 0
    
    for dim in DIMENSIONS:
        # Skip tourism dimension if industry is irrelevant
        if dim["id"] == "tourism_infrastructure" and data["Industry"] not in dim["industries"]:
            continue
        
        # Calculate action-based subtotal
        dim_data = data[dim["id"]]
        action_subtotal = sum([action["calc"](dim_data[action["name"]]) for action in dim["actions"]])
        
        # Add AI score (ensure it doesn't exceed max subtotal)
        ai_score = ai_evaluate_unlisted_criteria(dim) if OPENAI_AVAILABLE else 0
        dim_data["ai_score"] = ai_score
        total_subtotal = min(dim["max_subtotal"], action_subtotal + ai_score)
        
        # Calculate weighted score (contribution to total 100)
        weighted_score = round(total_subtotal * dim["weight"], 1)
        dimension_scores[dim["id"]] = {
            "name": dim["name"],
            "subtotal": total_subtotal,
            "weighted_score": weighted_score,
            "max_weighted": round(dim["max_subtotal"] * dim["weight"], 1)
        }
        total_score += weighted_score
    
    # Update session state
    data["dimension_scores"] = dimension_scores
    data["total_score"] = round(total_score, 1)
    st.session_state["production_data"] = data

def generate_report_content():
    """Generate text content for TXT/PDF reports."""
    data = st.session_state["production_data"]
    scores = data["dimension_scores"]
    
    # Header
    content = f"SDG 12 Production Responsibility Report\n"
    content += f"Company: {data['Company Name']}\n"
    content += f"Industry: {data['Industry']}\n"
    content += f"Total Score: {data['total_score']}/100\n"
    content += "="*50 + "\n\n"
    
    # Dimension Breakdown
    content += "1. Dimension Score Breakdown\n"
    for dim_id, dim_data in scores.items():
        content += f"- {dim_data['name']}: {dim_data['weighted_score']}/{dim_data['max_weighted']} (Subtotal: {dim_data['subtotal']}/{DIMENSIONS[[d['id'] for d in DIMENSIONS].index(dim_id)]['max_subtotal']})\n"
    
    # Key Metrics
    content += "\n2. Key Production Metrics\n"
    core_dims = ["resource_efficiency", "sustainable_production", "chemical_waste", "circular_economy"]
    for dim_id in core_dims:
        if dim_id not in data:
            continue
        dim_data = data[dim_id]
        dim_name = next(d["name"] for d in DIMENSIONS if d["id"] == dim_id)
        content += f"- {dim_name}:\n"
        for action in [a for a in DIMENSIONS[[d["id"] for d in DIMENSIONS].index(dim_id)]["actions"]]:
            content += f"  - {action['desc']}: {dim_data[action['name']]}\n"
    
    # Recommendations
    content += "\n3. Improvement Recommendations\n"
    recommendations = ai_generate_recommendations() if OPENAI_AVAILABLE else data["ai_recommendations"]
    data["ai_recommendations"] = recommendations
    for i, rec in enumerate(recommendations, 1):
        content += f"{i}. {rec}\n"
    
    # Mock ESG Excerpt
    content += "\n4. Mock ESG Report Excerpt\n"
    mock_excerpt = ai_generate_mock_esg() if OPENAI_AVAILABLE else data["mock_esg_excerpt"]
    data["mock_esg_excerpt"] = mock_excerpt
    content += mock_excerpt + "\n"
    
    return content, recommendations, mock_excerpt

# --- Sidebar UI (Data Input) ---
st.sidebar.header("📊 Data Input")

# 1. PDF Upload
st.sidebar.subheader("1. Upload ESG Report (Optional)")
uploaded_pdf = st.sidebar.file_uploader("Upload PDF for AI Data Extraction", type="pdf")
if uploaded_pdf:
    with st.spinner("🔍 Extracting text from PDF..."):
        pdf_text = extract_pdf_text(uploaded_pdf)
        st.session_state["production_data"]["extracted_pdf_text"] = pdf_text
        
        with st.sidebar.expander("View Extracted Text", expanded=False):
            st.text_area("PDF Content", pdf_text, height=200, disabled=True)
        
        if OPENAI_AVAILABLE:
            with st.spinner("🤖 Analyzing ESG data..."):
                esg_data = ai_extract_esg_data(pdf_text)
                if esg_data:
                    st.sidebar.success("✅ AI populated data! Review and edit below.")
                    # Update session state with extracted data
                    for key, value in esg_data.items():
                        if key in st.session_state["production_data"] and value is not None:
                            st.session_state["production_data"][key] = value

# 2. Basic Company Info
st.sidebar.subheader("2. Company Information")
company_name = st.sidebar.text_input(
    "Company Name",
    st.session_state["production_data"]["Company Name"],
    key="company_name"
)
industry = st.sidebar.selectbox(
    "Industry",
    ["Manufacturing", "Tourism", "Hospitality", "Infrastructure", "Other"],
    index=["Manufacturing", "Tourism", "Hospitality", "Infrastructure", "Other"].index(
        st.session_state["production_data"]["Industry"]
    ),
    key="industry"
)

# 3. Dimension-Specific Inputs
st.sidebar.subheader("3. Production Sustainability Data")
data = st.session_state["production_data"]

# Render inputs for each dimension
for dim in DIMENSIONS:
    # Skip tourism dimension if industry is irrelevant
    if dim["id"] == "tourism_infrastructure" and industry not in dim["industries"]:
        continue
    
    st.sidebar.markdown(f"**{dim['name']}**")
    dim_data = data[dim["id"]]
    
    # Render input for each action in the dimension
    for action in dim["actions"]:
        if "pct" in action["name"]:  # Percentage inputs
            value = st.sidebar.slider(
                f"{action['desc']} (%)",
                min_value=0,
                max_value=100,
                value=dim_data[action["name"]] if dim_data[action["name"]] is not None else 0,
                key=f"{dim['id']}_{action['name']}"
            )
        elif "count" in action["name"] or "improvements" in action["name"]:  # Numeric counts
            value = st.sidebar.number_input(
                f"{action['desc']}",
                min_value=0,
                value=dim_data[action["name"]] if dim_data[action["name"]] is not None else 0,
                key=f"{dim['id']}_{action['name']}"
            )
        else:  # Boolean (Yes/No)
            value = st.sidebar.checkbox(
                f"{action['desc']}",
                value=dim_data[action["name"]] if dim_data[action["name"]] is not None else False,
                key=f"{dim['id']}_{action['name']}"
            )
        dim_data[action["name"]] = value
    
    # Update session state with edited dimension data
    data[dim["id"]] = dim_data
    st.sidebar.markdown("---")

# 4. Save Data
if st.sidebar.button("💾 Save Data", use_container_width=True):
    # Update session state with all inputs
    data["Company Name"] = company_name
    data["Industry"] = industry
    st.session_state["production_data"] = data
    
    # Calculate scores
    calculate_dimension_scores()
    st.sidebar.success("✅ Data saved and scores calculated!")
    st.session_state["rerun_trigger"] = True

# --- Rerun Trigger ---
if st.session_state["rerun_trigger"]:
    st.session_state["rerun_trigger"] = False
    st.rerun()

# --- Main Dashboard (Results & Visualizations) ---
data = st.session_state["production_data"]
scores = data.get("dimension_scores", {})
total_score = data.get("total_score", 0)

st.title("🌱 SDG 12 Production Responsibility Evaluator")

# 1. Overview Card
st.subheader("📋 Evaluation Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Company Name", data["Company Name"])
with col2:
    st.metric("Industry", data["Industry"])
with col3:
    st.metric("Total SDG 12 Score", f"{total_score}/100")

# 2. Score Breakdown Chart
st.subheader("📊 Dimension Score Breakdown")
if scores:
    # Prepare data for chart
    dim_names = [v["name"].split(" (")[0] for v in scores.values()]
    dim_scores = [v["weighted_score"] for v in scores.values()]
    dim_max = [v["max_weighted"] for v in scores.values()]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(dim_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, dim_scores, width, label="Achieved Score", color="#2E8B57")
    bars2 = ax.bar(x + width/2, dim_max, width, label="Max Possible Score", color="#D3D3D3", alpha=0.7)
    
    ax.set_xlabel("Sustainability Dimensions")
    ax.set_ylabel("Score")
    ax.set_title("SDG 12 Dimension Score Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names, rotation=45, ha="right")
    ax.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{height}", ha="center", va="bottom")
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{height}", ha="center", va="bottom")
    
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("ℹ️ Enter and save data in the sidebar to generate score breakdown.")

# 3. Detailed Score Table
st.subheader("📋 Detailed Score Table")
if scores:
    table_data = []
    for dim_id, dim_data in scores.items():
        table_data.append({
            "Dimension": dim_data["name"],
            "Achieved Subtotal": f"{dim_data['subtotal']}/{DIMENSIONS[[d['id'] for d in DIMENSIONS].index(dim_id)]['max_subtotal']}",
            "Weighted Score": f"{dim_data['weighted_score']}/{dim_data['max_weighted']}"
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
else:
    st.info("ℹ️ No score data available. Save data to generate table.")

# 4. AI Recommendations
st.subheader("💡 Improvement Recommendations")
if OPENAI_AVAILABLE and scores:
    if st.button("Generate AI Recommendations", use_container_width=True):
        with st.spinner("🤖 Generating recommendations..."):
            recommendations = ai_generate_recommendations()
            data["ai_recommendations"] = recommendations
            st.session_state["production_data"] = data
    
    if data["ai_recommendations"]:
        for i, rec in enumerate(data["ai_recommendations"], 1):
            st.write(f"{i}. {rec}")
else:
    st.info("ℹ️ Save data and enable AI (with OPENAI_API_KEY) to generate recommendations.")

# 5. Mock ESG Excerpt
st.subheader("📄 Mock ESG Report Excerpt")
if OPENAI_AVAILABLE:
    if st.button("Generate Mock ESG Excerpt", use_container_width=True):
        with st.spinner("🤖 Writing mock ESG excerpt..."):
            mock_excerpt = ai_generate_mock_esg()
            data["mock_esg_excerpt"] = mock_excerpt
            st.session_state["production_data"] = data
    
    if data["mock_esg_excerpt"]:
        st.write(data["mock_esg_excerpt"])
else:
    st.info("ℹ️ Enable AI (with OPENAI_API_KEY) to generate a mock ESG excerpt.")

# 6. Report Export
st.subheader("📥 Export Report")
if scores:
    report_content, _, _ = generate_report_content()
    
    # TXT Export
    st.download_button(
        label="Download TXT Report",
        data=report_content,
        file_name=f"{data['Company Name']}_SDG12_Report.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    # PDF Export (requires wkhtmltopdf)
    if st.button("Generate PDF Report", use_container_width=True):
        if not OPENAI_AVAILABLE:
            st.warning("⚠️ AI is required to generate PDF reports.")
            st.stop()
        
        try:
            # Test wkhtmltopdf configuration
            pdfkit.configuration(wkhtmltopdf=pdfkit.from_url('http://google.com', False))
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                # Format HTML for PDF
                html_content = f"""
                <html>
                <head>
                    <title>{data['Company Name']} SDG 12 Report</title>
                    <style>
                        body {{ font-family: Arial; margin: 20px; }}
                        h1 {{ color: #2E8B57; }}
                        .header {{ margin-bottom: 30px; }}
                        .section {{ margin: 20px 0; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>SDG 12 Production Responsibility Report</h1>
                        <p><strong>Company:</strong> {data['Company Name']}</p>
                        <p><strong>Industry:</strong> {data['Industry']}</p>
                        <p><strong>Total Score:</strong> {data['total_score']}/100</p>
                    </div>
                    <div class="section">
                        <h2>1. Dimension Score Breakdown</h2>
                        <table>
                            <tr><th>Dimension</th><th>Achieved Score</th><th>Max Score</th></tr>
                            {''.join([f"<tr><td>{v['name']}</td><td>{v['weighted_score']}</td><td>{v['max_weighted']}</td></tr>" for v in scores.values()])}
                        </table>
                    </div>
                    <div class="section">
                        <h2>2. Improvement Recommendations</h2>
                        <ul>{''.join([f"<li>{rec}</li>" for rec in data['ai_recommendations']])}</ul>
                    </div>
                    <div class="section">
                        <h2>3. Mock ESG Excerpt</h2>
                        <p>{data['mock_esg_excerpt'].replace('\n', '<br>')}</p>
                    </div>
                </body>
                </html>
                """
                pdfkit.from_string(html_content, tmp.name)
                
                # Provide download
                with open(tmp.name, "rb") as f:
                    st.download_button(
                        label="Download PDF Report",
                        data=f,
                        file_name=f"{data['Company Name']}_SDG12_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            os.unlink(tmp.name)
        
        except Exception as e:
            st.error(f"⚠️ PDF generation failed: {str(e)}. Ensure wkhtmltopdf is installed (https://wkhtmltopdf.org/).")
else:
    st.info("ℹ️ Save data to generate and export reports.")
