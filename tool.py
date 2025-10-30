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
        return "❌ AI features require an OPENAI_API_KEY in Streamlit Secrets."
    
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
    """Extract production-related data from ESG report text with improved terminology handling."""
    if not pdf_text:
        return {}
    
    # Explicit terminology map to help AI recognize synonyms for key metrics
    terminology_map = """
    Key Metrics & Synonyms to Recognize:
    - Renewable energy percentage: "solar/wind energy占比", "可再生能源使用比例", "green energy share", "清洁能源占比"
    - Energy-efficient tech categories: "节能技术数量", "energy-saving technologies", "高效能设备种类", "节能设备数量"
    - Water reuse rate: "水资源循环利用率", "water recycling percentage", "中水回用率", "水循环利用比例"
    - Recycled material percentage: "再生材料占比", "recycled content", "回收原料使用率", "循环材料比例"
    - Waste intensity vs industry average: "单位产值废弃物强度", "waste per unit output vs peers", "废物强度行业对比", "单位产品废弃物排放量"
    - Eco-design certification: "生态设计认证", "environmental design certification", "绿色产品认证", "环保设计认证"
    - Hazardous chemical reduction: "危险化学品削减率", "hazardous substances reduction", "有害化学物质减排", "危险废物减少比例"
    - Waste recycling rate: "废弃物回收率", "production waste recycled", "废料再利用率", "垃圾回收比例"
    - Chemical compliance: "化学品合规性", "REACH/Stockholm Convention compliance", "化学品管理达标", "符合化学品公约要求"
    """
    
    prompt = f"""Extract the following data from this ESG report excerpt (return ONLY valid JSON, no extra text).
    Use the terminology map below to recognize synonyms for metrics:
    {terminology_map}
    
    {{
        "company_name": "Company name (string)",
        "industry": "Industry (Manufacturing/Tourism/Hospitality/Infrastructure/Other)",
        "resource_efficiency": {{
            "renewable_energy_pct": "Renewable energy percentage (number, 0-100; return null if not found)",
            "energy_tech_count": "Number of energy-efficient tech categories (number, 0+; return null if not found)",
            "water_reuse_pct": "Water reuse rate (number, 0-100; return null if not found)"
        }},
        "sustainable_production": {{
            "recycled_material_pct": "Recycled raw material percentage (number, 0-100; return null if not found)",
            "waste_intensity_pct": "Waste intensity vs industry average (number, 0+; return null if not found)",
            "eco_design_cert": "Eco-design certification (true/false/null if not found)"
        }},
        "chemical_waste": {{
            "hazardous_reduction_pct": "Hazardous chemical reduction vs baseline (number, 0-100; return null if not found)",
            "waste_recycling_pct": "Production waste recycling rate (number, 0-100; return null if not found)",
            "chemical_compliance": "Compliance with REACH/Stockholm Convention (true/false/null if not found)"
        }},
        "circular_economy": {{
            "takeback_program_pct": "Product lines with take-back programs (number, 0-100; return null if not found)",
            "packaging_sustainable_pct": "Sustainable packaging percentage (number, 0-100; return null if not found)",
            "certified_supplier_pct": "Certified sustainable suppliers (number, 0-100; return null if not found)"
        }},
        "sustainable_procurement": {{
            "procurement_criteria_count": "Sustainability criteria in procurement (number, 0+; return null if not found)",
            "sustainable_budget_pct": "Procurement budget for sustainable goods (number, 0-100; return null if not found)",
            "procurement_tracking": "Tracking sustainable procurement (true/false/null if not found)"
        }},
        "life_cycle_thinking": {{
            "lca_product_pct": "Products with Life-Cycle Assessment (number, 0-100; return null if not found)",
            "consumer_comm": "Sustainability info for consumers (true/false/null if not found)",
            "lca_improvements": "Product improvements from LCA (number, 0+; return null if not found)"
        }},
        "waste_management": {{
            "food_waste_reduction_pct": "Food/by-product waste reduction (number, 0-100; return null if not found)",
            "segregation_rate_pct": "Waste segregation rate (number, 0-100; return null if not found)",
            "circular_partnerships": "Partnerships with circular startups (true/false/null if not found)"
        }},
        "tourism_infrastructure": {{
            "sustainable_material_pct": "Sustainable building materials (number, 0-100; return null if not found)",
            "eco_tourism_pct": "Eco-tourism practice coverage (number, 0-100; return null if not found)",
            "energy_water_efficiency": "Energy/water use ≤30% industry average (true/false/null if not found)"
        }}
    }}
    
    PDF Text (expanded excerpt): {pdf_text[:5000]}  # Increased context for better extraction
    """
    
    response = get_ai_response(prompt, system_msg="You are a precise data extractor. Map report terminology to the requested metrics using the provided synonyms. Return ONLY valid JSON.")
    if not response:
        return {}
    
    try:
        extracted_data = json.loads(response)
        
        # Validate and flag missing fields for user feedback
        missing_fields = []
        for dim_name, dim_values in extracted_data.items():
            if isinstance(dim_values, dict):  # Check nested metrics
                for metric, value in dim_values.items():
                    if value is None:
                        missing_fields.append(f"{dim_name} → {metric}")
        
        # Show user which metrics were found/missing
        if missing_fields:
            with st.sidebar.expander("⚠️ Partial Data Extraction", expanded=True):
                st.info(f"Found {len(extracted_data.keys()) - len(missing_fields)} metrics. Missing:\n" + "\n".join(missing_fields[:5]) + ("..." if len(missing_fields) > 5 else ""))
        else:
            st.sidebar.success("✅ All metrics extracted successfully!")
        
        return extracted_data
    
    except json.JSONDecodeError:
        st.sidebar.error(f"⚠️ Failed to parse PDF data. Please enter data manually.")
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
        return max(0, min(5, float(response.strip())))  # Cap at 5 to prevent overscoring
    except (ValueError, TypeError):
        st.warning(f"⚠️ Invalid AI score for {dimension['name']}. Defaulting to 0.")
        return 0

def ai_generate_recommendations():
    """Generate tailored improvement recommendations based on scores."""
    data = st.session_state["production_data"]
    scores = data["dimension_scores"]  # Format: {dim_id: {"weighted_score": X, ...}, ...}
    company_industry = data["Industry"]
    
    # Identify low-performing dimensions (score <50% of max possible weighted score)
    low_dimensions = []
    for dim in DIMENSIONS:
        # Skip tourism dimension if industry is not relevant
        if dim["id"] == "tourism_infrastructure" and company_industry not in dim["industries"]:
            continue
        
        # Get weighted score (default to 0 if missing)
        dim_score = scores.get(dim["id"], {}).get("weighted_score", 0)
        # Calculate 50% of max possible weighted score
        max_weighted_threshold = (dim["max_subtotal"] * dim["weight"]) * 0.5
        
        if dim_score < max_weighted_threshold:
            low_dimensions.append(dim["name"])
    
    # Fallback if no low dimensions identified
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
    
    # Fallback recommendations if AI fails
    if not response:
        return [
            "1. Increase renewable energy adoption to 50% by 2026 (SDG 12.2) – Reduces reliance on fossil fuels and cuts carbon emissions by 30%.",
            "2. Expand product take-back programs to cover 50% of product lines (SDG 12.5) – Enhances circularity and reduces end-of-life waste by 25%.",
            "3. Implement eco-design certifications for all core products (SDG 12.3) – Improves product recyclability and aligns with global sustainability standards."
        ]
    
    # Clean and format recommendations
    return [line.strip() for line in response.split("\n") if line.strip() and line.strip()[0].isdigit()]

def ai_generate_mock_esg():
    """Generate a project-focused mock ESG excerpt (mimicking real reports)."""
    data = st.session_state["production_data"]
    scores = data["dimension_scores"]
    company_industry = data["Industry"]
    
    # Extract key metrics for project context
    renewable_pct = data["resource_efficiency"]["renewable_energy_pct"]
    recycled_mat_pct = data["sustainable_production"]["recycled_material_pct"]
    waste_recycling_pct = data["chemical_waste"]["waste_recycling_pct"]
    water_reuse_pct = data["resource_efficiency"]["water_reuse_pct"]
    
    # Identify 1-2 high-impact projects based on strong metrics
    projects = []
    if renewable_pct >= 40:
        projects.append(f"onsite solar installation (completed Q2 2024)")
    if recycled_mat_pct >= 30:
        projects.append(f"closed-loop material recycling program (launched 2023)")
    if water_reuse_pct >= 50:
        projects.append(f"water reclamation system upgrade (operational since Jan 2024)")
    
    # Default project if no strong metrics
    if not projects:
        projects = [f"pilot sustainable procurement initiative (initiated Q3 2024)"]
    
    prompt = f"""Write a 300-500 word excerpt from an ESG report for {data['Company Name']} (Industry: {company_industry}). 
    Focus EXCLUSIVELY on a specific sustainability project/action (not company overview) that demonstrates progress toward SDG 12. 
    Follow these rules:
    1. Open with the project name and timeline (e.g., "In 2024, we expanded our waste recycling program...").
    2. Include 3+ specific metrics (e.g., "reduced hazardous waste by 27%", "15,000 tons of recycled material reused").
    3. Link directly to 1+ SDG 12 targets (e.g., SDG 12.2 for resource efficiency, SDG 12.5 for circular economy).
    4. Mention challenges (e.g., "initial supplier resistance") and mitigation (e.g., "partnered with 3 local recyclers").
    5. Add forward-looking data (e.g., "targeting 45% renewable energy by 2026").
    6. Use formal, concise language (avoid jargon; mimic tone of Unilever/Apple ESG reports).
    7. Do NOT reintroduce the company or its mission—assume readers know this from earlier sections.
    
    Key metrics to reference:
    - Renewable energy: {renewable_pct}% of production needs
    - Recycled materials: {recycled_mat_pct}% in core products
    - Waste recycling: {waste_recycling_pct}% of production waste
    - Featured project examples: {', '.join(projects)}"""
    
    # Realistic fallback excerpt (mimicking Apple/Unilever reports)
    fallback = f"""In 2024, {data['Company Name']} scaled its closed-loop material recycling program across three manufacturing facilities, building on the success of our 2023 pilot. The initiative, aligned with SDG 12.5 (substantially reduce waste generation), focused on reclaiming post-production plastic scrap and reprocessing it for use in new product components.

During the reporting period, the program diverted 1,240 metric tons of plastic from landfills—equivalent to 32% of total production waste—representing a 17% increase from 2023. Of this, 890 metric tons (72% of reclaimed material) was reused in our core product line, reducing virgin plastic procurement by {recycled_mat_pct}% and lowering carbon emissions by 1,850 tons (due to reduced transportation of raw materials).

Implementation challenges included inconsistent scrap quality, which initially limited reuse rates to 58%. To address this, we partnered with two third-party recycling specialists to upgrade sorting equipment, increasing usable output by 24% within six months.

Looking ahead, we aim to expand the program to all five manufacturing sites by 2026, targeting 45% of production waste recycling (up from current {waste_recycling_pct}%) and 50% recycled material content in core products—directly contributing to SDG 12.3 (halve per capita global food waste) and our broader 2030 carbon neutrality commitment."""
    
    return get_ai_response(prompt, system_msg="You are an ESG report writer. Mimic the concise, data-heavy style of Fortune 500 ESG reports.") or fallback

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

# --- ESG Report Writing Recommendations ---
def display_writing_recommendations():
    st.subheader("✏️ ESG Report Writing Guidelines (SDG 12 Focus)")
    st.markdown("""
    Use these best practices to draft credible, impactful ESG report sections focused on production sustainability:
    
    ### 1. Focus on Specific Projects (Not General Claims)
    - **Bad**: "We improved resource efficiency."  
    - **Good**: "In 2024, we installed a water reclamation system at our Detroit plant, reducing freshwater use by 40M gallons/year (32% of total consumption)."  

    ### 2. Anchor Metrics to Context
    Always include:  
    - Baseline comparisons (e.g., "up from 18% in 2022")  
    - Industry benchmarks (e.g., "exceeds sector average of 22%")  
    - Future targets (e.g., "targeting 50% by 2026")  

    ### 3. Explicitly Link to SDG 12 Targets
    Specify which target your action supports:  
    - SDG 12.2: Sustainable management of resources (energy/water efficiency)  
    - SDG 12.5: Reduce waste generation (recycling, circular design)  
    - SDG 12.7: Sustainable procurement (supplier standards)  

    ### 4. Acknowledge Challenges
    Build credibility by noting setbacks and solutions:  
    - "Initial recycling rates were 15% below target due to supplier quality issues. We addressed this by launching a training program, improving rates to 85% by Q4."  

    ### 5. Avoid Greenwashing Jargon
    Replace vague terms with specifics:  
    - Instead of "eco-friendly," use "100% renewable electricity-powered production"  
    - Instead of "circular economy," use "90% of product components are recyclable"  

    ### 6. Highlight Stakeholder Collaboration
    Show collective impact:  
    - "Partnered with the Ellen MacArthur Foundation to redesign 3 product lines for circularity, reducing material waste by 27%."  
    """)

# --- Sidebar UI (Data Input) ---
st.sidebar.header("📊 Data Input")

# 1. PDF Upload
st.sidebar.subheader("1. Upload ESG Report (Optional)")
uploaded_pdf = st.sidebar.file_uploader("Upload PDF for AI Data Extraction", type="pdf")
if uploaded_pdf:
    with st.spinner("🔍 Extracting text from PDF..."):
        pdf_text = extract_pdf_text(uploaded_pdf)
        st.session_state["production_data"]["extracted_pdf_text"] = pdf_text
        
        # Show more extracted text for user verification
        with st.sidebar.expander("View Extracted Text (for verification)", expanded=False):
            st.text_area("PDF Content (first 5000 chars)", pdf_text[:5000], height=300, disabled=True)
        
        if OPENAI_AVAILABLE:
            with st.spinner("🤖 Analyzing ESG data (looking for metrics like 'renewable energy' or 'recycled materials')..."):
                esg_data = ai_extract_esg_data(pdf_text)
                if esg_data:
                    # Update session state with extracted data (only non-null values)
                    updated = False
                    for key, value in esg_data.items():
                        if key in st.session_state["production_data"] and value is not None:
                            # For nested dimensions, only update non-null metrics
                            if isinstance(value, dict):
                                for subkey, subval in value.items():
                                    if subval is not None:
                                        st.session_state["production_data"][key][subkey] = subval
                                        updated = True
                            else:  # For top-level fields (company name, industry)
                                st.session_state["production_data"][key] = value
                                updated = True
                    if updated:
                        st.sidebar.success("✅ Populated available metrics from PDF! Review and fill missing fields below.")
                else:
                    st.sidebar.warning("⚠️ No valid data extracted. Please enter data manually.")

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

# 6. ESG Writing Guidelines
display_writing_recommendations()

# 7. Report Export
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
