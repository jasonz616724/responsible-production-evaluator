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

# --- Page Configuration (Neutral Branding) ---
st.set_page_config(page_title="SDG 12 Production Responsibility Evaluator", layout="wide")
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- Initialize OpenAI Client (Generalized) ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except KeyError:
    st.warning("⚠️ OPENAI_API_KEY not found in Streamlit Secrets. AI features disabled.")
    OPENAI_AVAILABLE = False
except Exception as e:
    st.error(f"⚠️ OpenAI client error: {str(e)}")
    OPENAI_AVAILABLE = False

# --- Session State Initialization (100% Neutral) ---
if "production_data" not in st.session_state:
    st.session_state["production_data"] = {
        "Company Name": "Enter Company Name",
        "Industry": "Manufacturing",  # Neutral default
        "extracted_pdf_text": "",
        "extracted_sections": {},  # Stores key ESG chapters (Environment, Circular Economy)
        # Core SDG 12 Dimensions (Universal Metrics)
        "resource_efficiency": {"renewable_energy_pct": 0, "energy_tech_count": 0, "water_reuse_pct": 0, "ai_score": 0},
        "sustainable_production": {"recycled_material_pct": 0, "waste_intensity_pct": 0, "eco_design_cert": False, "ai_score": 0},
        "chemical_waste": {"hazardous_reduction_pct": 0, "waste_recycling_pct": 0, "chemical_compliance": False, "ai_score": 0},
        "circular_economy": {"takeback_program_pct": 0, "packaging_sustainable_pct": 0, "certified_supplier_pct": 0, "ai_score": 0},
        "sustainable_procurement": {"procurement_criteria_count": 0, "sustainable_budget_pct": 0, "procurement_tracking": False, "ai_score": 0},
        "life_cycle_thinking": {"lca_product_pct": 0, "consumer_comm": False, "lca_improvements": 0, "ai_score": 0},
        "waste_management": {"food_waste_reduction_pct": 0, "segregation_rate_pct": 0, "circular_partnerships": False, "ai_score": 0},
        "industry_specific": {"sector_metric_1": 0, "sector_metric_2": False, "sector_metric_3": 0, "ai_score": 0},  # Adaptable for any industry
        # Final Outputs
        "total_score": 0,
        "dimension_scores": {},
        "ai_recommendations": [],
        "mock_esg_excerpt": ""
    }

if "rerun_trigger" not in st.session_state:
    st.session_state["rerun_trigger"] = False

# --- Constants: Generalized SDG 12 Scoring Framework ---
# Expanded to support 8+ industries; industry-specific dimension adapts dynamically
DIMENSIONS = [
    {
        "id": "resource_efficiency",
        "name": "Resource Efficiency (SDG 12.2)",
        "weight": 0.25,
        "actions": [
            {"name": "renewable_energy_pct", "desc": "Renewable energy percentage (e.g., solar, wind)", "calc": lambda x: 10 if x >=50 else 5 if x >=30 else 0},
            {"name": "energy_tech_count", "desc": "Number of energy-efficient technologies (e.g., LED, heat pumps)", "calc": lambda x: min(10, x * 5)},
            {"name": "water_reuse_pct", "desc": "Water reuse/recycling rate (vs. total consumption)", "calc": lambda x: 10 if x >=70 else 5 if x >=40 else 0}
        ],
        "ai_criteria": "Evaluate 1) energy intensity vs. industry peers, 2) water conservation practices. Return 0-5 (no text).",
        "max_subtotal": 30,
        "industries": ["All"]  # Appears for all sectors
    },
    {
        "id": "sustainable_production",
        "name": "Sustainable Production (SDG 12.3)",
        "weight": 0.20,
        "actions": [
            {"name": "recycled_material_pct", "desc": "Recycled/upcycled material percentage in production", "calc": lambda x: 10 if x >=40 else 5 if x >=20 else 0},
            {"name": "waste_intensity_pct", "desc": "Waste intensity vs. industry average (%)", "calc": lambda x: 10 if x <=20 else 5 if x <=40 else 0},
            {"name": "eco_design_cert", "desc": "Eco-design certification (e.g., Cradle to Cradle, EU Ecolabel)", "calc": lambda x: 10 if x else 0}
        ],
        "ai_criteria": "Evaluate 1) material yield efficiency, 2) product recyclability potential. Return 0-5 (no text).",
        "max_subtotal": 30,
        "industries": ["All"]
    },
    {
        "id": "chemical_waste",
        "name": "Chemicals & Waste Management (SDG 12.4)",
        "weight": 0.18,
        "actions": [
            {"name": "hazardous_reduction_pct", "desc": "Hazardous chemical reduction vs. baseline (%)", "calc": lambda x: 7 if x >=50 else 3 if x >=20 else 0},
            {"name": "waste_recycling_pct", "desc": "Total production waste recycling rate (%)", "calc": lambda x: 7 if x >=80 else 3 if x >=50 else 0},
            {"name": "chemical_compliance", "desc": "Compliance with global chemical standards (e.g., REACH, OSHA)", "calc": lambda x: 6 if x else 0}
        ],
        "ai_criteria": "Evaluate 1) chemical spill prevention protocols, 2) hazardous waste treatment. Return 0-3 (no text).",
        "max_subtotal": 20,
        "industries": ["Manufacturing", "Construction", "Chemicals", "Food & Beverage"]  # Relevant sectors only
    },
    {
        "id": "circular_economy",
        "name": "Circular Economy Integration (SDG 12.5)",
        "weight": 0.10,
        "actions": [
            {"name": "takeback_program_pct", "desc": "Product lines with take-back/recycling programs (%)", "calc": lambda x: 4 if x >=50 else 2 if x >=20 else 0},
            {"name": "packaging_sustainable_pct", "desc": "Sustainable packaging percentage (recyclable/compostable)", "calc": lambda x: 3 if x >=80 else 1 if x >=50 else 0},
            {"name": "certified_supplier_pct", "desc": "Suppliers with sustainability certifications (%)", "calc": lambda x: 3 if x >=60 else 1 if x >=30 else 0}
        ],
        "ai_criteria": "Evaluate 1) tier 2 supplier sustainability practices, 2) product-as-a-service models. Return 0-2 (no text).",
        "max_subtotal": 10,
        "industries": ["All"]
    },
    {
        "id": "sustainable_procurement",
        "name": "Sustainable Procurement (SDG 12.7)",
        "weight": 0.08,
        "actions": [
            {"name": "procurement_criteria_count", "desc": "Sustainability criteria in procurement policy (e.g., carbon, labor)", "calc": lambda x: 3 if x >=3 else 1 if x >=1 else 0},
            {"name": "sustainable_budget_pct", "desc": "Procurement budget for sustainable goods/services (%)", "calc": lambda x: 3 if x >=30 else 1 if x >=10 else 0},
            {"name": "procurement_tracking", "desc": "Tracking of sustainable procurement performance", "calc": lambda x: 2 if x else 0}
        ],
        "ai_criteria": "Evaluate 1) supplier diversity, 2) alignment with ISO 20400. Return 0-2 (no text).",
        "max_subtotal": 8,
        "industries": ["All"]
    },
    {
        "id": "life_cycle_thinking",
        "name": "Life-Cycle Thinking (SDG 12.8)",
        "weight": 0.05,
        "actions": [
            {"name": "lca_product_pct", "desc": "Product lines with Life-Cycle Assessment (LCA) (%)", "calc": lambda x: 2 if x >=50 else 1 if x >=20 else 0},
            {"name": "consumer_comm", "desc": "Sustainability information shared with consumers", "calc": lambda x: 2 if x else 0},
            {"name": "lca_improvements", "desc": "Product improvements from LCA insights", "calc": lambda x: 1 if x >=2 else 0.5 if x >=1 else 0}
        ],
        "ai_criteria": "Evaluate 1) scope 3 emissions inclusion in LCA, 2) consumer engagement with eco-labels. Return 0-1 (no text).",
        "max_subtotal": 5,
        "industries": ["All"]
    },
    {
        "id": "waste_management",
        "name": "Waste Generation & Management (SDG 12.5.1)",
        "weight": 0.07,
        "actions": [
            {"name": "food_waste_reduction_pct", "desc": "Food/by-product waste reduction vs. baseline (%)", "calc": lambda x: 3 if x >=40 else 1 if x >=20 else 0},
            {"name": "segregation_rate_pct", "desc": "Waste segregation rate at source (%)", "calc": lambda x: 3 if x >=90 else 1 if x >=60 else 0},
            {"name": "circular_partnerships", "desc": "Partnerships for circular waste solutions (e.g., recycling firms)", "calc": lambda x: 1 if x else 0}
        ],
        "ai_criteria": "Evaluate 1) recycled waste purity, 2) revenue from by-product utilization. Return 0-2 (no text).",
        "max_subtotal": 7,
        "industries": ["Food & Beverage", "Retail", "Hospitality", "Manufacturing"]  # Relevant sectors
    },
    {
        "id": "industry_specific",
        "name": "Industry-Specific Sustainability (SDG 12)",
        "weight": 0.07,
        "actions": [
            {"name": "sector_metric_1", "desc": "Industry-specific metric 1 (e.g., 'Renewable fuel %' for Transport)", "calc": lambda x: 3 if x >=50 else 1 if x >=20 else 0},
            {"name": "sector_metric_2", "desc": "Industry-specific metric 2 (e.g., 'Eco-tourism certification' for Tourism)", "calc": lambda x: 3 if x else 0},
            {"name": "sector_metric_3", "desc": "Industry-specific metric 3 (e.g., 'Zero-waste store %' for Retail)", "calc": lambda x: 1 if x >=3 else 0.5 if x >=1 else 0}
        ],
        "ai_criteria": "Evaluate 1) industry-specific sustainability leadership, 2) alignment with sector SDG 12 targets. Return 0-2 (no text).",
        "max_subtotal": 7,
        "industries": ["All"],  # Adapts via dynamic labeling
        "sector_labels": {  # Dynamic metric names per industry
            "Manufacturing": {"metric1": "Lean production adoption %", "metric2": "Zero-waste factory count", "metric3": "Remanufacturing volume %"},
            "Retail": {"metric1": "Zero-waste store %", "metric2": "Reusable packaging program", "metric3": "Product repair service count"},
            "Transport": {"metric1": "Renewable fuel %", "metric2": "Electric fleet %", "metric3": "Carbon-neutral route count"},
            "Food & Beverage": {"metric1": "Organic ingredient %", "metric2": "Biodegradable packaging %", "metric3": "Food donation volume %"},
            "Construction": {"metric1": "Sustainable material %", "metric2": "LEED-certified projects %", "metric3": "Construction waste reuse %"},
            "Tourism": {"metric1": "Eco-tourism certification %", "metric2": "Water conservation program", "metric3": "Local sourcing %"},
            "Healthcare": {"metric1": "Medical waste recycling %", "metric2": "Energy-efficient equipment %", "metric3": "Sustainable pharmacy program %"},
            "Technology": {"metric1": "E-waste takeback %", "metric2": "Energy Star certification %", "metric3": "Circuit board recycling %"}
        }
    }
]

# --- Core AI Functions (Generalized for All Industries) ---
def get_ai_response(prompt, system_msg="You are a global ESG expert. Be concise and industry-agnostic."):
    if not OPENAI_AVAILABLE:
        return "❌ AI features require an OPENAI_API_KEY in Streamlit Secrets."
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            temperature=0.4,  # Balanced for generalizability
            timeout=30
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"⚠️ AI error: {str(e)}")
        return None

def extract_pdf_text(uploaded_file):
    """Extract key ESG sections from ANY company's report (no industry bias)"""
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf_doc:
            full_text = ""
            section_text = {
                "environment": "",    # Universal ESG chapter
                "circular_economy": "",# Universal ESG chapter
                "procurement": ""      # Universal ESG chapter
            }
            
            # Identify key ESG chapters (case-insensitive, industry-agnostic)
            current_section = None
            for page_num, page in enumerate(pdf_doc):
                page_text = page.get_text().strip()
                full_text += f"\n--- Page {page_num + 1} ---\n" + page_text
                
                # Trigger section detection for common ESG chapter titles
                if any(term in page_text.lower() for term in ["environment", "climate", "resource use"]):
                    current_section = "environment"
                elif any(term in page_text.lower() for term in ["circular economy", "take-back", "packaging"]):
                    current_section = "circular_economy"
                elif any(term in page_text.lower() for term in ["procurement", "supplier", "sustainable sourcing"]):
                    current_section = "procurement"
                elif any(term in page_text.lower() for term in ["governance", "dei", "social impact"]) and "chapter" in page_text.lower():
                    current_section = None  # Exit non-SDG 12 chapters
                
                # Append text to active section
                if current_section and page_text:
                    section_text[current_section] += page_text + "\n"
            
            st.session_state["production_data"]["extracted_sections"] = section_text
            return full_text[:50000]  # Cap at 50k chars (covers key ESG sections)
    except Exception as e:
        st.error(f"⚠️ PDF extraction failed: {str(e)} (report may be password-protected)")
        return ""

def ai_extract_esg_data():
    """Extract SDG 12 data from ANY company's ESG report (industry-agnostic)"""
    data = st.session_state["production_data"]
    section_text = data["extracted_sections"]
    company_industry = data["Industry"]
    
    # Prioritize universal ESG sections
    target_text = ""
    if section_text["environment"]:
        target_text += f"[Environment Section]\n{section_text['environment'][:15000]}\n"
    if section_text["circular_economy"]:
        target_text += f"[Circular Economy Section]\n{section_text['circular_economy'][:10000]}\n"
    if section_text["procurement"]:
        target_text += f"[Procurement Section]\n{section_text['procurement'][:10000]}\n"
    
    if not target_text:
        target_text = data["extracted_pdf_text"][:30000]  # Fallback to full text
    
    # Universal terminology map (works for all industries)
    general_terminology = """
    ESG Terms to Map to SDG 12 Metrics (All Industries):
    - "Renewable energy %", "Solar/wind adoption": renewable_energy_pct
    - "Water reuse", "Water recycling rate": water_reuse_pct
    - "Recycled material content", "Upcycled input": recycled_material_pct
    - "Waste recycling rate", "Circular waste recovery": waste_recycling_pct
    - "Supplier sustainability certification", "EcoVadis score": certified_supplier_pct
    - "Life-cycle assessment", "LCA completed": lca_product_pct
    - "Product take-back", "End-of-life program": takeback_program_pct
    - "Sustainable packaging", "Compostable packaging": packaging_sustainable_pct
    """

    prompt = f"""Extract SDG 12 production metrics from this company's ESG Report (Industry: {company_industry}).
    CRITICAL RULES:
    1. Return ONLY NUMBERS (no percentages, e.g., write 83 not "83%")
    2. Return 0 for missing data (NEVER use null)
    3. For booleans, return true/false ONLY
    4. Adapt to the company's industry (e.g., "electric fleet %" for Transport, "organic ingredient %" for Food & Beverage)
    
    {general_terminology}
    
    Return ONLY valid JSON (no extra text):
    {{
        "company_name": "Company name (extract from report)",
        "industry": "{company_industry} (keep as is)",
        "resource_efficiency": {{
            "renewable_energy_pct": "Number (0-100, no %)",
            "energy_tech_count": "Number (0+)",
            "water_reuse_pct": "Number (0-100)"
        }},
        "sustainable_production": {{
            "recycled_material_pct": "Number (0-100)",
            "waste_intensity_pct": "Number (0+)",
            "eco_design_cert": "true/false"
        }},
        "chemical_waste": {{
            "hazardous_reduction_pct": "Number (0-100)",
            "waste_recycling_pct": "Number (0-100)",
            "chemical_compliance": "true/false"
        }},
        "circular_economy": {{
            "takeback_program_pct": "Number (0-100)",
            "packaging_sustainable_pct": "Number (0-100)",
            "certified_supplier_pct": "Number (0-100)"
        }},
        "sustainable_procurement": {{
            "procurement_criteria_count": "Number (0+)",
            "sustainable_budget_pct": "Number (0-100)",
            "procurement_tracking": "true/false"
        }},
        "life_cycle_thinking": {{
            "lca_product_pct": "Number (0-100)",
            "consumer_comm": "true/false",
            "lca_improvements": "Number (0+)"
        }},
        "waste_management": {{
            "food_waste_reduction_pct": "Number (0-100, 0 if not applicable to industry)",
            "segregation_rate_pct": "Number (0-100)",
            "circular_partnerships": "true/false"
        }},
        "industry_specific": {{
            "sector_metric_1": "Industry-specific number (e.g., 40 for '40% electric fleet' in Transport)",
            "sector_metric_2": "Industry-specific boolean (e.g., true for 'reusable packaging' in Retail)",
            "sector_metric_3": "Industry-specific number (e.g., 2 for '2 zero-waste stores' in Retail)"
        }}
    }}
    
    ESG Report Text: {target_text}
    """
    
    response = get_ai_response(prompt, system_msg="You extract ESG data for ANY industry. Return ONLY valid JSON—no explanations.")
    if not response:
        return {}
    
    try:
        # Clean AI response (remove code blocks if present)
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:-3].strip()
        
        extracted_data = json.loads(cleaned_response)
        
        # --- Universal Data Cleaning ---
        for dim in extracted_data:
            if isinstance(extracted_data[dim], dict):
                for metric in extracted_data[dim]:
                    value = extracted_data[dim][metric]
                    # Remove percentage symbols (e.g., "75%" → 75)
                    if isinstance(value, str) and "%" in value:
                        extracted_data[dim][metric] = float(value.replace("%", "").strip())
                    # Convert string numbers to integers (e.g., "5" → 5)
                    if isinstance(value, str) and value.isdigit():
                        extracted_data[dim][metric] = int(value)
                    # Replace null with industry-appropriate defaults
                    if value is None:
                        if "pct" in metric or "count" in metric:
                            extracted_data[dim][metric] = 0
                        else:
                            extracted_data[dim][metric] = False
        
        # Ensure industry consistency
        extracted_data["industry"] = company_industry
        return extracted_data
    
    except json.JSONDecodeError as e:
        st.sidebar.error(f"⚠️ Data Parse Error: {str(e)}. Cleaned response: {cleaned_response[:200]}")
        # Fallback: Return empty dict to trigger manual input
        return {}

def ai_evaluate_unlisted_criteria(dimension):
    """Evaluate AI-only criteria for ANY industry"""
    data = st.session_state["production_data"]
    dim_data = data[dimension["id"]]
    company_industry = data["Industry"]
    section_text = data["extracted_sections"].get(dimension["id"], "")
    
    prompt = f"""Evaluate {dimension['name']} for a {company_industry} company:
    - Company Data: {dim_data}
    - ESG Report Context: {section_text[:2000]}
    - Evaluation Criteria: {dimension['ai_criteria']}
    Return ONLY the numeric score (0-5, no text)."""
    
    response = get_ai_response(prompt, system_msg="You analyze ESG performance for ANY industry. Return ONLY a number.")
    if not response:
        return 0
    try:
        return max(0, min(5, float(response.strip())))
    except (ValueError, TypeError):
        st.warning(f"⚠️ Invalid AI score for {dimension['name']}. Defaulting to 0.")
        return 0

def ai_generate_recommendations():
    """Generate industry-specific SDG 12 recommendations"""
    data = st.session_state["production_data"]
    scores = data["dimension_scores"]
    company_industry = data["Industry"]
    
    # Identify low-performing dimensions (universal logic)
    low_dimensions = []
    for dim in DIMENSIONS:
        # Skip dimensions irrelevant to the industry
        if company_industry not in dim["industries"] and dim["industries"] != ["All"]:
            continue
        
        dim_score = scores.get(dim["id"], {}).get("weighted_score", 0)
        max_threshold = (dim["max_subtotal"] * dim["weight"]) * 0.5
        if dim_score < max_threshold:
            low_dimensions.append(dim["name"])
    
    if not low_dimensions:
        low_dimensions = ["Resource Efficiency (SDG 12.2)", "Circular Economy Integration (SDG 12.5)"]
    
    prompt = f"""Generate 3 specific SDG 12 recommendations for a {company_industry} company.
    Focus on low-performing areas: {low_dimensions}.
    Each recommendation must:
    1. Link to a specific SDG 12 target (e.g., SDG 12.2 for resource efficiency).
    2. Be industry-relevant (e.g., "Expand electric fleet" for Transport, "Add zero-waste stores" for Retail).
    3. Include a measurable goal (e.g., "Reach 50% renewable energy by 2026").
    4. Explain impact (e.g., "Cuts carbon emissions by 25%").
    Number recommendations (1., 2., 3.)—no bullets."""
    
    response = get_ai_response(prompt, system_msg="You create industry-specific sustainability recommendations. Be actionable.")
    if not response:
        # Industry-agnostic fallback recommendations
        return [
            "1. Increase renewable energy adoption to 50% by 2026 (SDG 12.2) – Reduces fossil fuel reliance and cuts operational emissions by 30%.",
            "2. Expand product take-back programs to 50% of product lines (SDG 12.5) – Boosts circularity and reduces end-of-life waste by 25%.",
            "3. Implement 3 sustainability criteria in procurement policies (SDG 12.7) – Improves supply chain sustainability and aligns with global standards."
        ]
    return [line.strip() for line in response.split("\n") if line.strip() and line.strip()[0].isdigit()]

def ai_generate_mock_esg():
    """Generate industry-specific ESG excerpt (mimics real reports)"""
    data = st.session_state["production_data"]
    scores = data["dimension_scores"]
    company_name = data["Company Name"]
    company_industry = data["Industry"]
    
    # Identify strengths/improvements (universal logic)
    high_dims = [dim["name"] for dim in DIMENSIONS if scores.get(dim["id"], {}).get("weighted_score", 0) > (dim["max_subtotal"] * dim["weight"]) * 0.7]
    low_dims = [dim["name"] for dim in DIMENSIONS if scores.get(dim["id"], {}).get("weighted_score", 0) < (dim["max_subtotal"] * dim["weight"]) * 0.5]
    
    # Industry-specific project examples
    sector_projects = {
        "Manufacturing": "lean production initiative and equipment upgrades",
        "Retail": "zero-waste store rollout and reusable packaging program",
        "Transport": "electric fleet expansion and renewable fuel adoption",
        "Food & Beverage": "organic ingredient sourcing and food waste diversion program",
        "Construction": "sustainable material procurement and LEED certification push",
        "Tourism": "eco-tourism certification and local community sourcing",
        "Healthcare": "medical waste recycling and energy-efficient clinic upgrades",
        "Technology": "e-waste takeback and energy-efficient product design"
    }
    project = sector_projects.get(company_industry, "sustainability improvement project")
    
    prompt = f"""Write a 300-500 word ESG report excerpt for {company_name} (Industry: {company_industry}).
    Focus on a specific sustainability project: {project}.
    Include:
    1. Project timeline (e.g., "launched in 2023, expanded in 2024").
    2. 3+ industry-relevant metrics (e.g., "40% electric fleet" for Transport, "75% recycled packaging" for Retail).
    3. Strengths: {high_dims if high_dims else ['Sustainable procurement']}.
    4. Improvements: {low_dims if low_dims else ['Resource efficiency']}.
    5. Future goal tied to SDG 12 (e.g., "2030 circular production target").
    Use formal ESG report tone (no jargon). Do NOT reintroduce the company.
    
    Key Metrics to Reference:
    - Renewable energy: {data['resource_efficiency']['renewable_energy_pct']}%
    - Waste recycling: {data['chemical_waste']['waste_recycling_pct']}%
    - Sustainable packaging: {data['circular_economy']['packaging_sustainable_pct']}%
    - Industry metric: {data['industry_specific']['sector_metric_1']}%
    """
    
    fallback = f"""In 2024, {company_name} advanced its {project}, building on progress toward SDG 12 (Responsible Consumption and Production). The initiative focused on reducing operational environmental impact while aligning with {company_industry}-specific sustainability priorities.

During the year, the project delivered measurable results: renewable energy accounted for {data['resource_efficiency']['renewable_energy_pct']}% of total consumption, waste recycling reached {data['chemical_waste']['waste_recycling_pct']}%, and {data['circular_economy']['packaging_sustainable_pct']}% of packaging was sustainable (recyclable or compostable). Industry-specific milestones included {data['industry_specific']['sector_metric_1']}% adoption of a key sustainability practice, supported by partnerships with local recycling firms and sustainable suppliers.

Strengths included {high_dims[0]} (exceeding 2024 targets) and sustainable procurement (89% of high-risk suppliers met minimum sustainability criteria). Areas for improvement included {low_dims[0]} (currently below industry benchmarks) and water reuse (at {data['resource_efficiency']['water_reuse_pct']}%, below the 40% target).

Looking ahead, {company_name} aims to scale the project by 2026, targeting 50% renewable energy (SDG 12.2), 90% sustainable packaging (SDG 12.5), and {data['industry_specific']['sector_metric_1'] + 10}% adoption of the industry-specific practice—aligning with its long-term vision of fully circular operations by 2030."""
    
    return get_ai_response(prompt, system_msg="You write ESG excerpts for ANY industry. Use a formal, data-heavy tone.") or fallback

# --- Helper Functions (Generalized) ---
def calculate_dimension_scores():
    """Calculate SDG 12 scores for ANY industry"""
    data = st.session_state["production_data"]
    dimension_scores = {}
    total_score = 0
    company_industry = data["Industry"]
    
    for dim in DIMENSIONS:
        # Skip dimensions irrelevant to the company's industry
        if company_industry not in dim["industries"] and dim["industries"] != ["All"]:
            continue
        
        # Calculate action-based subtotal
        dim_data = data[dim["id"]]
        action_subtotal = sum([action["calc"](dim_data[action["name"]]) for action in dim["actions"]])
        
        # Add AI score (industry-adapted)
        ai_score = ai_evaluate_unlisted_criteria(dim) if OPENAI_AVAILABLE else 0
        dim_data["ai_score"] = ai_score
        total_subtotal = min(dim["max_subtotal"], action_subtotal + ai_score)
        
        # Weighted score (total = 100)
        weighted_score = round(total_subtotal * dim["weight"], 1)
        dimension_scores[dim["id"]] = {
            "name": dim["name"],
            "subtotal": total_subtotal,
            "weighted_score": weighted_score,
            "max_weighted": round(dim["max_subtotal"] * dim["weight"], 1)
        }
        total_score += weighted_score
    
    data["dimension_scores"] = dimension_scores
    data["total_score"] = round(total_score, 1)
    st.session_state["production_data"] = data

def generate_report_content():
    """Generate general-purpose SDG 12 report"""
    data = st.session_state["production_data"]
    scores = data["dimension_scores"]
    company_industry = data["Industry"]
    
    content = f"SDG 12 Production Responsibility Report\n"
    content += f"Company: {data['Company Name']}\n"
    content += f"Industry: {company_industry}\n"
    content += f"Total Score: {data['total_score']}/100\n"
    content += "="*50 + "\n\n"
    
    # Dimension Breakdown (Universal Format)
    content += "1. SDG 12 Dimension Score Breakdown\n"
    for dim_id, dim_data in scores.items():
        content += f"- {dim_data['name']}: {dim_data['weighted_score']}/{dim_data['max_weighted']}\n"
    
    # Key Metrics (Industry-Agnostic)
    content += "\n2. Key Production Sustainability Metrics\n"
    core_dims = ["resource_efficiency", "sustainable_production", "circular_economy"]
    for dim_id in core_dims:
        if dim_id not in data:
            continue
        dim_data = data[dim_id]
        dim_name = next(d["name"] for d in DIMENSIONS if d["id"] == dim_id)
        content += f"- {dim_name}:\n"
        for action in [a for a in DIMENSIONS[[d["id"] for d in DIMENSIONS].index(dim_id)]["actions"]]:
            content += f"  - {action['desc']}: {dim_data[action['name']]}\n"
    
    # Industry-Specific Metrics
    content += f"\n3. {company_industry}-Specific Metrics\n"
    sector_dim = next(d for d in DIMENSIONS if d["id"] == "industry_specific")
    sector_labels = sector_dim["sector_labels"].get(company_industry, {"metric1": "Metric 1", "metric2": "Metric 2", "metric3": "Metric 3"})
    content += f"- {sector_labels['metric1']}: {data['industry_specific']['sector_metric_1']}\n"
    content += f"- {sector_labels['metric2']}: {'Yes' if data['industry_specific']['sector_metric_2'] else 'No'}\n"
    content += f"- {sector_labels['metric3']}: {data['industry_specific']['sector_metric_3']}\n"
    
    # Recommendations + Excerpt
    content += "\n4. Improvement Recommendations\n"
    recommendations = ai_generate_recommendations() if OPENAI_AVAILABLE else data["ai_recommendations"]
    for i, rec in enumerate(recommendations, 1):
        content += f"{i}. {rec}\n"
    
    content += "\n5. Mock ESG Report Excerpt\n"
    mock_excerpt = ai_generate_mock_esg() if OPENAI_AVAILABLE else data["mock_esg_excerpt"]
    content += mock_excerpt + "\n"
    
    return content, recommendations, mock_excerpt

# --- ESG Report Writing Guidelines (Generalized) ---
def display_writing_recommendations():
    st.subheader("✏️ General ESG Report Writing Guidelines (SDG 12 Focus)")
    st.markdown("""
    Use these best practices to draft credible SDG 12 sections for ANY industry:
    
    ### 1. Focus on Specific, Industry-Relevant Projects
    - **Bad**: "We improved sustainability."  
    - **Good (Manufacturing)**: "In 2024, we upgraded 3 factories to lean production, cutting waste by 35% and saving 2M gallons of water."  
    - **Good (Retail)**: "We launched 5 zero-waste stores, diverting 90% of waste from landfills via in-store recycling programs."

    ### 2. Tie Metrics to Baselines & Targets
    Always include context:  
    - Baselines: "Renewable energy up from 25% in 2022 to 40% in 2024"  
    - Industry benchmarks: "Exceeds the retail sector average of 30% for sustainable packaging"  
    - Future goals: "Targeting 60% electric fleet by 2026 (SDG 12.2)"

    ### 3. Explicitly Link to SDG 12 Targets
    Specify which SDG 12 target your action supports:  
    - SDG 12.2: Sustainable resource use (energy/water efficiency)  
    - SDG 12.3: Halve food waste (or production waste for non-food industries)  
    - SDG 12.5: Reduce waste generation (recycling, circular design)  
    - SDG 12.7: Sustainable procurement (supplier standards)

    ### 4. Be Transparent About Gaps
    Build credibility by addressing setbacks:  
    - "Water reuse fell short of 40% target (reached 28%) due to delayed treatment plant installation—we’ll complete upgrades by Q2 2025."

    ### 5. Avoid Industry-Agnostic Jargon
    Use sector-specific language:  
    - Instead of "sustainable transport," use "electric delivery vans" (logistics)  
    - Instead of "circular economy," use "product take-back programs" (electronics)

    ### 6. Highlight Stakeholder Collaboration
    Show collective impact:  
    - "Partnered with 10 local farms to source 80% organic ingredients (Food & Beverage)"  
    - "Worked with 5 recycling firms to launch e-waste take-back in 200 stores (Retail)"
    """)

# --- Sidebar UI (General-Purpose Design) ---
st.sidebar.header("📊 SDG 12 Data Input (All Industries)")

# 1. PDF Upload (Any Company's ESG Report)
st.sidebar.subheader("1. Upload ESG Report (Optional)")
uploaded_pdf = st.sidebar.file_uploader("Upload PDF (supports any company's ESG report)", type="pdf", help="Upload 2022-2024 ESG report for AI data extraction")
if uploaded_pdf:
    with st.spinner("🔍 Extracting ESG sections (Environment, Circular Economy)..."):
        pdf_text = extract_pdf_text(uploaded_pdf)
        data = st.session_state["production_data"]
        data["extracted_pdf_text"] = pdf_text
        st.session_state["production_data"] = data
        
        # Show extracted sections for verification
        with st.sidebar.expander("View Extracted ESG Sections", expanded=False):
            section_text = data["extracted_sections"]
            if section_text["environment"]:
                st.text_area("Environment Section (First 2000 Chars)", section_text["environment"][:2000], height=150, disabled=True)
            if section_text["circular_economy"]:
                st.text_area("Circular Economy Section (First 2000 Chars)", section_text["circular_economy"][:2000], height=150, disabled=True)
        
        # Extract data (if AI is available)
        if OPENAI_AVAILABLE:
            with st.spinner(f"🤖 Extracting SDG 12 data for {data['Industry']} industry..."):
                esg_data = ai_extract_esg_data()
                if esg_data:
                    updated = False
                    for key, value in esg_data.items():
                        if key in data and value is not None:
                            if isinstance(value, dict):
                                for subkey, subval in value.items():
                                    if subval is not None:
                                        data[key][subkey] = subval
                                        updated = True
                            else:
                                data[key] = value
                                updated = True
                    if updated:
                        st.sidebar.success("✅ Populated ESG data! Review and edit below.")
                        st.session_state["production_data"] = data

# 2. Company Information (Expanded Industry List)
st.sidebar.subheader("2. Company Information")
company_name = st.sidebar.text_input(
    "Company Name",
    st.session_state["production_data"]["Company Name"],
    key="company_name"
)
# Expanded to 8+ mainstream industries
industries = ["Manufacturing", "Retail", "Transport", "Food & Beverage", "Construction", "Tourism", "Healthcare", "Technology", "Other"]
industry = st.sidebar.selectbox(
    "Industry",
    industries,
    index=industries.index(st.session_state["production_data"]["Industry"]),
    key="industry"
)

# 3. Dimension Inputs (Industry-Adapted Labels)
st.sidebar.subheader("3. SDG 12 Production Sustainability Data")
data = st.session_state["production_data"]
sector_dim = next(d for d in DIMENSIONS if d["id"] == "industry_specific")
sector_labels = sector_dim["sector_labels"].get(industry, {"metric1": "Industry Metric 1 (%)", "metric2": "Industry Metric 2", "metric3": "Industry Metric 3 (Count)"})

for dim in DIMENSIONS:
    # Skip dimensions irrelevant to the selected industry
    if industry not in dim["industries"] and dim["industries"] != ["All"]:
        continue
    
    st.sidebar.markdown(f"**{dim['name']}**")
    dim_data = data[dim["id"]]
    
    # Dynamic label for industry-specific dimension
    for action in dim["actions"]:
        # Get default value (0 if None)
        current_value = dim_data[action["name"]]
        initial_value = current_value if current_value is not None else 0
        
        # Industry-specific label adaptation
        if dim["id"] == "industry_specific":
            if action["name"] == "sector_metric_1":
                action_desc = sector_labels["metric1"]
            elif action["name"] == "sector_metric_2":
                action_desc = sector_labels["metric2"]
            else:
                action_desc = sector_labels["metric3"]
        else:
            action_desc = action["desc"]
        
        # Render input based on metric type
        if "pct" in action["name"] or action["name"] == "sector_metric_1":
            value = st.sidebar.slider(
                f"{action_desc} (%)",
                min_value=0,
                max_value=100,
                value=int(initial_value),
                key=f"{dim['id']}_{action['name']}",
                help=f"Example: 40 = 40% (relevant for {industry} industry)"
            )
        elif "count" in action["name"] or action["name"] == "sector_metric_3":
            value = st.sidebar.number_input(
                f"{action_desc} (Count)",
                min_value=0,
                value=int(initial_value),
                key=f"{dim['id']}_{action['name']}",
                help=f"Example: 5 = 5 locations (relevant for {industry} industry)"
            )
        else:
            value = st.sidebar.checkbox(
                f"{action_desc}",
                value=initial_value if isinstance(initial_value, bool) else False,
                key=f"{dim['id']}_{action['name']}",
                help=f"Check if your company has this practice (relevant for {industry} industry)"
            )
        
        dim_data[action["name"]] = value
    
    data[dim["id"]] = dim_data
    st.sidebar.markdown("---")

# 4. Save Data (Universal Logic)
if st.sidebar.button("💾 Save Data & Calculate SDG 12 Score", use_container_width=True):
    data["Company Name"] = company_name
    data["Industry"] = industry
    calculate_dimension_scores()
    st.sidebar.success(f"✅ Data saved! {company_name}'s SDG 12 score: {data['total_score']}/100")
    st.session_state["rerun_trigger"] = True

# --- Rerun Trigger ---
if st.session_state["rerun_trigger"]:
    st.session_state["rerun_trigger"] = False
    st.rerun()

# --- Main Dashboard (General-Purpose Layout) ---
data = st.session_state["production_data"]
scores = data.get("dimension_scores", {})
total_score = data.get("total_score", 0)

st.title("🌱 SDG 12 Production Responsibility Evaluator (All Industries)")

# 1. Overview Card (Neutral Design)
st.subheader("📋 SDG 12 Evaluation Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Company Name", data["Company Name"])
with col2:
    st.metric("Industry", data["Industry"])
with col3:
    st.metric("Total SDG 12 Score", f"{total_score}/100")

# 2. Score Breakdown Chart (Universal Colors)
st.subheader("📊 SDG 12 Dimension Score Breakdown")
if scores:
    dim_names = [v["name"].split(" (")[0] for v in scores.values()]
    dim_scores = [v["weighted_score"] for v in scores.values()]
    dim_max = [v["max_weighted"] for v in scores.values()]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(dim_names))
    width = 0.35
    
    # Neutral color scheme (works for all industries)
    bars1 = ax.bar(x - width/2, dim_scores, width, label="Achieved Score", color="#2E8B57")
    bars2 = ax.bar(x + width/2, dim_max, width, label="Max Possible Score", color="#D3D3D3", alpha=0.7)
    
    ax.set_xlabel("SDG 12 Sustainability Dimensions")
    ax.set_ylabel("Score")
    ax.set_title(f"SDG 12 Dimension Comparison ({data['Industry']} Industry)")
    ax.set_xticks(x)
    ax.set_xticklabels(dim_names, rotation=45, ha="right")
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f"{height}", ha="center", va="bottom")
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f"{height}", ha="center", va="bottom")
    
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("ℹ️ Upload an ESG report or enter data in the sidebar to generate scores.")

# 3. Detailed Score Table (Universal Format)
st.subheader("📋 Detailed Score Table")
if scores:
    table_data = []
    for dim_id, dim_data in scores.items():
        table_data.append({
            "SDG 12 Dimension": dim_data["name"],
            "Achieved Subtotal": f"{dim_data['subtotal']}/{DIMENSIONS[[d['id'] for d in DIMENSIONS].index(dim_id)]['max_subtotal']}",
            "Weighted Score": f"{dim_data['weighted_score']}/{dim_data['max_weighted']}"
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

# 4. Industry-Specific Recommendations
st.subheader("💡 SDG 12 Improvement Recommendations")
if OPENAI_AVAILABLE and scores:
    if st.button(f"Generate {data['Industry']}-Specific Recommendations", use_container_width=True):
        with st.spinner("🤖 Creating tailored recommendations..."):
            recommendations = ai_generate_recommendations()
            data["ai_recommendations"] = recommendations
            st.session_state["production_data"] = data
    
    if data["ai_recommendations"]:
        for i, rec in enumerate(data["ai_recommendations"], 1):
            st.write(f"{i}. {rec}")
else:
    st.info("ℹ️ Save data and enable AI to generate industry-specific recommendations.")

# 5. Mock ESG Excerpt (Industry-Adapted)
st.subheader("📄 Mock ESG Report Excerpt")
if OPENAI_AVAILABLE:
    if st.button(f"Generate {data['Industry']}-Style ESG Excerpt", use_container_width=True):
        with st.spinner("🤖 Writing ESG excerpt..."):
            mock_excerpt = ai_generate_mock_esg()
            data["mock_esg_excerpt"] = mock_excerpt
            st.session_state["production_data"] = data
    
    if data["mock_esg_excerpt"]:
        st.write(data["mock_esg_excerpt"])
else:
    st.info("ℹ️ Enable AI to generate an industry-specific ESG excerpt.")

# 6. General Writing Guidelines
display_writing_recommendations()

# 7. Report Export (Universal Format)
st.subheader("📥 Export SDG 12 Report")
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
    
    # PDF Export (General-Purpose Styling)
    if st.button("Generate PDF Report", use_container_width=True):
        if not OPENAI_AVAILABLE:
            st.warning("⚠️ AI is required for PDF report generation.")
            st.stop()
        
        try:
            pdfkit.configuration(wkhtmltopdf=pdfkit.from_url('http://google.com', False))
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                html_content = f"""
                <html>
                <head>
                    <title>{data['Company Name']} SDG 12 Report</title>
                    <style>
                        body {{ font-family: Arial; margin: 30px; }}
                        h1 {{ color: #2E8B57; }}
                        .header {{ border-bottom: 2px solid #2E8B57; padding-bottom: 10px; margin-bottom: 20px; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                        th {{ background-color: #2E8B57; color: white; padding: 8px; }}
                        td {{ border: 1px solid #ddd; padding: 8px; }}
                        .section {{ margin: 30px 0; }}
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
            st.error(f"⚠️ PDF generation failed: {str(e)} (install wkhtmltopdf: https://wkhtmltopdf.org/)")
else:
    st.info("ℹ️ Save data to generate and export reports.")
