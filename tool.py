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

# --- Session State Initialization (No GSK Defaults—Neutral Baseline) ---
if "production_data" not in st.session_state:
    st.session_state["production_data"] = {
        "Company Name": "Enter Company Name",
        "Industry": "Manufacturing",  # Neutral default (adjustable)
        "extracted_pdf_text": "",
        "extracted_sections": {},  # Stores GSK's key chapters (Environment, Procurement)
        # Core Dimensions Data (Neutral Defaults)
        "resource_efficiency": {"renewable_energy_pct": 0, "energy_tech_count": 0, "water_reuse_pct": 0, "ai_score": 0},
        "sustainable_production": {"recycled_material_pct": 0, "waste_intensity_pct": 0, "eco_design_cert": False, "ai_score": 0},
        "chemical_waste": {"hazardous_reduction_pct": 0, "waste_recycling_pct": 0, "chemical_compliance": False, "ai_score": 0},
        "circular_economy": {"takeback_program_pct": 0, "packaging_sustainable_pct": 0, "certified_supplier_pct": 0, "ai_score": 0},
        "sustainable_procurement": {"procurement_criteria_count": 0, "sustainable_budget_pct": 0, "procurement_tracking": False, "ai_score": 0},
        "life_cycle_thinking": {"lca_product_pct": 0, "consumer_comm": False, "lca_improvements": 0, "ai_score": 0},
        "waste_management": {"food_waste_reduction_pct": 0, "segregation_rate_pct": 0, "circular_partnerships": False, "ai_score": 0},
        "tourism_infrastructure": {"sustainable_material_pct": 0, "eco_tourism_pct": 0, "energy_water_efficiency": False, "ai_score": 0},
        # Final Outputs
        "total_score": 0,
        "dimension_scores": {},
        "ai_recommendations": [],
        "mock_esg_excerpt": ""
    }

if "rerun_trigger" not in st.session_state:
    st.session_state["rerun_trigger"] = False

# --- Constants (Scoring Framework + GSK Report Alignment) ---
DIMENSIONS = [
    {
        "id": "resource_efficiency",
        "name": "Resource Efficiency (SDG 12.2)",
        "weight": 0.25,
        "actions": [
            {"name": "renewable_energy_pct", "desc": "Renewable energy percentage (e.g., GSK's 'imported renewable electricity')", "calc": lambda x: 10 if x >=50 else 5 if x >=30 else 0},
            {"name": "energy_tech_count", "desc": "Number of energy-efficient tech categories (e.g., solar, heat recovery)", "calc": lambda x: min(10, x * 5)},
            {"name": "water_reuse_pct", "desc": "Water reuse rate percentage (vs. baseline, e.g., GSK's 24% reduction)", "calc": lambda x: 10 if x >=70 else 5 if x >=40 else 0}
        ],
        "ai_criteria": "Evaluate 1) energy intensity vs. pharma/industry peers, 2) water leak prevention. Return 0-5 (no text).",
        "max_subtotal": 30
    },
    {
        "id": "sustainable_production",
        "name": "Sustainable Production (SDG 12.3)",
        "weight": 0.20,
        "actions": [
            {"name": "recycled_material_pct", "desc": "Recycled material percentage (e.g., GSK's paper packaging)", "calc": lambda x: 10 if x >=40 else 5 if x >=20 else 0},
            {"name": "waste_intensity_pct", "desc": "Waste intensity vs. industry average (%)", "calc": lambda x: 10 if x <=20 else 5 if x <=40 else 0},
            {"name": "eco_design_cert", "desc": "Eco-design certification (e.g., GSK's low-carbon inhalers)", "calc": lambda x: 10 if x else 0}
        ],
        "ai_criteria": "Evaluate 1) material yield improvement, 2) product recyclability. Return 0-5 (no text).",
        "max_subtotal": 30
    },
    {
        "id": "chemical_waste",
        "name": "Chemicals & Waste Management (SDG 12.4)",
        "weight": 0.18,
        "actions": [
            {"name": "hazardous_reduction_pct", "desc": "Hazardous chemical reduction vs. baseline (%)", "calc": lambda x: 7 if x >=50 else 3 if x >=20 else 0},
            {"name": "waste_recycling_pct", "desc": "Production waste recycling rate (e.g., GSK's 53% circular recovery)", "calc": lambda x: 7 if x >=80 else 3 if x >=50 else 0},
            {"name": "chemical_compliance", "desc": "Compliance with REACH/AMR Alliance (e.g., GSK's API limits)", "calc": lambda x: 6 if x else 0}
        ],
        "ai_criteria": "Evaluate 1) chemical spill protocols, 2) hazardous waste treatment. Return 0-3 (no text).",
        "max_subtotal": 20
    },
    {
        "id": "circular_economy",
        "name": "Circular Economy Integration (SDG 12.5)",
        "weight": 0.10,
        "actions": [
            {"name": "takeback_program_pct", "desc": "Product lines with take-back programs (%)", "calc": lambda x: 4 if x >=50 else 2 if x >=20 else 0},
            {"name": "packaging_sustainable_pct", "desc": "Sustainable packaging percentage (e.g., GSK's 86% deforestation-free paper)", "calc": lambda x: 3 if x >=80 else 1 if x >=50 else 0},
            {"name": "certified_supplier_pct", "desc": "Sustainable suppliers (%) (e.g., GSK's 98% palm oil)", "calc": lambda x: 3 if x >=60 else 1 if x >=30 else 0}
        ],
        "ai_criteria": "Evaluate 1) tier 2 supplier practices, 2) product-as-a-service models. Return 0-2 (no text).",
        "max_subtotal": 10
    },
    {
        "id": "sustainable_procurement",
        "name": "Sustainable Procurement (SDG 12.7)",
        "weight": 0.08,
        "actions": [
            {"name": "procurement_criteria_count", "desc": "Sustainability criteria in procurement (e.g., GSK's EcoVadis)", "calc": lambda x: 3 if x >=3 else 1 if x >=1 else 0},
            {"name": "sustainable_budget_pct", "desc": "Procurement budget for sustainable goods (%)", "calc": lambda x: 3 if x >=30 else 1 if x >=10 else 0},
            {"name": "procurement_tracking", "desc": "Tracking sustainable procurement performance", "calc": lambda x: 2 if x else 0}
        ],
        "ai_criteria": "Evaluate 1) supplier diversity, 2) ISO 20400 alignment. Return 0-2 (no text).",
        "max_subtotal": 8
    },
    {
        "id": "life_cycle_thinking",
        "name": "Life-Cycle Thinking (SDG 12.8)",
        "weight": 0.05,
        "actions": [
            {"name": "lca_product_pct", "desc": "Products with Life-Cycle Assessment (%)", "calc": lambda x: 2 if x >=50 else 1 if x >=20 else 0},
            {"name": "consumer_comm", "desc": "Sustainability info for consumers", "calc": lambda x: 2 if x else 0},
            {"name": "lca_improvements", "desc": "Product improvements from LCA (e.g., GSK's inhaler redesign)", "calc": lambda x: 1 if x >=2 else 0.5 if x >=1 else 0}
        ],
        "ai_criteria": "Evaluate 1) scope 3 emissions in LCA, 2) eco-label engagement. Return 0-1 (no text).",
        "max_subtotal": 5
    },
    {
        "id": "waste_management",
        "name": "Waste Generation & Management (SDG 12.5.1)",
        "weight": 0.07,
        "actions": [
            {"name": "food_waste_reduction_pct", "desc": "Food/by-product waste reduction vs. baseline (%)", "calc": lambda x: 3 if x >=40 else 1 if x >=20 else 0},
            {"name": "segregation_rate_pct", "desc": "Waste segregation rate (%)", "calc": lambda x: 3 if x >=90 else 1 if x >=60 else 0},
            {"name": "circular_partnerships", "desc": "Partnerships for circularity (e.g., GSK's WOTR water project)", "calc": lambda x: 1 if x else 0}
        ],
        "ai_criteria": "Evaluate 1) recycled waste purity, 2) by-product revenue. Return 0-2 (no text).",
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
        "ai_criteria": "Evaluate 1) community impact, 2) infrastructure durability. Return 0-2 (no text).",
        "max_subtotal": 7,
        "industries": ["Tourism", "Hospitality", "Infrastructure"]
    }
]

# --- Core AI Functions (Optimized for GSK Report) ---
def get_ai_response(prompt, system_msg="You are an ESG expert specializing in pharmaceutical/industrial sustainability. Be concise."):
    if not OPENAI_AVAILABLE:
        return "❌ AI requires OPENAI_API_KEY in Streamlit Secrets."
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temp for precise data extraction
            timeout=30  # Longer timeout for GSK's large text
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"⚠️ AI error: {str(e)}")
        return None

def extract_pdf_text(uploaded_file):
    """Extract FULL text from GSK 2023 ESG Report + isolate key SDG 12 chapters"""
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf_doc:
            full_text = ""
            section_text = {
                "environment": "",  # GSK's "Environment" chapter (pages 18-25)
                "procurement": "",  # GSK's "Sustainable Procurement" (Ethical Standards)
                "circular_economy": ""  # GSK's circular initiatives (Environment/Access)
            }
            
            # GSK report uses consistent headers—target SDG 12-relevant sections
            current_section = None
            for page_num, page in enumerate(pdf_doc):
                page_text = page.get_text().strip()
                full_text += f"\n--- Page {page_num + 1} ---\n" + page_text
                
                # Identify GSK's key chapters (case-insensitive for consistency)
                if "environment" in page_text.lower() and "climate change" in page_text.lower():
                    current_section = "environment"
                elif "sustainable procurement" in page_text.lower() or "supplier" in page_text.lower() and "ecovadis" in page_text.lower():
                    current_section = "procurement"
                elif "circular economy" in page_text.lower() or "take-back" in page_text.lower() or "packaging" in page_text.lower():
                    current_section = "circular_economy"
                elif "chapter" in page_text.lower() and any(chapter in page_text.lower() for chapter in ["access", "dei", "ethical standards"]):
                    current_section = None  # Exit when non-SDG 12 chapters start
                
                # Append text to current section (avoid duplicate headers)
                if current_section and page_text:
                    section_text[current_section] += page_text + "\n"
            
            # Store for later use (reduce AI processing time)
            st.session_state["production_data"]["extracted_sections"] = section_text
            return full_text[:50000]  # Cap at 50k chars (covers GSK's key SDG 12 sections)
    except Exception as e:
        st.error(f"⚠️ GSK PDF extraction failed: {str(e)} (report may be password-protected)")
        return ""

def ai_extract_esg_data():
    """Extract SDG 12 data from GSK's report—replace None with 0/False"""
    data = st.session_state["production_data"]
    section_text = data["extracted_sections"]
    
    # Prioritize GSK's "Environment" section (richest SDG 12 data)
    target_text = ""
    if section_text["environment"]:
        target_text += f"[Environment Section]\n{section_text['environment'][:15000]}\n"
    if section_text["procurement"]:
        target_text += f"[Procurement Section]\n{section_text['procurement'][:10000]}\n"
    if section_text["circular_economy"]:
        target_text += f"[Circular Economy Section]\n{section_text['circular_economy'][:10000]}\n"
    
    if not target_text:
        target_text = data["extracted_pdf_text"][:30000]  # Fallback to full text if sections not found
    
    # GSK-specific terminology map (critical for accurate extraction)
    gsk_terminology = """
    GSK-Specific Terms to Map to SDG 12 Metrics:
    - "Imported renewable electricity %": renewable_energy_pct
    - "Water use reduction vs 2020 baseline": water_reuse_pct
    - "Low-carbon Ventolin": eco_design_cert (true) + lca_improvements (count)
    - "Waste recovered via circular routes": waste_recycling_pct
    - "AMR Alliance wastewater compliance": chemical_compliance (true)
    - "Deforestation-free paper packaging": packaging_sustainable_pct
    - "Sustainably certified palm oil": certified_supplier_pct
    - "EcoVadis supplier score": sustainable_procurement → procurement_criteria_count
    - "LCA for products": lca_product_pct
    """
    
def ai_extract_esg_data():
    # ... (keep existing code) ...

    prompt = f"""Extract SDG 12 production metrics from this GSK ESG Report text. 
    CRITICAL RULES:
    1. Return ONLY NUMBERS (no percentages, e.g., write 83 not "83%")
    2. Return 0 for missing data (NEVER use null)
    3. For booleans, return true/false ONLY
    
    {gsk_terminology}
    
    Return ONLY valid JSON (no extra text):
    {{
        "company_name": "GSK PLC (fixed)",
        "industry": "Pharmaceuticals (fixed)",
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
            "food_waste_reduction_pct": "Number (0-100)",
            "segregation_rate_pct": "Number (0-100)",
            "circular_partnerships": "true/false"
        }}
    }}
    
    GSK Report Text: {target_text}
    """
    
    response = get_ai_response(prompt, system_msg="You extract ESG data from pharmaceutical reports. Return ONLY valid JSON—no explanations.")
    if not response:
        return {}
    
    try:
        extracted_data = json.loads(response)
        
        # --- GSK-Specific Data Cleaning ---
        # 1. Remove percentage symbols and convert to float
        for dim in extracted_data:
            if isinstance(extracted_data[dim], dict):
                for metric in extracted_data[dim]:
                    value = extracted_data[dim][metric]
                    # Fix percentages with "%" (e.g., "83%" → 83)
                    if isinstance(value, str) and "%" in value:
                        extracted_data[dim][metric] = float(value.replace("%", "").strip())
                    # Fix nulls (replace with 0 or false)
                    if value is None:
                        if metric in ["renewable_energy_pct", "water_reuse_pct"]:
                            extracted_data[dim][metric] = 0  # Numeric defaults
                        else:
                            extracted_data[dim][metric] = False  # Boolean defaults
        
        # 2. Force GSK-specific values
        extracted_data["company_name"] = "GSK PLC"
        extracted_data["industry"] = "Pharmaceuticals"
        
        return extracted_data
    
    except json.JSONDecodeError as e:
        st.sidebar.error(f"⚠️ GSK Data Parse Error: {str(e)}. Raw response: {response[:200]}")
        return {}
def ai_evaluate_unlisted_criteria(dimension):
    """Evaluate AI-only criteria (uses GSK's section text for context)"""
    data = st.session_state["production_data"]
    dim_data = data[dimension["id"]]
    section_text = data["extracted_sections"].get(dimension["id"], "")
    
    prompt = f"""Evaluate {dimension['name']} for GSK:
    - Company Data: {dim_data}
    - GSK Report Context: {section_text[:2000]}  # Use relevant section
    - Criteria: {dimension['ai_criteria']}
    Return ONLY the numeric score (0-5, no text)."""
    
    response = get_ai_response(prompt, system_msg="You analyze pharmaceutical ESG performance. Return ONLY a number.")
    if not response:
        return 0
    try:
        return max(0, min(5, float(response.strip())))
    except (ValueError, TypeError):
        st.warning(f"⚠️ Invalid AI score for {dimension['name']}. Defaulting to 0.")
        return 0

def ai_generate_recommendations():
    """Generate GSK-aligned recommendations (focus on pharma-specific actions)"""
    data = st.session_state["production_data"]
    scores = data["dimension_scores"]
    company_industry = data["Industry"]
    
    # Identify low-performing dimensions (GSK's SDG 12 focus areas)
    low_dimensions = []
    for dim in DIMENSIONS:
        if dim["id"] == "tourism_infrastructure" and company_industry not in dim["industries"]:
            continue
        dim_score = scores.get(dim["id"], {}).get("weighted_score", 0)
        max_threshold = (dim["max_subtotal"] * dim["weight"]) * 0.5
        if dim_score < max_threshold:
            low_dimensions.append(dim["name"])
    
    if not low_dimensions:
        low_dimensions = ["Resource Efficiency (SDG 12.2)", "Circular Economy (SDG 12.5)"]
    
    prompt = f"""Generate 3 pharma-specific recommendations for {data['Company Name']} (Industry: {company_industry}).
    Focus on low areas: {low_dimensions}.
    Each recommendation must:
    1. Link to SDG 12 (e.g., SDG 12.3 for product design).
    2. Include measurable pharma actions (e.g., "Expand low-carbon inhaler production").
    3. Explain impact (e.g., "Reduces Scope 3 emissions by 40%").
    Number recommendations (1., 2., 3.)—no bullets."""
    
    response = get_ai_response(prompt, system_msg="You are a pharmaceutical sustainability consultant. Be specific.")
    if not response:
        return [
            "1. Expand renewable electricity to 100% by 2026 (SDG 12.2) – Reduces Scope 2 emissions by 50% (aligns with GSK's targets).",
            "2. Increase sustainable packaging to 95% by 2027 (SDG 12.5) – Cuts plastic waste by 35% in pharmaceutical supply chains.",
            "3. Require 100% of high-risk suppliers to have science-based targets by 2025 (SDG 12.7) – Lowers supply chain emissions by 25%."
        ]
    return [line.strip() for line in response.split("\n") if line.strip() and line.strip()[0].isdigit()]

def ai_generate_mock_esg():
    """Generate GSK-style ESG excerpt (pharma-specific, project-focused)"""
    data = st.session_state["production_data"]
    scores = data["dimension_scores"]
    
    # Use GSK's section text for realistic context
    env_text = data["extracted_sections"].get("environment", "")[:500]
    high_dims = [dim["name"] for dim in DIMENSIONS if scores.get(dim["id"], {}).get("weighted_score", 0) > (dim["max_subtotal"] * dim["weight"]) * 0.7]
    low_dims = [dim["name"] for dim in DIMENSIONS if scores.get(dim["id"], {}).get("weighted_score", 0) < (dim["max_subtotal"] * dim["weight"]) * 0.5]
    
    prompt = f"""Write a 300-500 word GSK-style ESG excerpt for {data['Company Name']} (Industry: {data['Industry']}).
    Include:
    1. A specific project (e.g., low-carbon product, water reuse).
    2. GSK-like metrics: renewable energy ({data['resource_efficiency']['renewable_energy_pct']}%), waste recycling ({data['chemical_waste']['waste_recycling_pct']}%), sustainable packaging ({data['circular_economy']['packaging_sustainable_pct']}%).
    3. Strengths: {high_dims if high_dims else ['Sustainable procurement']}.
    4. Improvements: {low_dims if low_dims else ['Resource efficiency']}.
    5. Future goal tied to SDG 12 (e.g., "2030 net-zero production").
    Use formal pharma ESG tone (avoid jargon). Do NOT introduce the company.
    
    GSK Context Reference: {env_text}"""
    
    fallback = f"""In 2024, {data['Company Name']} advanced its low-carbon production initiative, building on progress in sustainable manufacturing. The project, aligned with SDG 12.3 (sustainable production patterns), focused on optimizing solvent recovery and scaling recycled packaging for pharmaceutical products. 

    During the year, the initiative reduced production waste intensity by {data['sustainable_production']['waste_intensity_pct']}% vs. the industry average, while 86% of paper packaging was sourced from deforestation-free suppliers—exceeding the 2024 target of 80%. Renewable electricity accounted for {data['resource_efficiency']['renewable_energy_pct']}% of total consumption, supported by on-site solar installations at three manufacturing sites. 

    Key strengths included chemical waste management (100% compliance with AMR Alliance discharge limits) and sustainable procurement (89% of high-risk suppliers met EcoVadis thresholds). Areas for improvement included water reuse (currently {data['resource_efficiency']['water_reuse_pct']}%, below the 40% target) and product take-back programs (covering just 20% of product lines). 

    Looking ahead, the company aims to achieve 90% recycled packaging by 2026 (SDG 12.5) and expand water reuse systems to all high-stress sites—aligning with its broader 2030 commitment to circular production."""
    
    return get_ai_response(prompt, system_msg="You write pharmaceutical ESG excerpts. Mimic GSK's formal, data-heavy style.") or fallback

# --- Helper Functions (GSK Compatibility) ---
def calculate_dimension_scores():
    """Calculate scores (uses GSK's section text for AI evaluation)"""
    data = st.session_state["production_data"]
    dimension_scores = {}
    total_score = 0
    
    for dim in DIMENSIONS:
        if dim["id"] == "tourism_infrastructure" and data["Industry"] not in dim["industries"]:
            continue
        
        # Calculate action-based subtotal
        dim_data = data[dim["id"]]
        action_subtotal = sum([action["calc"](dim_data[action["name"]]) for action in dim["actions"]])
        
        # Add AI score (uses GSK's section text for context)
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
    """Generate GSK-style report content"""
    data = st.session_state["production_data"]
    scores = data["dimension_scores"]
    
    content = f"SDG 12 Production Responsibility Report\n"
    content += f"Company: {data['Company Name']}\n"
    content += f"Industry: {data['Industry']}\n"
    content += f"Total Score: {data['total_score']}/100\n"
    content += "="*50 + "\n\n"
    
    # Dimension Breakdown (GSK-like structure)
    content += "1. SDG 12 Dimension Breakdown\n"
    for dim_id, dim_data in scores.items():
        content += f"- {dim_data['name']}: {dim_data['weighted_score']}/{dim_data['max_weighted']}\n"
    
    # Key Metrics (Pharma Focus)
    content += "\n2. Key Pharmaceutical Production Metrics\n"
    core_dims = ["resource_efficiency", "sustainable_production", "chemical_waste"]
    for dim_id in core_dims:
        if dim_id not in data:
            continue
        dim_data = data[dim_id]
        dim_name = next(d["name"] for d in DIMENSIONS if d["id"] == dim_id)
        content += f"- {dim_name}:\n"
        for action in [a for a in DIMENSIONS[[d["id"] for d in DIMENSIONS].index(dim_id)]["actions"]]:
            content += f"  - {action['desc']}: {dim_data[action['name']]}\n"
    
    # Recommendations + Mock Excerpt
    content += "\n3. Improvement Recommendations\n"
    recommendations = ai_generate_recommendations() if OPENAI_AVAILABLE else data["ai_recommendations"]
    for i, rec in enumerate(recommendations, 1):
        content += f"{i}. {rec}\n"
    
    content += "\n4. Mock ESG Excerpt (GSK Style)\n"
    mock_excerpt = ai_generate_mock_esg() if OPENAI_AVAILABLE else data["mock_esg_excerpt"]
    content += mock_excerpt + "\n"
    
    return content, recommendations, mock_excerpt

# --- ESG Report Writing Guidelines (GSK Example) ---
def display_writing_recommendations():
    st.subheader("✏️ ESG Report Writing Guidelines (GSK Example Reference)")
    st.markdown("""
    Use GSK’s 2023 ESG Report as a template for SDG 12 sections:
    
    ### 1. Focus on Pharma-Specific Projects
    - **GSK Example**: "Low-carbon Ventolin inhaler (90% emissions reduction) – Phase III trials in 2024"  
    - **Your Report**: Highlight product redesigns, API waste reduction, or sustainable packaging for pharmaceuticals.

    ### 2. Link Metrics to Baselines/Targes
    - **GSK Example**: "Water use reduced by 24% vs. 2020 baseline (11% in high-stress regions)"  
    - **Your Report**: Include year-over-year changes (e.g., "Waste recycling up from 45% in 2022 to 53% in 2023").

    ### 3. Reference Industry Standards
    - **GSK Example**: "89% of suppliers meet EcoVadis minimum score; 100% compliant with AMR Alliance limits"  
    - **Your Report**: Cite REACH, ISO 20400, or pharma-specific standards (e.g., "Compliant with FDA’s sustainable manufacturing guidelines").

    ### 4. Be Transparent About Gaps
    - **GSK Example**: "Supplier compliance down from 94% to 87% (expanded supplier scope)"  
    - **Your Report**: Explain setbacks (e.g., "Water reuse below target due to delayed treatment plant installation").
    """)

# --- Sidebar UI (Optimized for GSK PDF) ---
st.sidebar.header("📊 Data Input (GSK ESG Report Compatible)")

# 1. GSK PDF Upload
st.sidebar.subheader("1. Upload GSK 2023 ESG Report (esg-performance-report-2023.pdf)")
uploaded_pdf = st.sidebar.file_uploader("Upload PDF", type="pdf", help="Upload GSK's 2023 ESG Report (120+ pages)")
if uploaded_pdf:
    with st.spinner("🔍 Extracting GSK's SDG 12 sections (Environment, Procurement)..."):
        pdf_text = extract_pdf_text(uploaded_pdf)
        data = st.session_state["production_data"]
        data["extracted_pdf_text"] = pdf_text
        st.session_state["production_data"] = data
        
        # Show GSK's extracted sections (for verification)
        with st.sidebar.expander("View Extracted GSK Sections", expanded=False):
            section_text = data["extracted_sections"]
            if section_text["environment"]:
                st.text_area("Environment Section (First 2000 Chars)", section_text["environment"][:2000], height=150, disabled=True)
            if section_text["procurement"]:
                st.text_area("Procurement Section (First 2000 Chars)", section_text["procurement"][:2000], height=150, disabled=True)
        
        # Extract GSK data (if AI is available)
        if OPENAI_AVAILABLE:
            with st.spinner("🤖 Extracting SDG 12 data from GSK's report..."):
                esg_data = ai_extract_esg_data()
                if esg_data:
                    # Update only non-null fields (preserve manual inputs)
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
                        st.sidebar.success("✅ Populated GSK's SDG 12 data! Review below.")
                        st.session_state["production_data"] = data

# 2. Company Info (Neutral—No GSK Default)
st.sidebar.subheader("2. Company Information")
company_name = st.sidebar.text_input(
    "Company Name",
    st.session_state["production_data"]["Company Name"],
    key="company_name"
)
industry = st.sidebar.selectbox(
    "Industry",
    ["Pharmaceuticals", "Manufacturing", "Tourism", "Hospitality", "Infrastructure", "Other"],
    index=["Pharmaceuticals", "Manufacturing", "Tourism", "Hospitality", "Infrastructure", "Other"].index(
        st.session_state["production_data"]["Industry"]
    ),
    key="industry"
)

# 3. Dimension Inputs (Pharma-Aligned Labels + GSK Data None Handling)
st.sidebar.subheader("3. SDG 12 Production Data")
data = st.session_state["production_data"]

for dim in DIMENSIONS:
    # Skip tourism dimension for non-relevant industries (GSK = Pharmaceuticals)
    if dim["id"] == "tourism_infrastructure" and industry not in dim["industries"]:
        continue
    
    st.sidebar.markdown(f"**{dim['name']}**")
    dim_data = data[dim["id"]]
    
    for action in dim["actions"]:
        # Get current value—default to 0 if None (critical for GSK report gaps)
        current_value = dim_data[action["name"]]
        initial_value = current_value if current_value is not None else 0
        
        if "pct" in action["name"]:  # Percentage sliders (GSK uses % for most metrics)
            value = st.sidebar.slider(
                f"{action['desc']} (%)",
                min_value=0,
                max_value=100,
                value=initial_value,  # No more None—safe for slider
                key=f"{dim['id']}_{action['name']}",
                help=f"GSK reference: See {dim['name']} in Environment chapter (pages 18-25)"
            )
        elif "count" in action["name"] or "improvements" in action["name"]:  # Numeric counts
            value = st.sidebar.number_input(
                f"{action['desc']}",
                min_value=0,
                value=int(initial_value),  # Ensure integer for counts (e.g., GSK's energy tech count)
                key=f"{dim['id']}_{action['name']}",
                help=f"GSK reference: Count of initiatives (e.g., solar/wind projects)"
            )
        else:  # Boolean checkboxes (GSK uses "compliance" or "certification" for these)
            value = st.sidebar.checkbox(
                f"{action['desc']}",
                value=initial_value if isinstance(initial_value, bool) else False,  # Fix boolean None
                key=f"{dim['id']}_{action['name']}",
                help=f"GSK reference: Check if GSK mentions compliance/certification"
            )
        
        # Update session state with valid value (no None)
        dim_data[action["name"]] = value
    
    # Save updated dimension data back to session state
    data[dim["id"]] = dim_data
    st.sidebar.markdown("---")

# 4. Save Data
if st.sidebar.button("💾 Save Data & Calculate Scores", use_container_width=True):
    data["Company Name"] = company_name
    data["Industry"] = industry
    calculate_dimension_scores()
    st.sidebar.success("✅ Data saved! Check main dashboard for results.")
    st.session_state["rerun_trigger"] = True

# --- Rerun Trigger ---
if st.session_state["rerun_trigger"]:
    st.session_state["rerun_trigger"] = False
    st.rerun()

# --- Main Dashboard (GSK-Style Results) ---
data = st.session_state["production_data"]
scores = data.get("dimension_scores", {})
total_score = data.get("total_score", 0)

st.title("🌱 SDG 12 Production Responsibility Evaluator (GSK Report Compatible)")

# 1. Overview
st.subheader("📋 Evaluation Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Company Name", data["Company Name"])
with col2:
    st.metric("Industry", data["Industry"])
with col3:
    st.metric("Total SDG 12 Score", f"{total_score}/100")

# 2. Score Breakdown Chart (GSK Colors)
st.subheader("📊 Dimension Score Breakdown")
if scores:
    dim_names = [v["name"].split(" (")[0] for v in scores.values()]
    dim_scores = [v["weighted_score"] for v in scores.values()]
    dim_max = [v["max_weighted"] for v in scores.values()]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(dim_names))
    width = 0.35
    
    # GSK's brand colors (teal for achieved, light gray for max)
    bars1 = ax.bar(x - width/2, dim_scores, width, label="Achieved Score", color="#00857C")
    bars2 = ax.bar(x + width/2, dim_max, width, label="Max Possible Score", color="#E0E0E0")
    
    ax.set_xlabel("SDG 12 Dimensions")
    ax.set_ylabel("Score")
    ax.set_title("SDG 12 Dimension Comparison (GSK-Aligned)")
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
    st.info("ℹ️ Upload GSK's ESG report and save data to generate scores.")

# 3. Detailed Score Table
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

# 4. AI Recommendations (Pharma-Specific)
st.subheader("💡 Pharma-Specific Recommendations")
if OPENAI_AVAILABLE and scores:
    if st.button("Generate GSK-Style Recommendations", use_container_width=True):
        with st.spinner("🤖 Generating recommendations..."):
            recommendations = ai_generate_recommendations()
            data["ai_recommendations"] = recommendations
            st.session_state["production_data"] = data
    
    if data["ai_recommendations"]:
        for i, rec in enumerate(data["ai_recommendations"], 1):
            st.write(f"{i}. {rec}")
else:
    st.info("ℹ️ Upload GSK's report and enable AI to generate recommendations.")

# 5. Mock ESG Excerpt (GSK Style)
st.subheader("📄 Mock ESG Excerpt (GSK Template)")
if OPENAI_AVAILABLE:
    if st.button("Generate GSK-Style Excerpt", use_container_width=True):
        with st.spinner("🤖 Writing ESG excerpt..."):
            mock_excerpt = ai_generate_mock_esg()
            data["mock_esg_excerpt"] = mock_excerpt
            st.session_state["production_data"] = data
    
    if data["mock_esg_excerpt"]:
        st.write(data["mock_esg_excerpt"])
else:
    st.info("ℹ️ Enable AI to generate a GSK-style ESG excerpt.")

# 6. Writing Guidelines (GSK Reference)
display_writing_recommendations()

# 7. Report Export (GSK Format)
st.subheader("📥 Export GSK-Style Report")
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
    
    # PDF Export (GSK-Style Formatting)
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
                        h1 {{ color: #00857C; }}
                        .header {{ border-bottom: 2px solid #00857C; padding-bottom: 10px; margin-bottom: 20px; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                        th {{ background-color: #00857C; color: white; padding: 8px; }}
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
    st.info("ℹ️ Save data to export reports.")
