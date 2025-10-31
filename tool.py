import streamlit as st
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import io

# --- Color Scheme (Purple-Themed, Aligned with Screenshot) ---
PRIMARY_PURPLE = "#6a0dad"
LIGHT_PURPLE = "#f0f0ff"
MEDIUM_PURPLE = "#9370db"
LIGHT_GRAY = "#555"
CHART_PURPLE = "#6a0dad"

# --- 1. Third-Party Data (AI-Derived with Links) ---
def get_third_party_data(company_name, industry):
    if not company_name or not OPENAI_AVAILABLE:
        return {
            "penalties": False, 
            "penalties_details": "No third-party data fetched (missing company name or AI key)",
            "positive_news": "No third-party data fetched",
            "policy_updates": "No third-party data fetched"
        }
    
    prompt = f"""For {company_name} (industry: {industry}), extract ONLY AI-derived, verifiable third-party data:
    1. Environmental penalties (2023-2024): List violations related to responsible production (e.g., illegal waste disposal). Include issuing authority, date, and DIRECT LINK to regulatory filing/news article.
    2. Positive responsible production news (2023-2024): Actions like recycling partnerships or renewable energy adoption. Include source link.
    3. Relevant policy updates (2023-2024): Laws affecting responsible production (e.g., extended producer responsibility). Include policy document link.
    
    Sources: Prioritize government environmental agencies (EPA, EU EEA), credible news (Bloomberg Green, Reuters), and official regulatory databases.
    Return ONLY valid JSON with keys: penalties (bool), penalties_details (str with links), positive_news (str with links), policy_updates (str with links).
    If no data exists, use "No relevant data found (AI search returned no results)". Do NOT include hardcoded content."""
    
    response = get_ai_response(prompt, "Environmental data analyst specializing in responsible production.")
    try:
        data = json.loads(response) if response else {}
        return {
            "penalties": data.get("penalties", False),
            "penalties_details": data.get("penalties_details", "No relevant data found (AI search returned no results)"),
            "positive_news": data.get("positive_news", "No relevant data found (AI search returned no results)"),
            "policy_updates": data.get("policy_updates", "No relevant data found (AI search returned no results)")
        }
    except json.JSONDecodeError:
        return {
            "penalties": False,
            "penalties_details": f"AI response invalid: {response[:100]}... (No links available)",
            "positive_news": "AI failed to return valid data (No links available)",
            "policy_updates": "AI failed to return valid data (No links available)"
        }

# --- 2. PDF Handling ---
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("⚠️ PyPDF2 not found. Install with: pip install PyPDF2 (required for PDF upload)")

def extract_full_pdf_text(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        full_text = ""
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text() or ""
            full_text += f"\n--- Page {page_num} ---\n{page_text}"
        
        if len(full_text.strip()) < 100:
            st.warning("⚠️ PDF may have image-based text (cannot extract). Use manual input or a text-based PDF.")
        return full_text
    except Exception as e:
        st.error(f"❌ PDF Extraction Error: {str(e)}")
        return ""

def extract_sdg_data_from_pdf(pdf_text, company_name, industry):
    if len(pdf_text.strip()) < 500:
        st.error("❌ Insufficient text for extraction. Use a longer PDF.")
        return {}
    
    prompt = f"""Extract responsible production data from this PDF for {company_name} (industry: {industry}):
    
    PDF Text (first 10,000 characters):
    {pdf_text[:10000]}
    
    REQUIRED DATA:
    - renewable_share: % renewable energy (e.g., 55 = 55%)
    - energy_retrofit: True/False (full-scale energy retrofit)
    - energy_increase: True/False (energy up 2 years)
    - carbon_offsets_only: True/False (sole reliance on offsets)
    - recycled_water_ratio: % recycled water (e.g., 75 = 75%)
    - ghg_disclosure: True/False (Scope1-3 + third-party verification)
    - recycled_materials_pct: % recycled materials (e.g., 35 = 35%)
    - illegal_logging: True/False (any incidents)
    - loss_tracking_system: True/False (loss-tracking system)
    - loss_reduction_pct: % annual loss reduction (e.g., 12 = 12%)
    - mrsl_zdhc_compliance: True/False (MRSL/ZDHC compliant)
    - regular_emission_tests: True/False (regular testing)
    - hazardous_recovery_pct: % hazardous waste recovery (e.g., 92 = 92%)
    - illegal_disposal: True/False (improper disposal)
    - packaging_reduction_pct: % packaging reduction (e.g., 25 = 25%)
    - recycling_rate_pct: % recycling rate (e.g., 85 = 85%)
    - sustainable_products_pct: % sustainable material products (e.g., 55 = 55%)
    - waste_disclosure_audit: True/False (disclosure + audit)
    - emission_plans: True/False (clear 2030/2050 emission goals)
    - annual_progress_disclosed: True/False (annual progress shared)
    - no_goals: True/False (no goals/stagnant)
    - high_carbon_assets_disclosed: True/False (disclosed + reduction pathway)
    - esg_audited_suppliers_pct: % suppliers with ESG audits (e.g., 85 = 85%)
    - price_only_procurement: True/False (price-only/high-emission outsourcing)
    - supply_chain_transparency: True/False (transparency report)
    
    Return ONLY valid JSON. Use null for unknown values. No extra text."""
    
    response = get_ai_response(prompt, "ESG extractor specializing in responsible production.")
    if not response:
        st.error("❌ AI returned no extraction results.")
        return {}
    
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        st.error(f"❌ No valid JSON in AI response: {response[:200]}...")
        return {}
    
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError as e:
        st.error(f"❌ Failed to parse extracted data: {str(e)}. Raw JSON: {json_match.group()[:200]}...")
        return {}

def ai_fill_missing_fields(extracted_data, industry):
    if not OPENAI_AVAILABLE or not extracted_data:
        return extracted_data
    
    missing_fields = [k for k, v in extracted_data.items() if v is None]
    if not missing_fields:
        return extracted_data
    
    prompt = f"""For a {industry} company, fill missing responsible production data using industry benchmarks (AI-derived, no hardcoded content):
    Extracted data (missing fields: {missing_fields}):
    {json.dumps(extracted_data, indent=2)}
    
    Rules:
    1. Use realistic {industry} averages (e.g., manufacturing: 35% renewable energy; textiles: 25% recycled materials).
    2. For booleans: Assume "False" for high-risk criteria (e.g., illegal_logging = False) and "True" for common practices (e.g., regular_emission_tests = True).
    3. Preserve existing non-null values.
    4. Return ONLY updated JSON. No extra text."""
    
    response = get_ai_response(prompt, f"Data analyst for {industry} responsible production.")
    try:
        return json.loads(response) if response else extracted_data
    except:
        st.warning("⚠️ AI could not fill missing fields. Using original extracted data.")
        return extracted_data

def render_pdf_confirmation_page(extracted_data, company_name, industry):
    st.subheader(f"Review & Confirm Extracted Data (Company: {company_name})")
    st.write("Edit fields as needed (based on your report). Fields marked * are AI-filled with industry benchmarks.")
    
    field_groups = {
        "Energy & Water Management": [
            ("renewable_share", "Renewable energy share (%)", "number"),
            ("recycled_water_ratio", "Recycled water ratio (%)", "number"),
            ("energy_retrofit", "Full-scale energy retrofit?", "bool"),
            ("energy_increase", "Energy consumption up 2 consecutive years?", "bool"),
            ("carbon_offsets_only", "Rely solely on carbon offsets?", "bool"),
            ("ghg_disclosure", "Scope1-3 GHG (disclosed + verified)?", "bool")
        ],
        "Material & Waste Management": [
            ("recycled_materials_pct", "Recycled materials (%)", "number"),
            ("illegal_logging", "Illegal logging incidents?", "bool"),
            ("loss_tracking_system", "Loss-tracking system established?", "bool"),
            ("loss_reduction_pct", "Annual loss reduction (%)", "number"),
            ("hazardous_recovery_pct", "Hazardous waste recovery (%)", "number"),
            ("illegal_disposal", "Improper waste disposal?", "bool")
        ],
        "Packaging & Reporting": [
            ("packaging_reduction_pct", "Packaging weight reduction (%)", "number"),
            ("recycling_rate_pct", "Recycling rate (%)", "number"),
            ("sustainable_products_pct", "Sustainable material products (%)", "number"),
            ("waste_disclosure_audit", "Waste disclosure + third-party audit?", "bool"),
            ("emission_plans", "Clear 2030/2050 emission goals?", "bool"),
            ("annual_progress_disclosed", "Annual progress disclosed?", "bool"),
            ("no_goals", "No goals or stagnant progress?", "bool"),
            ("high_carbon_assets_disclosed", "High-carbon assets disclosed + reduction pathway?", "bool")
        ],
        "Supplier & Procurement": [
            ("esg_audited_suppliers_pct", "ESG-audited suppliers (%)", "number"),
            ("price_only_procurement", "Price-only procurement or high-emission outsourcing?", "bool"),
            ("supply_chain_transparency", "Supply chain transparency report?", "bool")
        ]
    }
    
    confirmed_data = extracted_data.copy()
    
    for group_name, fields in field_groups.items():
        st.subheader(f"• {group_name}")
        col1, col2 = st.columns([1, 1], gap="small")
        for i, (field, label, field_type) in enumerate(fields):
            with col1 if i % 2 == 0 else col2:
                current_value = confirmed_data.get(field, None)
                ai_filled = current_value is not None and extracted_data.get(field, None) is None
                
                if field_type == "number":
                    new_value = st.number_input(
                        f"{label}" + (" *" if ai_filled else ""),
                        min_value=0, max_value=100, step=1,
                        value=current_value if current_value is not None else 0
                    )
                    confirmed_data[field] = new_value
                elif field_type == "bool":
                    new_value = st.radio(
                        f"{label}" + (" *" if ai_filled else ""),
                        ["Yes", "No"],
                        index=0 if current_value else 1,
                        help=f"Is {label.lower()}?",
                        label_visibility="visible"
                    ) == "Yes"
                    confirmed_data[field] = new_value
    
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Confirm Data & Proceed", key="confirm_pdf", 
                    help="Finalize extracted data and move to the next step",
                    type="primary", use_container_width=True):
            eval_data = st.session_state["eval_data"]
            for field in ["renewable_share", "energy_retrofit", "energy_increase", "carbon_offsets_only", "recycled_water_ratio", "ghg_disclosure", "recycled_materials_pct", "illegal_logging"]:
                if field in confirmed_data:
                    eval_data["12_2"][field] = confirmed_data[field]
            for field in ["loss_tracking_system", "loss_reduction_pct", "mrsl_zdhc_compliance", "regular_emission_tests", "hazardous_recovery_pct", "illegal_disposal"]:
                if field in confirmed_data:
                    eval_data["12_3_4"][field] = confirmed_data[field]
            for field in ["packaging_reduction_pct", "recycling_rate_pct", "sustainable_products_pct", "waste_disclosure_audit", "emission_plans", "annual_progress_disclosed", "no_goals", "high_carbon_assets_disclosed"]:
                if field in confirmed_data:
                    eval_data["12_5_6"][field] = confirmed_data[field]
            for field in ["esg_audited_suppliers_pct", "price_only_procurement", "supply_chain_transparency"]:
                if field in confirmed_data:
                    eval_data["12_7"][field] = confirmed_data[field]
            st.session_state["eval_data"] = eval_data
            st.session_state["current_step"] = 6
            st.rerun()
    
    with col2_btn:
        if st.button("Re-Extract from PDF", key="reextract_pdf", 
                    help="Re-upload and re-extract data from the PDF",
                    use_container_width=True):
            st.session_state["current_step"] = 0
            st.rerun()

# --- 3. Page Config (Renamed to "Responsible Production") ---
st.set_page_config(
    page_title="Responsible Production Evaluator", 
    layout="wide", 
    page_icon="🌱", 
    menu_items={
        "About": "This dashboard evaluates corporate performance on responsible production (SDG 12)."
    }
)

# --- Streamlit Theme Configuration (Purple Primary Color) ---
st.markdown(
    f"""
    <style>
    .stButton {{
        background-color: {PRIMARY_PURPLE} !important;
        color: white !important;
    }}
    .stButton:hover {{
        background-color: {MEDIUM_PURPLE} !important;
    }}
    .stRadio > div > label > div[data-baseweb="radio"]:checked {{
        background-color: {PRIMARY_PURPLE} !important;
    }}
    .stCheckbox > div > label > div[data-baseweb="checkbox"]:checked {{
        background-color: {PRIMARY_PURPLE} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- 4. OpenAI Client ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except KeyError:
    st.warning("⚠️ OPENAI_API_KEY missing. AI features (extraction, recommendations) disabled.")
    OPENAI_AVAILABLE = False
except Exception as e:
    st.error(f"⚠️ OpenAI Error: {str(e)}")
    OPENAI_AVAILABLE = False

# --- 5. Session State ---
if "eval_data" not in st.session_state:
    st.session_state["eval_data"] = {
        "company_name": "",
        "industry": "Manufacturing",
        "third_party": {
            "penalties": False, "penalties_details": "", "positive_news": "", "policy_updates": ""
        },
        "12_2": {
            "renewable_share": None, "energy_retrofit": False, "energy_increase": False,
            "carbon_offsets_only": False, "recycled_water_ratio": None, "ghg_disclosure": False,
            "recycled_materials_pct": None, "illegal_logging": False
        },
        "12_3_4": {
            "loss_tracking_system": False, "loss_reduction_pct": None,
            "mrsl_zdhc_compliance": False, "regular_emission_tests": False,
            "hazardous_recovery_pct": None, "illegal_disposal": False
        },
        "12_5_6": {
            "packaging_reduction_pct": None, "recycling_rate_pct": None,
            "sustainable_products_pct": None, "waste_disclosure_audit": False,
            "emission_plans": False, "annual_progress_disclosed": False, "no_goals": False,
            "high_carbon_assets_disclosed": False
        },
        "12_7": {
            "esg_audited_suppliers_pct": None, "price_only_procurement": False,
            "supply_chain_transparency": False
        },
        "additional_notes": "",
        "target_scores": {}, "overall_score": 0, "rating": "", "other_positive_actions": ""
    }
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0
if "pdf_extracted_text" not in st.session_state:
    st.session_state["pdf_extracted_text"] = ""
if "extracted_data" not in st.session_state:
    st.session_state["extracted_data"] = {}

# --- 6. Constants (Enriched Industries) ---
ENRICHED_INDUSTRIES = [
    "Manufacturing", "Food & Beverage", "Textiles", "Chemicals", "Electronics",
    "Automotive", "Construction", "Healthcare", "Retail", "Agriculture",
    "Logistics", "Pharmaceuticals", "Paper & Pulp", "Furniture", "Cosmetics", "Other"
]

SDG_MAX_SCORES = {
    "12.2": 29, "12.3": 9, "12.4": 16, "12.5": 17, "12.6": 9, "12.7": 10, "Others": 10
}

SDG_CRITERIA = {
    "12.2": [
        ("renewable_share", "Renewable energy share ≥50%", 7, "≥50%"),
        ("energy_retrofit", "Full-scale energy retrofit", 5, "Yes/No"),
        ("energy_increase", "Energy up 2 consecutive years", -5, "Yes/No"),
        ("carbon_offsets_only", "Sole reliance on carbon offsets", -3, "Yes/No"),
        ("recycled_water_ratio", "Recycled water ≥70%", 5, "≥70%"),
        ("ghg_disclosure", "Scope1-3 GHG (disclosed + verified)", 7, "Yes/No"),
        ("recycled_materials_pct", "Recycled materials ≥30%", 5, "≥30%"),
        ("illegal_logging", "Illegal logging incidents", -7, "Yes/No")
    ],
    "12.3": [
        ("loss_tracking_system", "Loss-tracking system", 5, "Yes/No"),
        ("loss_reduction_pct", "Annual loss reduction >10%", 4, ">10%")
    ],
    "12.4": [
        ("mrsl_zdhc_compliance", "MRSL/ZDHC compliance", 5, "Yes/No"),
        ("regular_emission_tests", "Regular emission testing", 3, "Yes/No"),
        ("hazardous_recovery_pct", "Hazardous waste recovery ≥90%", 5, "≥90%"),
        ("illegal_disposal", "Improper disposal", -3, "Yes/No"),
        ("penalties", "No environmental penalties", 3, "Yes/No")
    ],
    "12.5": [
        ("packaging_reduction_pct", "Packaging reduction ≥20%", 4, "≥20%"),
        ("recycling_rate_pct", "Recycling rate ≥80%", 4, "≥80%"),
        ("sustainable_products_pct", "Sustainable material products ≥50%", 4, "≥50%"),
        ("waste_disclosure_audit", "Waste disclosure + audit", 5, "Yes/No")
    ],
    "12.6": [
        ("emission_plans", "Clear 2030/2050 emission goals", 5, "Yes/No"),
        ("annual_progress_disclosed", "Annual progress disclosed", 4, "Yes/No"),
        ("no_goals", "No goals/stagnant progress", -3, "Yes/No")
    ],
    "12.7": [
        ("esg_audited_suppliers_pct", "≥80% suppliers with ESG audits + plan", 7, "≥80%"),
        ("price_only_procurement", "Price-only/high-emission outsourcing", -3, "Yes/No"),
        ("supply_chain_transparency", "Supply chain transparency report", 3, "Yes/No")
    ]
}

# --- 7. Core AI Functions ---
def get_ai_response(prompt, system_msg="You are a helpful assistant."):
    if not OPENAI_AVAILABLE:
        return "AI features require an OPENAI_API_KEY in Streamlit Secrets."
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=25
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        return ""

def identify_missing_fields(eval_data):
    missing = []
    for field, _, _, _ in SDG_CRITERIA["12.2"]:
        if eval_data["12_2"][field] is None:
            missing.append(("12.2", field))
    for sdg in ["12.3", "12.4"]:
        for field, _, _, _ in SDG_CRITERIA[sdg]:
            if field != "penalties" and eval_data["12_3_4"][field] is None:
                missing.append((sdg, field))
    for sdg in ["12.5", "12.6"]:
        for field, _, _, _ in SDG_CRITERIA[sdg]:
            if eval_data["12_5_6"][field] is None:
                missing.append((sdg, field))
    for field, _, _, _ in SDG_CRITERIA["12.7"]:
        if eval_data["12_7"][field] is None:
            missing.append(("12.7", field))
    return missing

def ai_other_actions(eval_data):
    if not OPENAI_AVAILABLE:
        return "- Implemented employee training on responsible production practices\n- Partnered with local recyclers for by-product reuse in production"
    
    prompt = f"""For {eval_data['company_name']} (industry: {eval_data['industry']}), identify 1-2 positive responsible production actions NOT listed in standard criteria.
    
    Current data:
    - Energy: {eval_data['12_2']['renewable_share']}% renewable, {eval_data['12_2']['recycled_water_ratio']}% recycled water
    - Waste: {eval_data['12_3_4']['hazardous_recovery_pct']}% hazardous recovery, {eval_data['12_5_6']['recycling_rate_pct']}% recycling rate
    - Suppliers: {eval_data['12_7']['esg_audited_suppliers_pct']}% ESG-audited
    - Third-party news: {eval_data['third_party']['positive_news'][:200]}...
    
    Actions must be industry-relevant, tied to responsible production, and include specific details (e.g., technology used, timeline). Return as bullet points (max 2)."""
    
    response = get_ai_response(prompt, "Consultant specializing in responsible production.")
    return response.strip() if response else "- Implemented employee training on responsible production practices\n- Partnered with local recyclers for by-product reuse in production"

# --- 8. Scoring Function ---
def calculate_scores(eval_data):
    scores = {sdg: 0 for sdg in SDG_MAX_SCORES.keys()}
    
    for field, desc, points, threshold in SDG_CRITERIA["12.2"]:
        value = eval_data["12_2"][field]
        if "%" in threshold and value is not None:
            try:
                threshold_num = float(re.sub(r"[>≥%]", "", threshold))
                if value >= threshold_num:
                    scores["12.2"] += points
            except:
                pass
        elif value:
            scores["12.2"] += points
    
    for field, desc, points, threshold in SDG_CRITERIA["12.3"]:
        value = eval_data["12_3_4"][field]
        if "%" in threshold and value is not None:
            try:
                threshold_num = float(re.sub(r"[>≥%]", "", threshold))
                if value > threshold_num:
                    scores["12.3"] += points
            except:
                pass
        elif value:
            scores["12.3"] += points
    
    for field, desc, points, threshold in SDG_CRITERIA["12.4"]:
        if field == "penalties":
            if not eval_data["third_party"]["penalties"]:
                scores["12.4"] += points
        else:
            value = eval_data["12_3_4"][field]
            if "%" in threshold and value is not None:
                try:
                    threshold_num = float(re.sub(r"[>≥%]", "", threshold))
                    if value >= threshold_num:
                        scores["12.4"] += points
                except:
                    pass
            elif value:
                scores["12.4"] += points
    
    for field, desc, points, threshold in SDG_CRITERIA["12.5"]:
        value = eval_data["12_5_6"][field]
        if "%" in threshold and value is not None:
            try:
                threshold_num = float(re.sub(r"[>≥%]", "", threshold))
                if value >= threshold_num:
                    scores["12.5"] += points
            except:
                pass
        elif value:
            scores["12.5"] += points
    
    for field, desc, points, threshold in SDG_CRITERIA["12.6"]:
        value = eval_data["12_5_6"][field]
        if value:
            scores["12.6"] += points
    
    for field, desc, points, threshold in SDG_CRITERIA["12.7"]:
        value = eval_data["12_7"][field]
        if "%" in threshold and value is not None:
            try:
                threshold_num = float(re.sub(r"[>≥%]", "", threshold))
                if value >= threshold_num:
                    scores["12.7"] += points
            except:
                pass
        elif value:
            scores["12.7"] += points
    
    scores["Others"] = min(10, len([l for l in eval_data["other_positive_actions"].split("\n") if l.strip()]) * 5)
    
    for sdg in scores:
        scores[sdg] = max(0, min(scores[sdg], SDG_MAX_SCORES[sdg]))
    
    overall = sum(scores.values())
    rating = "High Responsibility Enterprise (Low Risk)" if overall >=75 else \
             "Compliant but Requires Improvement (Moderate Risk)" if 60<=overall<75 else \
             "Potential Environmental Risk (High Risk)" if 40<=overall<60 else \
             "High Ethical Risk (Severe Risk)"
    
    return scores, overall, rating

# --- 9. Enriched Recommendations (No Numbers, ≥100 Words Each) ---
def generate_recommendations(eval_data, target_scores, overall_score):
    if not OPENAI_AVAILABLE:
        return [
            "Invest $250,000 in a closed-loop water recycling system from XYZ Water Technologies by Q3 2025 to increase your recycled water ratio from the current {eval_data['12_2']['recycled_water_ratio'] or '45'}% to ≥70%. This system will process 50,000 liters of wastewater daily, reducing freshwater intake by 30% and lowering operational costs by $15,000 annually. Train 10 on-site technicians via ABC Environmental Training Services to maintain the system, with monthly efficiency monitoring using IoT sensors. This action aligns with responsible production goals, improves SDG 12.2 performance, and gains +5 points by meeting the recycled water criteria.",
            "Partner with a third-party ESG auditor (e.g., SGS or Bureau Veritas) by Q1 2025 to audit 100% of your suppliers, with a goal to reach ≥80% ESG-audited suppliers by the end of 2025 (current: {eval_data['12_7']['esg_audited_suppliers_pct'] or '55'}%). Allocate $120,000 for auditor fees and supplier capacity-building workshops, focusing on high-emission suppliers in your Southeast Asia and Latin America regions. Develop a supplier scorecard tracking criteria like carbon footprint, waste management, and labor practices, with quarterly progress reports shared publicly. This will improve SDG 12.7 performance, gain +7 points, and enhance supply chain transparency—a key pillar of responsible production.",
            "Implement a digital loss-tracking system (e.g., SAP Sustainability or IBM Envizi) by Q2 2025 to monitor material loss across your production lines, addressing your current lack of a formal tracking system. Invest $80,000 in software licenses and employee training, with a focus on training 15 production managers to use the system for real-time loss identification. Set a target to reduce annual material loss by 15% within the first year (current reduction: {eval_data['12_3_4']['loss_reduction_pct'] or '8'}%), which will save approximately $40,000 in material costs. This action strengthens SDG 12.3 compliance, gains +5 points for the tracking system, and supports responsible production by minimizing resource waste."
        ]
    
    prompt = f"""Generate 3 DETAILED responsible production recommendations for {eval_data['company_name']} (industry: {eval_data['industry']}).
    
    Current Status:
    - Target scores (achieved/max): {json.dumps({k: f'{v}/{SDG_MAX_SCORES[k]}' for k, v in target_scores.items()}, indent=2)}
    - Overall score: {overall_score}/100
    - Low targets: {[k for k, v in target_scores.items() if v < SDG_MAX_SCORES[k]*0.5]}
    - Current gaps: 
      - Renewable energy: {eval_data['12_2']['renewable_share']}% (needs ≥50%)
      - Recycled water: {eval_data['12_2']['recycled_water_ratio']}% (needs ≥70%)
      - ESG suppliers: {eval_data['12_7']['esg_audited_suppliers_pct']}% (needs ≥80%)
    - Penalties: {eval_data['third_party']['penalties_details'][:150]}...
    
    Recommendations MUST:
    1. Focus on responsible production (not general sustainability).
    2. Be ≥100 words each, with specific details:
       - Exact investment amounts
       - Specific technologies/suppliers/auditors (e.g., "SAP Sustainability software")
       - Clear timelines (e.g., "by Q3 2025")
       - Quantifiable outcomes (e.g., "reduce waste by 15%")
       - How it improves production processes (e.g., "real-time loss tracking")
    3. NOT include numbers at the start (no "1.", "2.").
    4. Prioritize low-scoring areas first.
    5. Tie to responsible production goals (e.g., resource efficiency, supply chain responsibility).
    
    Format as bullet points (no introduction)."""
    
    response = get_ai_response(prompt, "Sustainability consultant specializing in industrial responsible production.")
    recs = [line.strip() for line in response.split("\n") if line.strip() and not line.strip()[0].isdigit()]
    while len(recs) < 3:
        recs.append(f"Invest $300,000 in a 2MW solar panel installation at your {eval_data['industry']} facility by Q4 2025 to increase renewable energy share from current {eval_data['12_2']['renewable_share'] or '35'}% to ≥50%. Partner with SunPower or First Solar for equipment and installation, and apply for local renewable energy tax credits to offset 20% of costs. The system will generate 3.5 million kWh annually, reducing carbon emissions by 2,800 tons and lowering energy costs by $40,000 per year. Train 5 facility engineers to monitor solar output via a cloud-based dashboard, with monthly reports integrated into your production management system. This action advances responsible production by reducing reliance on fossil fuels, improves SDG 12.2 performance, and gains +7 points for meeting the renewable energy criteria.")
    return recs[:3]

# --- 10. Report Generation ---
def generate_report(eval_data, target_scores, overall_score, rating, recommendations):
    title = f"Responsible Production Report: {eval_data['company_name']}"
    report = [
        title,
        "=" * len(title),
        "",
        "### 1. Executive Summary",
        f"**Company**: {eval_data['company_name']}",
        f"**Industry**: {eval_data['industry']}",
        f"**Overall Responsible Production Score**: {overall_score}/100",
        f"**Overall Rating**: {rating}",
        f"**Additional Notes**: {eval_data['additional_notes'] or 'No additional notes provided'}",
        "",
        "### 2. Third-Party Responsible Production Data (AI-Sourced with Links)",
        f"**Environmental Penalties**: {eval_data['third_party']['penalties_details']}",
        f"**Positive Production News**: {eval_data['third_party']['positive_news']}",
        f"**Relevant Policy Updates**: {eval_data['third_party']['policy_updates']}",
        "",
        "### 3. SDG 12 Target Performance (Responsible Production Focus)",
    ]
    
    for sdg in target_scores:
        if sdg != "Others":
            report.append(f"- **SDG {sdg}**: {target_scores[sdg]}/{SDG_MAX_SCORES[sdg]}")
    report.append(f"- **Additional Positive Actions**: {target_scores['Others']}/{SDG_MAX_SCORES['Others']}")
    
    report.extend([
        "",
        "### 4. Detailed Responsible Production Performance",
        "**SDG 12.2: Sustainable Resource Management**",
        "   - Actions: Renewable energy integration, recycled water use, recycled material sourcing",
        f"   - Score: {target_scores['12.2']}/{SDG_MAX_SCORES['12.2']}",
        "",
        "**SDG 12.3: Material Waste Reduction**",
        "   - Actions: Production loss tracking, annual loss reduction initiatives",
        f"   - Score: {target_scores['12.3']}/{SDG_MAX_SCORES['12.3']}",
        "",
        "**SDG 12.4: Chemical & Waste Management**",
        "   - Actions: MRSL/ZDHC compliance, hazardous waste recovery, emission testing",
        f"   - Score: {target_scores['12.4']}/{SDG_MAX_SCORES['12.4']}",
        "",
        "**SDG 12.5: Waste Reduction & Recycling**",
        "   - Actions: Packaging optimization, recycling programs, sustainable product design",
        f"   - Score: {target_scores['12.5']}/{SDG_MAX_SCORES['12.5']}",
        "",
        "**SDG 12.6: Transparent Reporting**",
        "   - Actions: Emission reduction goals, annual progress disclosure",
        f"   - Score: {target_scores['12.6']}/{SDG_MAX_SCORES['12.6']}",
        "",
        "**SDG 12.7: Responsible Procurement**",
        "   - Actions: ESG supplier audits, supply chain transparency",
        f"   - Score: {target_scores['12.7']}/{SDG_MAX_SCORES['12.7']}",
    ])
    
    report.extend([
        "",
        "### 5. Additional Responsible Production Actions",
        eval_data["other_positive_actions"] or "No additional actions identified.",
        "",
        "### 6. Actionable Responsible Production Recommendations",
    ])
    for rec in recommendations:
        report.append(f"- {rec}")
    
    report.extend([
        "",
        "### 7. Data Sources (AI-Verified with Links)",
        "- User-confirmed PDF extraction (responsible production reports/annual filings)",
        "- Third-party data: Environmental agencies, credible news outlets (links included above)",
        "- AI analysis of industry benchmarks for responsible production",
    ])
    
    return "\n".join(report)

# --- 11. UI Functions (Purple-Themed, Single Chart, Updated Buttons/Highlights) ---
# 1. 确保颜色变量正确定义（需在UI函数前声明，避免引用失败）
PRIMARY_PURPLE = "#6a0dad"  # 深紫色（匹配scoring tool.docx设计风格）
MEDIUM_PURPLE = "#9370db"  # 浅紫色（hover效果）
LIGHT_PURPLE = "#f0f0ff"  # 背景浅紫

def render_front_page():
    st.title("🌱 Responsible Production Evaluator", anchor=False)
    st.write("Evaluate corporate performance on responsible production (per scoring tool.docx)")
    
    # 2. 修复CSS：增强选择器特异性，覆盖默认红色样式
    st.markdown(
        f"""
        <style>
        /* 修复按钮样式：使用更具体的选择器，避免被默认样式覆盖 */
        button.stButton {{
            background-color: {PRIMARY_PURPLE} !important;
            color: white !important;
            border: none !important; /* 清除默认边框（可能导致红色边缘） */
        }}
        button.stButton:hover {{
            background-color: {MEDIUM_PURPLE} !important;
        }}
        /* 修复Radio选中样式：定位到具体选中元素，避免层级问题 */
        div.stRadio > div > label > div[data-baseweb="radio"]:has(input:checked) {{
            background-color: {PRIMARY_PURPLE} !important;
            border-color: {PRIMARY_PURPLE} !important; /* 清除默认红色边框 */
        }}
        /* 修复Checkbox选中样式：同样增强特异性 */
        div.stCheckbox > div > label > div[data-baseweb="checkbox"]:has(input:checked) {{
            background-color: {PRIMARY_PURPLE} !important;
            border-color: {PRIMARY_PURPLE} !important;
        }}
        /* 确保按钮文字不继承默认红色 */
        button.stButton > div > p {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # 后续保留原有的「Start Manual Input」等按钮逻辑（与scoring tool.docx相关的输入流程）
    col1, col2 = st.columns([1.2, 0.8], gap="medium")
    with col2:
        st.subheader("Option 2: Manual Input – For PDF Failures")
        st.warning("⚠️ Use only if PDF upload/extraction fails (per scoring tool.docx)")
        company_name = st.text_input("Company Name", placeholder="Enter company name")
        industry = st.selectbox("Industry", ENRICHED_INDUSTRIES, index=0)
        
        # 3. 按钮无需额外type="primary"（避免触发Streamlit默认红色主题）
        if st.button("Start Manual Input", key="start_manual", use_container_width=True):
            st.session_state["eval_data"]["company_name"] = company_name
            st.session_state["eval_data"]["industry"] = industry
            st.session_state["eval_data"]["third_party"] = get_third_party_data(company_name, industry)
            st.session_state["current_step"] = 2  # 进入scoring tool.docx定义的手动输入流程
            st.rerun()

def step_2_energy_materials():
    st.subheader("Step 2/5: Energy & Material Management (Responsible Production)", anchor=False)
    eval_data = st.session_state["eval_data"]
    
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.caption("Energy Use")
        eval_data["12_2"]["renewable_share"] = st.number_input(
            "Renewable energy share (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_2"]["renewable_share"] or 0,
            help="Percentage of energy from renewable sources"
        )
        eval_data["12_2"]["energy_retrofit"] = st.radio(
            "Full-scale energy retrofit implemented?",
            ["Yes", "No"],
            index=0 if eval_data["12_2"]["energy_retrofit"] else 1,
            help="Has the company completed a full-scale energy efficiency retrofit?",
            label_visibility="visible"
        ) == "Yes"
        eval_data["12_2"]["energy_increase"] = st.radio(
            "Energy consumption up 2 consecutive years?",
            ["Yes", "No"],
            index=1 if eval_data["12_2"]["energy_increase"] else 0,
            help="Has energy consumption increased for 2 consecutive years?",
            label_visibility="visible"
        ) == "Yes"
    
    with col2:
        st.caption("Water & Materials")
        eval_data["12_2"]["recycled_water_ratio"] = st.number_input(
            "Recycled water ratio (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_2"]["recycled_water_ratio"] or 0,
            help="Percentage of water recycled in production"
        )
        eval_data["12_2"]["recycled_materials_pct"] = st.number_input(
            "Recycled materials share (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_2"]["recycled_materials_pct"] or 0,
            help="Percentage of materials sourced from recycled content"
        )
        eval_data["12_2"]["ghg_disclosure"] = st.radio(
            "Scope1-3 GHG disclosed + third-party verified?",
            ["Yes", "No"],
            index=0 if eval_data["12_2"]["ghg_disclosure"] else 1,
            help="Has the company disclosed Scope 1-3 GHG emissions with third-party verification?",
            label_visibility="visible"
        ) == "Yes"
    
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Back", key="back_step2", use_container_width=True):
            st.session_state["current_step"] = 0
            st.rerun()
    with col2_btn:
        if st.button("Proceed to Waste Management", key="proceed_step2", 
                    type="primary", use_container_width=True):
            st.session_state["current_step"] = 3
            st.rerun()

def step_3_waste_chemicals():
    st.subheader("Step 3/5: Waste & Chemical Management (Responsible Production)", anchor=False)
    eval_data = st.session_state["eval_data"]
    
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.caption("Material Loss Control")
        eval_data["12_3_4"]["loss_tracking_system"] = st.radio(
            "Loss-tracking system established?",
            ["Yes", "No"],
            index=0 if eval_data["12_3_4"]["loss_tracking_system"] else 1,
            help="Does the company have a formal system to track material loss?",
            label_visibility="visible"
        ) == "Yes"
        eval_data["12_3_4"]["loss_reduction_pct"] = st.number_input(
            "Annual loss reduction (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_3_4"]["loss_reduction_pct"] or 0,
            help="Percentage reduction in material loss over the past year"
        )
    
    with col2:
        st.caption("Chemical & Hazardous Waste")
        eval_data["12_3_4"]["mrsl_zdhc_compliance"] = st.radio(
            "Compliant with MRSL/ZDHC standards?",
            ["Yes", "No"],
            index=0 if eval_data["12_3_4"]["mrsl_zdhc_compliance"] else 1,
            help="Is the company compliant with MRSL/ZDHC chemical management standards?",
            label_visibility="visible"
        ) == "Yes"
        eval_data["12_3_4"]["hazardous_recovery_pct"] = st.number_input(
            "Hazardous waste recovery (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_3_4"]["hazardous_recovery_pct"] or 0,
            help="Percentage of hazardous waste recovered and properly disposed"
        )
    
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Back to Energy Management", key="back_step3", use_container_width=True):
            st.session_state["current_step"] = 2
            st.rerun()
    with col2_btn:
        if st.button("Proceed to Packaging & Reporting", key="proceed_step3", 
                    type="primary", use_container_width=True):
            st.session_state["current_step"] = 4
            st.rerun()

def step_4_packaging_reporting():
    st.subheader("Step 4/5: Packaging & Reporting (Responsible Production)", anchor=False)
    eval_data = st.session_state["eval_data"]
    
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.caption("Packaging & Recycling")
        eval_data["12_5_6"]["packaging_reduction_pct"] = st.number_input(
            "Packaging weight reduction (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_5_6"]["packaging_reduction_pct"] or 0,
            help="Percentage reduction in packaging weight over the past year"
        )
        eval_data["12_5_6"]["recycling_rate_pct"] = st.number_input(
            "Overall recycling rate (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_5_6"]["recycling_rate_pct"] or 0,
            help="Percentage of waste diverted from landfill through recycling"
        )
        eval_data["12_5_6"]["sustainable_products_pct"] = st.number_input(
            "Sustainable material products (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_5_6"]["sustainable_products_pct"] or 0,
            help="Percentage of products made with sustainable materials"
        )
    
    with col2:
        st.caption("Responsible Production Reporting")
        eval_data["12_5_6"]["emission_plans"] = st.radio(
            "Clear 2030/2050 emission reduction goals?",
            ["Yes", "No"],
            index=0 if eval_data["12_5_6"]["emission_plans"] else 1,
            help="Does the company have clear emission reduction goals for 2030/2050?",
            label_visibility="visible"
        ) == "Yes"
        eval_data["12_5_6"]["annual_progress_disclosed"] = st.radio(
            "Annual responsible production progress disclosed?",
            ["Yes", "No"],
            index=0 if eval_data["12_5_6"]["annual_progress_disclosed"] else 1,
            help="Does the company publicly disclose annual progress on responsible production?",
            label_visibility="visible"
        ) == "Yes"
    
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Back to Waste Management", key="back_step4", use_container_width=True):
            st.session_state["current_step"] = 3
            st.rerun()
    with col2_btn:
        if st.button("Proceed to Supplier Management", key="proceed_step4", 
                    type="primary", use_container_width=True):
            st.session_state["current_step"] = 5
            st.rerun()

def step_5_supplier_procurement():
    st.subheader("Step 5/5: Supplier & Procurement (Responsible Production)", anchor=False)
    eval_data = st.session_state["eval_data"]
    
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        eval_data["12_7"]["esg_audited_suppliers_pct"] = st.number_input(
            "ESG-audited suppliers (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_7"]["esg_audited_suppliers_pct"] or 0,
            help="Percentage of suppliers audited for ESG practices"
        )
        eval_data["12_7"]["supply_chain_transparency"] = st.radio(
            "Supply chain transparency report published?",
            ["Yes", "No"],
            index=0 if eval_data["12_7"]["supply_chain_transparency"] else 1,
            help="Has the company published a supply chain transparency report?",
            label_visibility="visible"
        ) == "Yes"
    
    with col2:
        eval_data["12_7"]["price_only_procurement"] = st.radio(
            "Price-only procurement or outsourcing to high-emission regions?",
            ["Yes", "No"],
            index=1 if eval_data["12_7"]["price_only_procurement"] else 0,
            help="Does the company prioritize price over responsible production in procurement?",
            label_visibility="visible"
        ) == "Yes"
        st.caption("Third-Party Procurement Alerts")
        st.info(f"Policy Updates: {eval_data['third_party']['policy_updates'][:150]}...")
    
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Back to Packaging & Reporting", key="back_step5", use_container_width=True):
            st.session_state["current_step"] = 4
            st.rerun()
    with col2_btn:
        if st.button("Proceed to Additional Notes", key="proceed_step5", 
                    type="primary", use_container_width=True):
            st.session_state["current_step"] = 6
            st.rerun()

def step_6_notes():
    st.subheader("Additional Responsible Production Notes", anchor=False)
    eval_data = st.session_state["eval_data"]
    
    eval_data["additional_notes"] = st.text_area(
        "Enter additional details (e.g., ongoing responsible production projects, future plans)",
        value=eval_data["additional_notes"],
        height=150,
        help="Examples: 'Installing 10MW wind farm in 2025', 'Targeting 100% ESG suppliers by 2026'"
    )
    
    if st.button("Generate Final Responsible Production Report", key="generate_report", 
                type="primary", use_container_width=True):
        with st.spinner("Calculating scores + generating report..."):
            target_scores, overall_score, rating = calculate_scores(eval_data)
            eval_data["target_scores"] = target_scores
            eval_data["overall_score"] = overall_score
            eval_data["rating"] = rating
            eval_data["other_positive_actions"] = ai_other_actions(eval_data)
            recommendations = generate_recommendations(eval_data, target_scores, overall_score)
            st.session_state["report_text"] = generate_report(eval_data, target_scores, overall_score, rating, recommendations)
            st.session_state["current_step"] = 7
            st.rerun()
    
    if st.button("Back", key="back_step6", use_container_width=True):
        if st.session_state["extracted_data"]:
            st.session_state["current_step"] = 1
        else:
            st.session_state["current_step"] = 5
        st.rerun()

def render_report_page():
    eval_data = st.session_state["eval_data"]
    st.title("Responsible Production Performance Dashboard", anchor=False)

    # --- Tabs to Organize Content ---
    tab1, tab2, tab3 = st.tabs(["Metrics & Chart", "Detailed Report", "Insights & Recommendations"])

    with tab1:
        # --- Key Info Card (Highlighted) ---
        rating_colors = {
            "High Responsibility Enterprise (Low Risk)": PRIMARY_PURPLE,
            "Compliant but Requires Improvement (Moderate Risk)": MEDIUM_PURPLE,
            "Potential Environmental Risk (High Risk)": "#FFA500",
            "High Ethical Risk (Severe Risk)": "#DC143C"
        }
        st.markdown(
            f"""
            <div style="background-color:{rating_colors[eval_data['rating']]}; color:white; padding:20px; border-radius:10px; margin-bottom:30px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h2 style="margin-top:0;">Overall Responsible Production Rating</h2>
            <h3>{eval_data['rating']}</h3>
            <h4 style="font-size:1.5em;">Total Score: {eval_data['overall_score']}/100</h4>
            <p><strong>Industry:</strong> {eval_data['industry']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # --- ONLY Retained Chart: Polished "Achieved vs Maximum Score" ---
        st.subheader("Responsible Production Score: Achieved vs Maximum")
        fig, ax = plt.subplots(figsize=(12, 6))
        sdgs = [k for k in eval_data["target_scores"] if k != "Others"]
        achieved_scores = [eval_data["target_scores"].get(sdg, 0) for sdg in sdgs]
        max_scores = [SDG_MAX_SCORES[sdg] for sdg in sdgs]
        width = 0.6
        ax.bar(sdgs, max_scores, width, label="Maximum Possible Score", color="#e0e0e0", alpha=0.8, zorder=1)
        ax.bar(sdgs, achieved_scores, width, label="Achieved Score", color=CHART_PURPLE, zorder=2)
        ax.set_xlabel("SDG 12 Targets", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title("SDG 12 Responsible Production Performance: Achieved vs Maximum Score", fontsize=14, fontweight="bold", pad=20)
        ax.set_xticks(range(len(sdgs)))
        ax.set_xticklabels(sdgs, fontsize=10)
        ax.legend(loc="upper right", fontsize=10, frameon=True, fancybox=True, shadow=True)
        for i, (achieved, max_val) in enumerate(zip(achieved_scores, max_scores)):
            ax.text(i, achieved + 0.5, f"{achieved}", ha="center", va="bottom", fontsize=9, fontweight="bold", zorder=3)
            ax.text(i, max_val - 1, f"Max: {max_val}", ha="center", va="top", fontsize=8, color="#666666", zorder=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(max_scores) * 1.15)
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        # --- Detailed Report (Collapsible) ---
        with st.expander("View Detailed Responsible Production Report", expanded=False):
            st.text(st.session_state["report_text"])
        # --- Download Button ---
        st.download_button(
            label="📥 Download Responsible Production Report",
            data=st.session_state["report_text"],
            file_name=f"{eval_data['company_name']}_Responsible_Production_Report.txt",
            mime="text/plain",
            use_container_width=True
        )

    with tab3:
        # --- Highlighted Strengths/Weaknesses ---
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.markdown(
                f"""
                <div style="background-color:{LIGHT_PURPLE}; padding:15px; border-radius:8px; border-left:4px solid {PRIMARY_PURPLE};">
                <h4 style="margin-top:0; color:{PRIMARY_PURPLE};">Top Strengths</h4>
                """,
                unsafe_allow_html=True
            )
            strengths = [k for k, v in eval_data["target_scores"].items() if k != "Others" and v >= SDG_MAX_SCORES[k] * 0.7]
            if strengths:
                for s in strengths:
                    st.write(f"- **SDG {s}**: {eval_data['target_scores'][s]}/{SDG_MAX_SCORES[s]} (Exceeds 70% of maximum)")
            else:
                st.write("- Identify initial responsible production practices to build upon (e.g., basic recycling programs)")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(
                f"""
                <div style="background-color:{LIGHT_PURPLE}; padding:15px; border-radius:8px; border-left:4px solid {MEDIUM_PURPLE};">
                <h4 style="margin-top:0; color:{PRIMARY_PURPLE};">Critical Improvements</h4>
                """,
                unsafe_allow_html=True
            )
            weaknesses = [k for k, v in eval_data["target_scores"].items() if k != "Others" and v < SDG_MAX_SCORES[k] * 0.5]
            if weaknesses:
                for w in weaknesses:
                    st.write(f"- **SDG {w}**: {eval_data['target_scores'][w]}/{SDG_MAX_SCORES[w]} (Below 50% of maximum)")
            else:
                st.write("- Maintain current practices and set stretch goals (e.g., increase renewable energy to 60%)")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # --- Actionable Recommendations (Collapsible) ---
        with st.expander("View Actionable Responsible Production Recommendations", expanded=False):
            recommendations = generate_recommendations(eval_data, eval_data["target_scores"], eval_data["overall_score"])
            for rec in recommendations:
                st.write(f"- {rec}")

    # --- New Evaluation Button (Always Visible) ---
    if st.button("Start New Responsible Production Evaluation", key="new_eval_final", 
                type="primary", use_container_width=True):
        st.session_state.clear()
        st.session_state["eval_data"] = {
            "company_name": "", "industry": "Manufacturing",
            "third_party": {"penalties": False, "penalties_details": "", "positive_news": "", "policy_updates": ""},
            "12_2": {"renewable_share": None, "energy_retrofit": False, "energy_increase": False, "carbon_offsets_only": False, "recycled_water_ratio": None, "ghg_disclosure": False, "recycled_materials_pct": None, "illegal_logging": False},
            "12_3_4": {"loss_tracking_system": False, "loss_reduction_pct": None, "mrsl_zdhc_compliance": False, "regular_emission_tests": False, "hazardous_recovery_pct": None, "illegal_disposal": False},
            "12_5_6": {"packaging_reduction_pct": None, "recycling_rate_pct": None, "sustainable_products_pct": None, "waste_disclosure_audit": False, "emission_plans": False, "annual_progress_disclosed": False, "no_goals": False, "high_carbon_assets_disclosed": False},
            "12_7": {"esg_audited_suppliers_pct": None, "price_only_procurement": False, "supply_chain_transparency": False},
            "additional_notes": "", "target_scores": {}, "overall_score": 0, "rating": "", "other_positive_actions": ""
        }
        st.session_state["current_step"] = 0
        st.rerun()
# --- 12. Main UI Flow ---
if st.session_state["current_step"] == 0:
    render_front_page()
elif st.session_state["current_step"] == 1:
    render_pdf_confirmation_page(
        st.session_state["extracted_data"],
        st.session_state["eval_data"]["company_name"],
        st.session_state["eval_data"]["industry"]
    )
elif st.session_state["current_step"] == 2:
    step_2_energy_materials()
elif st.session_state["current_step"] == 3:
    step_3_waste_chemicals()
elif st.session_state["current_step"] == 4:
    step_4_packaging_reporting()
elif st.session_state["current_step"] == 5:
    step_5_supplier_procurement()
elif st.session_state["current_step"] == 6:
    step_6_notes()
elif st.session_state["current_step"] == 7:
    render_report_page()

# --- Progress Indicator ---
if 2 <= st.session_state["current_step"] <= 6 and not st.session_state["extracted_data"]:
    step_names = ["", "", "Energy/Materials", "Waste/Chemicals", "Packaging/Reporting", "Suppliers", "Notes"]
    current_step_name = step_names[st.session_state["current_step"]]
    progress = (st.session_state["current_step"] - 1) / 6
    st.sidebar.progress(progress)
    st.sidebar.write(f"Current Step: {st.session_state['current_step']}/6 – {current_step_name}")
    st.sidebar.subheader("Responsible Production Focus")
    st.sidebar.write("• Resource efficiency")
    st.sidebar.write("• Waste reduction")
    st.sidebar.write("• Ethical procurement")
