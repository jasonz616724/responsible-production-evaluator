import streamlit as st
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import io

# --- Theme Configuration (Purple Palette for UI Consistency)
PRIMARY_PURPLE = "#6a0dad"    # Primary color for buttons/active elements
MEDIUM_PURPLE = "#9370db"     # Hover state color
LIGHT_PURPLE = "#f0f0ff"     # Background color for cards
TEXT_COLOR = "#333333"       # Text color for readability

# --- Third-Party Data Retrieval (Per Evaluation Standards)
def get_third_party_data(company_name, industry):
    """Retrieve AI-sourced third-party data aligned with assessment criteria (2023-2024)."""
    if not company_name or not OPENAI_AVAILABLE:
        return {
            "penalties": False, 
            "penalties_details": "Third-party data not retrieved (missing company name or AI key)",
            "positive_news": "Third-party data not retrieved",
            "policy_updates": "Third-party data not retrieved"
        }
    
    prompt = f"""For {company_name} (industry: {industry}), extract ONLY the following verified third-party data per evaluation standards:
    1. Environmental penalties (2023-2024): Violations related to responsible production (e.g., illegal waste disposal). Include authority, date, and direct regulatory/news link.
    2. Positive production news (2023-2024): Actions like recycling partnerships or renewable energy adoption. Include source link.
    3. Policy updates (2023-2024): Regional laws impacting responsible production (e.g., extended producer responsibility). Include policy document link.
    
    Prioritize sources: Government environmental agencies (EPA, EU EEA), Bloomberg Green, Reuters, official regulatory databases.
    Return ONLY JSON with keys: penalties (bool), penalties_details (str with links), positive_news (str with links), policy_updates (str with links). Use "No relevant data found (AI search returned no results)" for empty fields. No hardcoded content."""
    
    response = get_ai_response(prompt, "Environmental data analyst specializing in responsible production assessments")
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
            "penalties_details": f"Invalid AI response: {response[:100]}... (No links available)",
            "positive_news": "No valid third-party data found (No links available)",
            "policy_updates": "No valid third-party data found (No links available)"
        }

# --- PDF Processing (Aligned with Data Extraction Standards)
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("⚠️ PyPDF2 library not found. Install with 'pip install PyPDF2' to enable PDF upload (required for automated data extraction).")

def extract_full_pdf_text(file):
    """Extract text from PDF for assessment data retrieval."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        full_text = ""
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text() or ""
            full_text += f"\n--- Page {page_num} ---\n{page_text}"
        
        if len(full_text.strip()) < 100:
            st.warning("⚠️ PDF may contain image-based text (unextractable). Use a text-based PDF or manual input.")
        return full_text
    except Exception as e:
        st.error(f"❌ PDF Extraction Error: {str(e)} (Text-based PDF required for assessment).")
        return ""

def extract_assessment_data_from_pdf(pdf_text, company_name, industry):
    """Extract responsible production data from PDF per evaluation metrics."""
    if len(pdf_text.strip()) < 500:
        st.error("❌ Insufficient text for data extraction. Use a complete responsible production report.")
        return {}
    
    prompt = f"""Extract responsible production data from the following PDF text for {company_name} (industry: {industry}) per standard evaluation metrics:
    
    PDF Text (first 10,000 characters):
    {pdf_text[:10000]}
    
    Required Metrics:
    - renewable_share: % renewable energy (e.g., 55 = 55%)
    - energy_retrofit: True/False (full-scale energy efficiency retrofit completed)
    - energy_increase: True/False (energy consumption up 2 consecutive years)
    - carbon_offsets_only: True/False (relies solely on carbon offsets)
    - recycled_water_ratio: % recycled water used (e.g., 75 = 75%)
    - ghg_disclosure: True/False (Scope 1-3 GHG disclosed + third-party verified)
    - recycled_materials_pct: % recycled materials in production (e.g., 35 = 35%)
    - illegal_logging: True/False (any illegal logging incidents)
    - loss_tracking_system: True/False (material loss tracking system in place)
    - loss_reduction_pct: % annual material loss reduction (e.g., 12 = 12%)
    - mrsl_zdhc_compliance: True/False (compliant with MRSL/ZDHC standards)
    - regular_emission_tests: True/False (regular emission testing conducted)
    - hazardous_recovery_pct: % hazardous waste recovered (e.g., 92 = 92%)
    - illegal_disposal: True/False (any improper waste disposal)
    - packaging_reduction_pct: % packaging weight reduction (e.g., 25 = 25%)
    - recycling_rate_pct: % waste recycled (e.g., 85 = 85%)
    - sustainable_products_pct: % products with sustainable materials (e.g., 55 = 55%)
    - waste_disclosure_audit: True/False (waste data disclosed + third-party audited)
    - emission_plans: True/False (clear 2030/2050 emission reduction goals)
    - annual_progress_disclosed: True/False (annual responsible production progress published)
    - no_goals: True/False (no goals or stagnant progress)
    - high_carbon_assets_disclosed: True/False (high-carbon assets disclosed + reduction pathway)
    - esg_audited_suppliers_pct: % suppliers with ESG audits (e.g., 85 = 85%)
    - price_only_procurement: True/False (price-only procurement or outsourcing to high-emission regions)
    - supply_chain_transparency: True/False (supply chain transparency report published)
    
    Return ONLY valid JSON. Use null for unknown values. No extra text."""
    
    response = get_ai_response(prompt, "ESG data extractor trained on responsible production evaluation metrics")
    if not response:
        st.error("❌ AI returned no extraction results. Manual data input required.")
        return {}
    
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        st.error(f"❌ No valid JSON in AI response: {response[:200]}... (Manual input required)")
        return {}
    
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError as e:
        st.error(f"❌ Extracted data parsing failed: {str(e)}. Raw JSON: {json_match.group()[:200]}...")
        return {}

def ai_fill_missing_metrics(extracted_data, industry):
    """AI-populate missing metrics using industry benchmarks (per evaluation standards)."""
    if not OPENAI_AVAILABLE or not extracted_data:
        return extracted_data
    
    missing_fields = [k for k, v in extracted_data.items() if v is None]
    if not missing_fields:
        return extracted_data
    
    prompt = f"""Fill missing responsible production metrics for a {industry} company using industry benchmarks (per evaluation standards):
    Extracted data (missing fields: {missing_fields}):
    {json.dumps(extracted_data, indent=2)}
    
    Rules:
    1. Use realistic {industry} averages (e.g., manufacturing: 35% renewable energy; textiles: 25% recycled materials).
    2. Booleans: Assume "False" for high-risk metrics (e.g., illegal_logging = False) and "True" for common practices (e.g., regular_emission_tests = True).
    3. Preserve existing non-null values.
    Return ONLY updated JSON. No extra text."""
    
    response = get_ai_response(prompt, f"Data analyst specializing in {industry} responsible production benchmarks")
    try:
        return json.loads(response) if response else extracted_data
    except:
        st.warning("⚠️ AI could not fill missing metrics. Using original extracted data.")
        return extracted_data

def render_pdf_confirmation_page(extracted_data, company_name, industry):
    """PDF-extracted data confirmation page (for evaluation validation)."""
    st.subheader(f"Extracted Data Confirmation (Company: {company_name})")
    st.write("Review and edit extracted data (marked * = AI-populated per industry benchmarks).")
    
    # Metric grouping aligned with evaluation framework
    field_groups = {
        "Energy & Water Management": [
            ("renewable_share", "Renewable energy share (%)", "number"),
            ("recycled_water_ratio", "Recycled water ratio (%)", "number"),
            ("energy_retrofit", "Full-scale energy retrofit completed?", "bool"),
            ("energy_increase", "Energy consumption up 2 consecutive years?", "bool"),
            ("carbon_offsets_only", "Relies solely on carbon offsets?", "bool"),
            ("ghg_disclosure", "Scope 1-3 GHG disclosed + verified?", "bool")
        ],
        "Material & Waste Management": [
            ("recycled_materials_pct", "Recycled materials share (%)", "number"),
            ("illegal_logging", "Any illegal logging incidents?", "bool"),
            ("loss_tracking_system", "Material loss tracking system in place?", "bool"),
            ("loss_reduction_pct", "Annual material loss reduction (%)", "number"),
            ("hazardous_recovery_pct", "Hazardous waste recovery (%)", "number"),
            ("illegal_disposal", "Any improper waste disposal?", "bool")
        ],
        "Packaging & Reporting": [
            ("packaging_reduction_pct", "Packaging weight reduction (%)", "number"),
            ("recycling_rate_pct", "Overall recycling rate (%)", "number"),
            ("sustainable_products_pct", "Products with sustainable materials (%)", "number"),
            ("waste_disclosure_audit", "Waste data disclosed + audited?", "bool"),
            ("emission_plans", "Clear 2030/2050 emission goals?", "bool"),
            ("annual_progress_disclosed", "Annual progress published?", "bool"),
            ("no_goals", "No goals or stagnant progress?", "bool"),
            ("high_carbon_assets_disclosed", "High-carbon assets disclosed + reduction pathway?", "bool")
        ],
        "Supplier & Procurement": [
            ("esg_audited_suppliers_pct", "ESG-audited suppliers (%)", "number"),
            ("price_only_procurement", "Price-only procurement or high-emission outsourcing?", "bool"),
            ("supply_chain_transparency", "Supply chain transparency report published?", "bool")
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
                        index=0 if current_value else 1
                    ) == "Yes"
                    confirmed_data[field] = new_value
    
    # Confirmation buttons (purple theme)
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Confirm Data & Proceed", key="confirm_pdf", use_container_width=True):
            eval_data = st.session_state["eval_data"]
            # Map confirmed data to session state (aligned with evaluation metrics)
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
            st.session_state["current_step"] = 6  # Move to notes step
            st.rerun()
    
    with col2_btn:
        if st.button("Re-Extract from PDF", key="reextract_pdf", use_container_width=True):
            st.session_state["current_step"] = 0
            st.rerun()

# --- OpenAI Client Setup (For Evaluation-Specific AI Functions)
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except KeyError:
    st.warning("⚠️ OPENAI_API_KEY not configured (add to .streamlit/secrets.toml). AI features (extraction, recommendations) disabled.")
    OPENAI_AVAILABLE = False
except Exception as e:
    st.error(f"⚠️ OpenAI Initialization Error: {str(e)}. AI features disabled.")
    OPENAI_AVAILABLE = False

def get_ai_response(prompt, system_msg="You are an expert in responsible production evaluation."):
    """Generate AI responses aligned with evaluation standards."""
    if not OPENAI_AVAILABLE:
        return "AI features require an OPENAI_API_KEY (add to .streamlit/secrets.toml)."
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=25
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"AI Request Failed: {str(e)}. Manual input recommended.")
        return ""

# --- Session State Initialization (Aligned with Evaluation Metrics)
if "eval_data" not in st.session_state:
    st.session_state["eval_data"] = {
        "company_name": "",
        "industry": "Manufacturing",
        "third_party": {"penalties": False, "penalties_details": "", "positive_news": "", "policy_updates": ""},
        # Metric group 1: Energy & Resource Management
        "12_2": {
            "renewable_share": None, "energy_retrofit": False, "energy_increase": False,
            "carbon_offsets_only": False, "recycled_water_ratio": None, "ghg_disclosure": False,
            "recycled_materials_pct": None, "illegal_logging": False
        },
        # Metric group 2: Loss & Waste Management
        "12_3_4": {
            "loss_tracking_system": False, "loss_reduction_pct": None,
            "mrsl_zdhc_compliance": False, "regular_emission_tests": False,
            "hazardous_recovery_pct": None, "illegal_disposal": False
        },
        # Metric group 3: Packaging & Reporting
        "12_5_6": {
            "packaging_reduction_pct": None, "recycling_rate_pct": None,
            "sustainable_products_pct": None, "waste_disclosure_audit": False,
            "emission_plans": False, "annual_progress_disclosed": False, "no_goals": False,
            "high_carbon_assets_disclosed": False
        },
        # Metric group 4: Supplier Management
        "12_7": {
            "esg_audited_suppliers_pct": None, "price_only_procurement": False,
            "supply_chain_transparency": False
        },
        "additional_notes": "",
        "target_scores": {}, "overall_score": 0, "rating": "", "other_positive_actions": ""
    }
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0  # 0: Home, 1: PDF Confirmation, 2-6: Manual Input, 7: Report
if "pdf_extracted_text" not in st.session_state:
    st.session_state["pdf_extracted_text"] = ""
if "extracted_data" not in st.session_state:
    st.session_state["extracted_data"] = {}

# --- Evaluation Constants (Metrics & Scoring)
# Industry list aligned with evaluation coverage
ENRICHED_INDUSTRIES = [
    "Manufacturing", "Food & Beverage", "Textiles", "Chemicals", "Electronics",
    "Automotive", "Construction", "Healthcare", "Retail", "Agriculture",
    "Logistics", "Pharmaceuticals", "Paper & Pulp", "Furniture", "Cosmetics", "Other"
]

# Maximum scores per evaluation metric group
METRIC_MAX_SCORES = {
    "12.2": 29, "12.3": 9, "12.4": 16, "12.5": 17, "12.6": 9, "12.7": 10, "Others": 10
}

# Detailed metric criteria (scoring rules)
METRIC_CRITERIA = {
    "12.2": [
        ("renewable_share", "Renewable energy share ≥50%", 7, "≥50%"),
        ("energy_retrofit", "Full-scale energy retrofit completed", 5, "Yes/No"),
        ("energy_increase", "Energy consumption up 2 consecutive years", -5, "Yes/No"),
        ("carbon_offsets_only", "Relies solely on carbon offsets", -3, "Yes/No"),
        ("recycled_water_ratio", "Recycled water ratio ≥70%", 5, "≥70%"),
        ("ghg_disclosure", "Scope 1-3 GHG disclosed + verified", 7, "Yes/No"),
        ("recycled_materials_pct", "Recycled materials ≥30%", 5, "≥30%"),
        ("illegal_logging", "Illegal logging incidents", -7, "Yes/No")
    ],
    "12.3": [
        ("loss_tracking_system", "Material loss tracking system in place", 5, "Yes/No"),
        ("loss_reduction_pct", "Annual material loss reduction >10%", 4, ">10%")
    ],
    "12.4": [
        ("mrsl_zdhc_compliance", "Compliant with MRSL/ZDHC standards", 5, "Yes/No"),
        ("regular_emission_tests", "Regular emission testing conducted", 3, "Yes/No"),
        ("hazardous_recovery_pct", "Hazardous waste recovery ≥90%", 5, "≥90%"),
        ("illegal_disposal", "Improper waste disposal", -3, "Yes/No"),
        ("penalties", "No environmental penalties", 3, "Yes/No")
    ],
    "12.5": [
        ("packaging_reduction_pct", "Packaging weight reduction ≥20%", 4, "≥20%"),
        ("recycling_rate_pct", "Overall recycling rate ≥80%", 4, "≥80%"),
        ("sustainable_products_pct", "Sustainable material products ≥50%", 4, "≥50%"),
        ("waste_disclosure_audit", "Waste data disclosed + audited", 5, "Yes/No")
    ],
    "12.6": [
        ("emission_plans", "Clear 2030/2050 emission goals", 5, "Yes/No"),
        ("annual_progress_disclosed", "Annual progress published", 4, "Yes/No"),
        ("no_goals", "No goals or stagnant progress", -3, "Yes/No")
    ],
    "12.7": [
        ("esg_audited_suppliers_pct", "≥80% suppliers with ESG audits + implementation plan", 7, "≥80%"),
        ("price_only_procurement", "Price-only procurement or high-emission outsourcing", -3, "Yes/No"),
        ("supply_chain_transparency", "Supply chain transparency report published", 3, "Yes/No")
    ]
}

# --- Core Evaluation Functions
def identify_missing_metrics(eval_data):
    """Identify missing metrics required for evaluation."""
    missing = []
    for field, _, _, _ in METRIC_CRITERIA["12.2"]:
        if eval_data["12_2"][field] is None:
            missing.append(("12.2", field))
    for metric_group in ["12.3", "12.4"]:
        for field, _, _, _ in METRIC_CRITERIA[metric_group]:
            if field != "penalties" and eval_data["12_3_4"][field] is None:
                missing.append((metric_group, field))
    for metric_group in ["12.5", "12.6"]:
        for field, _, _, _ in METRIC_CRITERIA[metric_group]:
            if eval_data["12_5_6"][field] is None:
                missing.append((metric_group, field))
    for field, _, _, _ in METRIC_CRITERIA["12.7"]:
        if eval_data["12_7"][field] is None:
            missing.append(("12.7", field))
    return missing

def ai_identify_additional_actions(eval_data):
    """AI-identify additional positive actions (aligned with evaluation's 'Others' category)."""
    if not OPENAI_AVAILABLE:
        return "- Implemented employee training on responsible production practices\n- Partnered with local recyclers for by-product reuse"
    
    prompt = f"""For {eval_data['company_name']} (industry: {eval_data['industry']}), identify 1-2 positive responsible production actions NOT included in standard metrics (aligned with evaluation 'Others' category).
    
    Current Data:
    - Energy: {eval_data['12_2']['renewable_share']}% renewable, {eval_data['12_2']['recycled_water_ratio']}% recycled water
    - Waste: {eval_data['12_3_4']['hazardous_recovery_pct']}% hazardous recovery, {eval_data['12_5_6']['recycling_rate_pct']}% recycling rate
    - Suppliers: {eval_data['12_7']['esg_audited_suppliers_pct']}% ESG-audited
    - Third-party news: {eval_data['third_party']['positive_news'][:200]}...
    
    Requirements:
    1. Industry-relevant (e.g., manufacturing: solar panel installation; textiles: water recycling).
    2. Clear environmental benefits tied to responsible production goals.
    3. No overlap with standard metrics.
    Return as bullet points (max 2). No extra text."""
    
    response = get_ai_response(prompt, "Sustainability consultant specializing in responsible production evaluations")
    return response.strip() if response else "- Implemented employee training on responsible production practices\n- Partnered with local recyclers for by-product reuse"

def calculate_evaluation_scores(eval_data):
    """Calculate scores per evaluation metrics and rating."""
    scores = {metric: 0 for metric in METRIC_MAX_SCORES.keys()}
    
    # Score metric group 12.2
    for field, desc, points, threshold in METRIC_CRITERIA["12.2"]:
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
    
    # Score metric group 12.3
    for field, desc, points, threshold in METRIC_CRITERIA["12.3"]:
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
    
    # Score metric group 12.4 (includes third-party penalty data)
    for field, desc, points, threshold in METRIC_CRITERIA["12.4"]:
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
    
    # Score metric group 12.5
    for field, desc, points, threshold in METRIC_CRITERIA["12.5"]:
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
    
    # Score metric group 12.6
    for field, desc, points, threshold in METRIC_CRITERIA["12.6"]:
        value = eval_data["12_5_6"][field]
        if value:
            scores["12.6"] += points
    
    # Score metric group 12.7
    for field, desc, points, threshold in METRIC_CRITERIA["12.7"]:
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
    
    # Score "Others" category
    scores["Others"] = min(10, len([l for l in eval_data["other_positive_actions"].split("\n") if l.strip()]) * 5)
    
    # Apply score caps/floors
    for metric in scores:
        scores[metric] = max(0, min(scores[metric], METRIC_MAX_SCORES[metric]))
    
    # Calculate overall rating
    overall = sum(scores.values())
    rating = "High Responsibility Enterprise (Low Risk)" if overall >=75 else \
             "Compliant but Requires Improvement (Moderate Risk)" if 60<=overall<75 else \
             "Potential Environmental Risk (High Risk)" if 40<=overall<60 else \
             "High Ethical Risk (Severe Risk)"
    
    return scores, overall, rating

def generate_improvement_recommendations(eval_data, target_scores, overall_score):
    """Generate detailed improvement recommendations (≥100 words each, no numbering)."""
    if not OPENAI_AVAILABLE:
        return [
            "Invest $250,000 in a closed-loop water recycling system (e.g., XYZ Water Technologies) to be installed by Q3 2025, increasing recycled water ratio from current {eval_data['12_2']['recycled_water_ratio'] or '45'}% to ≥70%. The system will process 50,000 liters of wastewater daily, reducing freshwater intake by 30% and cutting operational costs by $15,000 annually. Train 10 on-site technicians via ABC Environmental Training Services to maintain the system, with monthly efficiency monitoring using IoT sensors. This action enhances resource efficiency, aligns with responsible production goals, and improves performance in the energy/resource management metric group.",
            "Partner with a third-party ESG auditor (e.g., SGS or Bureau Veritas) by Q1 2025 to audit 100% of suppliers, aiming for ≥80% ESG-audited suppliers by end-2025 (current: {eval_data['12_7']['esg_audited_suppliers_pct'] or '55'}%). Allocate $120,000 for auditor fees and supplier capacity-building workshops, focusing on high-emission suppliers in Southeast Asia and Latin America. Develop a supplier scorecard tracking carbon footprint, waste management, and labor practices, with quarterly progress reports published publicly. This strengthens supply chain responsibility and improves performance in the supplier management metric group.",
            "Implement a digital loss-tracking system (e.g., SAP Sustainability or IBM Envizi) by Q2 2025 to address the lack of formal material loss monitoring. Invest $80,000 in software licenses and employee training, focusing on 15 production managers to use the system for real-time loss identification. Set a target to reduce annual material loss by 15% in the first year (current reduction: {eval_data['12_3_4']['loss_reduction_pct'] or '8'}%), projected to save $40,000 in material costs. This action minimizes resource waste and improves performance in the loss/waste management metric group."
        ]
    
    prompt = f"""Generate 3 detailed responsible production improvement recommendations for {eval_data['company_name']} (industry: {eval_data['industry']}) aligned with evaluation standards.
    
    Current Status:
    - Metric scores (achieved/max): {json.dumps({k: f'{v}/{METRIC_MAX_SCORES[k]}' for k, v in target_scores.items()}, indent=2)}
    - Overall score: {overall_score}/100
    - Low-performing metrics: {[k for k, v in target_scores.items() if v < METRIC_MAX_SCORES[k]*0.5]}
    - Current gaps:
      - Renewable energy: {eval_data['12_2']['renewable_share']}% (needs ≥50%)
      - Recycled water: {eval_data['12_2']['recycled_water_ratio']}% (needs ≥70%)
      - ESG suppliers: {eval_data['12_7']['esg_audited_suppliers_pct']}% (needs ≥80%)
    - Penalties: {eval_data['third_party']['penalties_details'][:150]}...
    
    Recommendations Must:
    1. Focus on responsible production (not general sustainability).
    2. Be ≥100 words each, including:
       - Exact investment amounts (e.g., "$250,000").
       - Specific technologies/suppliers/auditors (e.g., "SAP Sustainability software").
       - Clear timelines (e.g., "by Q3 2025").
       - Quantifiable outcomes (e.g., "reduce waste by 15%").
       - Alignment with production processes (e.g., "real-time loss tracking").
    3. Have no numbering (no "1.", "2.").
    4. Prioritize low-performing metrics first.
    5. Tie to responsible production goals (resource efficiency, supply chain responsibility).
    
    Format as bullet points. No introduction."""
    
    response = get_ai_response(prompt, f"Sustainability consultant specializing in industrial responsible production evaluations")
    recs = [line.strip() for line in response.split("\n") if line.strip() and not line.strip()[0].isdigit()]
    # Ensure 3 recommendations
    while len(recs) < 3:
        recs.append(f"Invest $300,000 in a 2MW solar panel installation at {eval_data['industry']} facilities by Q4 2025, increasing renewable energy share from current {eval_data['12_2']['renewable_share'] or '35'}% to ≥50%. Partner with SunPower or First Solar for equipment and installation, and apply for local renewable energy tax credits to offset 20% of costs. The system will generate 3.5 million kWh annually, reducing carbon emissions by 2,800 tons and lowering energy costs by $40,000 per year. Train 5 facility engineers to monitor solar output via a cloud-based dashboard, with monthly reports integrated into production management systems. This action reduces fossil fuel reliance, aligns with responsible production goals, and improves performance in the energy/resource management metric group.")
    return recs[:3]

def generate_evaluation_report(eval_data, target_scores, overall_score, rating, recommendations):
    """Generate final evaluation report."""
    title = f"Responsible Production Evaluation Report: {eval_data['company_name']}"
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
        "### 3. Metric Performance Breakdown",
    ]
    
    # Metric score details
    for metric in target_scores:
        if metric != "Others":
            report.append(f"- **Metric Group {metric}**: {target_scores[metric]}/{METRIC_MAX_SCORES[metric]}")
    report.append(f"- **Additional Positive Actions**: {target_scores['Others']}/{METRIC_MAX_SCORES['Others']}")
    
    # Detailed performance by metric group
    report.extend([
        "",
        "### 4. Detailed Responsible Production Performance",
        "**Metric Group 12.2: Sustainable Resource Management**",
        "   - Actions: Renewable energy integration, recycled water use, recycled material sourcing",
        f"   - Score: {target_scores['12.2']}/{METRIC_MAX_SCORES['12.2']}",
        "",
        "**Metric Group 12.3: Material Waste Reduction**",
        "   - Actions: Production loss tracking, annual loss reduction initiatives",
        f"   - Score: {target_scores['12.3']}/{METRIC_MAX_SCORES['12.3']}",
        "",
        "**Metric Group 12.4: Chemical & Waste Management**",
        "   - Actions: MRSL/ZDHC compliance, hazardous waste recovery, emission testing",
        f"   - Score: {target_scores['12.4']}/{METRIC_MAX_SCORES['12.4']}",
        "",
        "**Metric Group 12.5: Waste Reduction & Recycling**",
        "   - Actions: Packaging optimization, recycling programs, sustainable product design",
        f"   - Score: {target_scores['12.5']}/{METRIC_MAX_SCORES['12.5']}",
        "",
        "**Metric Group 12.6: Transparent Reporting**",
        "   - Actions: Emission reduction goals, annual progress disclosure",
        f"   - Score: {target_scores['12.6']}/{METRIC_MAX_SCORES['12.6']}",
        "",
        "**Metric Group 12.7: Responsible Procurement**",
        "   - Actions: ESG supplier audits, supply chain transparency",
        f"   - Score: {target_scores['12.7']}/{METRIC_MAX_SCORES['12.7']}",
    ])
    
    # Additional actions and recommendations
    report.extend([
        "",
        "### 5. Additional Positive Actions",
        eval_data["other_positive_actions"] or "No additional actions identified.",
        "",
        "### 6. Actionable Improvement Recommendations",
    ])
    for rec in recommendations:
        report.append(f"- {rec}")
    
    # Data sources
    report.extend([
        "",
        "### 7. Data Sources",
        "- User-confirmed PDF extraction (responsible production/annual reports)",
        "- Third-party data: Environmental agencies, credible news outlets (links included above)",
        "- AI analysis of industry benchmarks for responsible production",
    ])
    
    return "\n".join(report)

# --- UI Functions (Purple Theme, No File Name Mentions)
def render_home_page():
    """Home page (PDF upload + manual input options)."""
    st.title("🌱 Responsible Production Evaluation Tool", anchor=False)
    st.write("Evaluate corporate performance on responsible production (Environmental Dimension of ESG)")
    
    # Fix purple UI styling (override default red)
    st.markdown(
        f"""
        <style>
        /* Button styling (purple theme) */
        button.stButton {{
            background-color: {PRIMARY_PURPLE} !important;
            color: white !important;
            border: none !important;
        }}
        button.stButton:hover {{
            background-color: {MEDIUM_PURPLE} !important;
        }}
        /* Radio button styling (purple selected state) */
        div.stRadio > div > label > div[data-baseweb="radio"]:has(input:checked) {{
            background-color: {PRIMARY_PURPLE} !important;
            border-color: {PRIMARY_PURPLE} !important;
        }}
        /* Checkbox styling (purple selected state) */
        div.stCheckbox > div > label > div[data-baseweb="checkbox"]:has(input:checked) {{
            background-color: {PRIMARY_PURPLE} !important;
            border-color: {PRIMARY_PURPLE} !important;
        }}
        /* Button text styling (ensure white) */
        button.stButton > div > p {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([2, 2], gap="medium")
    
    with col1:
        st.subheader("Option 1: Upload Responsible Production Report (PDF) – Recommended")
        if not PDF_AVAILABLE:
            st.info("⚠️ Install PyPDF2 first: 'pip install PyPDF2'")
        else:
            company_name = st.text_input(
                "Company Name (required for third-party data retrieval)",
                value=st.session_state["eval_data"]["company_name"],
                placeholder="Enter company name"
            )
            industry = st.selectbox(
                "Industry",
                ENRICHED_INDUSTRIES,
                index=ENRICHED_INDUSTRIES.index(st.session_state["eval_data"]["industry"]),
                key="industry_pdf"
            )
            uploaded_file = st.file_uploader(
                "Upload Text-Based PDF (e.g., Responsible Production/ESG Report)",
                type="pdf",
                accept_multiple_files=False
            )
            
            if uploaded_file and company_name and st.button("Extract Data from PDF", key="extract_pdf", use_container_width=True):
                with st.spinner("Extracting text + retrieving third-party data..."):
                    pdf_text = extract_full_pdf_text(uploaded_file)
                    st.session_state["pdf_extracted_text"] = pdf_text
                    
                    if OPENAI_AVAILABLE:
                        extracted_data = extract_assessment_data_from_pdf(pdf_text, company_name, industry)
                        filled_data = ai_fill_missing_metrics(extracted_data, industry)
                        st.session_state["extracted_data"] = filled_data
                    else:
                        st.session_state["extracted_data"] = {}
                        st.warning("⚠️ AI disabled – manual data confirmation required.")
                    
                    st.session_state["eval_data"]["company_name"] = company_name
                    st.session_state["eval_data"]["industry"] = industry
                    st.session_state["eval_data"]["third_party"] = get_third_party_data(company_name, industry)
                    
                    st.session_state["current_step"] = 1  # Move to PDF confirmation
                    st.rerun()
    
    with col2:
        st.subheader("Option 2: Manual Input – For PDF Failures")
        st.warning("⚠️ Use only if PDF upload/extraction fails (e.g., image-based PDFs).")
        company_name = st.text_input(
            "Company Name",
            value=st.session_state["eval_data"]["company_name"],
            placeholder="Enter company name"
        )
        industry = st.selectbox(
            "Industry",
            ENRICHED_INDUSTRIES,
            index=ENRICHED_INDUSTRIES.index("Manufacturing"),
            key="industry_manual"
        )
        
        if st.button("Start Manual Input", key="start_manual", use_container_width=True):
            st.session_state["eval_data"]["company_name"] = company_name
            st.session_state["eval_data"]["industry"] = industry
            st.session_state["eval_data"]["third_party"] = get_third_party_data(company_name, industry)
            st.session_state["current_step"] = 2  # Move to first manual input step
            st.rerun()

def step_2_energy_resources():
    """Step 2: Energy & Resource Management (manual input)."""
    st.subheader("Step 2/5: Energy & Resource Management", anchor=False)
    eval_data = st.session_state["eval_data"]
    
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.caption("Energy Use")
        eval_data["12_2"]["renewable_share"] = st.number_input(
            "Renewable energy share (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_2"]["renewable_share"] or 0,
            help="Percentage of energy from renewable sources (e.g., solar, wind)"
        )
        eval_data["12_2"]["energy_retrofit"] = st.radio(
            "Full-scale energy retrofit completed?",
            ["Yes", "No"],
            index=0 if eval_data["12_2"]["energy_retrofit"] else 1,
            help="Has the company completed a full-scale energy efficiency retrofit?"
        ) == "Yes"
        eval_data["12_2"]["energy_increase"] = st.radio(
            "Energy consumption up 2 consecutive years?",
            ["Yes", "No"],
            index=1 if eval_data["12_2"]["energy_increase"] else 0,
            help="Has energy consumption increased for 2 consecutive years?"
        ) == "Yes"
    
    with col2:
        st.caption("Water & Materials")
        eval_data["12_2"]["recycled_water_ratio"] = st.number_input(
            "Recycled water ratio (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_2"]["recycled_water_ratio"] or 0,
            help="Percentage of water recycled in production processes"
        )
        eval_data["12_2"]["recycled_materials_pct"] = st.number_input(
            "Recycled materials share (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_2"]["recycled_materials_pct"] or 0,
            help="Percentage of materials sourced from recycled content"
        )
        eval_data["12_2"]["ghg_disclosure"] = st.radio(
            "Scope 1-3 GHG disclosed + third-party verified?",
            ["Yes", "No"],
            index=0 if eval_data["12_2"]["ghg_disclosure"] else 1,
            help="Has the company disclosed Scope 1-3 GHG emissions with third-party verification?"
        ) == "Yes"
    
    # Navigation buttons
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Back to Home", key="back_step2", use_container_width=True):
            st.session_state["current_step"] = 0
            st.rerun()
    with col2_btn:
        if st.button("Proceed to Waste Management", key="proceed_step2", use_container_width=True):
            st.session_state["current_step"] = 3
            st.rerun()

def step_3_waste_chemicals():
    """Step 3: Waste & Chemical Management (manual input)."""
    st.subheader("Step 3/5: Waste & Chemical Management", anchor=False)
    eval_data = st.session_state["eval_data"]
    
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.caption("Material Loss Control")
        eval_data["12_3_4"]["loss_tracking_system"] = st.radio(
            "Material loss tracking system in place?",
            ["Yes", "No"],
            index=0 if eval_data["12_3_4"]["loss_tracking_system"] else 1,
            help="Does the company have a formal system to track material loss?"
        ) == "Yes"
        eval_data["12_3_4"]["loss_reduction_pct"] = st.number_input(
            "Annual material loss reduction (%)",
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
            help="Is the company compliant with MRSL/ZDHC chemical management standards?"
        ) == "Yes"
        eval_data["12_3_4"]["hazardous_recovery_pct"] = st.number_input(
            "Hazardous waste recovery (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_3_4"]["hazardous_recovery_pct"] or 0,
            help="Percentage of hazardous waste recovered and properly disposed"
        )
    
    # Navigation buttons
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Back to Energy Management", key="back_step3", use_container_width=True):
            st.session_state["current_step"] = 2
            st.rerun()
    with col2_btn:
        if st.button("Proceed to Packaging & Reporting", key="proceed_step3", use_container_width=True):
            st.session_state["current_step"] = 4
            st.rerun()

def step_4_packaging_reporting():
    """Step 4: Packaging & Reporting (manual input)."""
    st.subheader("Step 4/5: Packaging & Reporting", anchor=False)
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
            "Products with sustainable materials (%)",
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
            help="Does the company have clear emission reduction goals for 2030/2050?"
        ) == "Yes"
        eval_data["12_5_6"]["annual_progress_disclosed"] = st.radio(
            "Annual progress published?",
            ["Yes", "No"],
            index=0 if eval_data["12_5_6"]["annual_progress_disclosed"] else 1,
            help="Does the company publicly disclose annual responsible production progress?"
        ) == "Yes"
    
    # Navigation buttons
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Back to Waste Management", key="back_step4", use_container_width=True):
            st.session_state["current_step"] = 3
            st.rerun()
    with col2_btn:
        if st.button("Proceed to Supplier Management", key="proceed_step4", use_container_width=True):
            st.session_state["current_step"] = 5
            st.rerun()

def step_5_supplier_procurement():
    """Step 5: Supplier & Procurement (manual input)."""
    st.subheader("Step 5/5: Supplier & Procurement", anchor=False)
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
            help="Has the company published a supply chain transparency report?"
        ) == "Yes"
    
    with col2:
        eval_data["12_7"]["price_only_procurement"] = st.radio(
            "Price-only procurement or high-emission outsourcing?",
            ["Yes", "No"],
            index=1 if eval_data["12_7"]["price_only_procurement"] else 0,
            help="Does the company prioritize price over responsible production in procurement?"
        ) == "Yes"
        st.caption("Third-Party Procurement Alerts")
        st.info(f"Policy Updates: {eval_data['third_party']['policy_updates'][:150]}...")
    
    # Navigation buttons
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Back to Packaging & Reporting", key="back_step5", use_container_width=True):
            st.session_state["current_step"] = 4
            st.rerun()
    with col2_btn:
        if st.button("Proceed to Additional Notes", key="proceed_step5", use_container_width=True):
            st.session_state["current_step"] = 6
            st.rerun()

def step_6_additional_notes():
    """Step 6: Additional Notes (final input step)."""
    st.subheader("Step 6/6: Additional Notes", anchor=False)
    eval_data = st.session_state["eval_data"]
    
    eval_data["additional_notes"] = st.text_area(
        "Enter additional details (e.g., ongoing projects, future plans)",
        value=eval_data["additional_notes"],
        height=150,
        help="Examples: 'Installing 10MW wind farm in 2025', 'Targeting 100% ESG suppliers by 2026'"
    )
    
    if st.button("Generate Final Evaluation Report", key="generate_report", use_container_width=True):
        with st.spinner("Calculating scores + generating report..."):
            target_scores, overall_score, rating = calculate_evaluation_scores(eval_data)
            eval_data["target_scores"] = target_scores
            eval_data["overall_score"] = overall_score
            eval_data["rating"] = rating
            eval_data["other_positive_actions"] = ai_identify_additional_actions(eval_data)
            recommendations = generate_improvement_recommendations(eval_data, target_scores, overall_score)
            st.session_state["report_text"] = generate_evaluation_report(eval_data, target_scores, overall_score, rating, recommendations)
            st.session_state["current_step"] = 7  # Move to report page
            st.rerun()
    
    if st.button("Back", key="back_step6", use_container_width=True):
        if st.session_state["extracted_data"]:
            st.session_state["current_step"] = 1
        else:
            st.session_state["current_step"] = 5
        st.rerun()

def render_report_page():
    """Final evaluation report page (tabs for compact layout)."""
    eval_data = st.session_state["eval_data"]
    st.title("Responsible Production Evaluation Report", anchor=False)

    # Tabs to organize content (reduce vertical space)
    tab1, tab2, tab3 = st.tabs(["Metrics & Chart", "Detailed Report", "Insights & Recommendations"])

    with tab1:
        # Rating card (purple theme)
        rating_colors = {
            "High Responsibility Enterprise (Low Risk)": PRIMARY_PURPLE,
            "Compliant but Requires Improvement (Moderate Risk)": MEDIUM_PURPLE,
            "Potential Environmental Risk (High Risk)": "#FFA500",
            "High Ethical Risk (Severe Risk)": "#DC143C"
        }
        st.markdown(
            f"""
            <div style="background-color:{rating_colors[eval_data['rating']]}; color:white; padding:20px; border-radius:10px; margin-bottom:30px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h2 style="margin-top:0;">Overall Rating</h2>
            <h3>{eval_data['rating']}</h3>
            <h4 style="font-size:1.5em;">Total Score: {eval_data['overall_score']}/100</h4>
            <p><strong>Industry:</strong> {eval_data['industry']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Only retained chart: Achieved vs Maximum Score
        st.subheader("Metric Performance: Achieved vs Maximum Score")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare chart data (exclude "Others" for clarity)
        metrics = [m for m in eval_data["target_scores"] if m != "Others"]
        achieved = [eval_data["target_scores"][m] for m in metrics]
        max_scores = [METRIC_MAX_SCORES[m] for m in metrics]
        
        # Plot maximum scores first (background)
        x = range(len(metrics))
        width = 0.6
        ax.bar(x, max_scores, width, label="Maximum Possible Score", color="#e0e0e0", alpha=0.8, zorder=1)
        # Plot achieved scores (foreground)
        ax.bar(x, achieved, width, label="Achieved Score", color=PRIMARY_PURPLE, zorder=2)
        
        # Chart styling
        ax.set_xlabel("Metric Groups", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title("Responsible Production Metric Performance", fontsize=14, fontweight="bold", pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.legend(loc="upper right", fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        # Add score labels
        for i, (a, m) in enumerate(zip(achieved, max_scores)):
            ax.text(i, a + 0.5, f"{a}", ha="center", va="bottom", fontsize=9, fontweight="bold", zorder=3)
            ax.text(i, m - 1, f"Max: {m}", ha="center", va="top", fontsize=8, color="#666", zorder=3)
        
        # Clean up chart spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(max_scores) * 1.15)
        
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        # Collapsible detailed report
        with st.expander("View Detailed Evaluation Report", expanded=False):
            st.text(st.session_state["report_text"])
        # Download button
        st.download_button(
            label="📥 Download Evaluation Report",
            data=st.session_state["report_text"],
            file_name=f"{eval_data['company_name']}_Responsible_Production_Report.txt",
            mime="text/plain",
            use_container_width=True
        )

    with tab3:
        # Strengths & weaknesses (purple-themed cards)
        col1, col2 = st.columns([1, 1], gap="medium")
        with col1:
            st.markdown(
                f"""
                <div style="background-color:{LIGHT_PURPLE}; padding:15px; border-radius:8px; border-left:4px solid {PRIMARY_PURPLE};">
                <h4 style="margin-top:0; color:{PRIMARY_PURPLE};">Top Strengths</h4>
                """,
                unsafe_allow_html=True
            )
            strengths = [m for m in eval_data["target_scores"] if m != "Others" and eval_data["target_scores"][m] >= METRIC_MAX_SCORES[m] * 0.7]
            if strengths:
                for s in strengths:
                    st.write(f"- **Metric Group {s}**: {eval_data['target_scores'][s]}/{METRIC_MAX_SCORES[s]} (Exceeds 70% of maximum)")
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
            weaknesses = [m for m in eval_data["target_scores"] if m != "Others" and eval_data["target_scores"][m] < METRIC_MAX_SCORES[m] * 0.5]
            if weaknesses:
                for w in weaknesses:
                    st.write(f"- **Metric Group {w}**: {eval_data['target_scores'][w]}/{METRIC_MAX_SCORES[w]} (Below 50% of maximum)")
            else:
                st.write("- Maintain current practices and set stretch goals (e.g., increase renewable energy to 60%)")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Collapsible recommendations
        with st.expander("View Improvement Recommendations", expanded=False):
            recommendations = generate_improvement_recommendations(eval_data, eval_data["target_scores"], eval_data["overall_score"])
            for rec in recommendations:
                st.write(f"- {rec}")

    # New evaluation button
    if st.button("Start New Evaluation", key="new_eval", use_container_width=True):
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

# --- Main UI Flow
if st.session_state["current_step"] == 0:
    render_home_page()
elif st.session_state["current_step"] == 1:
    render_pdf_confirmation_page(
        st.session_state["extracted_data"],
        st.session_state["eval_data"]["company_name"],
        st.session_state["eval_data"]["industry"]
    )
elif st.session_state["current_step"] == 2:
    step_2_energy_resources()
elif st.session_state["current_step"] == 3:
    step_3_waste_chemicals()
elif st.session_state["current_step"] == 4:
    step_4_packaging_reporting()
elif st.session_state["current_step"] == 5:
    step_5_supplier_procurement()
elif st.session_state["current_step"] == 6:
    step_6_additional_notes()
elif st.session_state["current_step"] == 7:
    render_report_page()

# --- Progress Indicator (Manual Input Flow)
if 2 <= st.session_state["current_step"] <= 6 and not st.session_state["extracted_data"]:
    step_names = ["", "", "Energy/Resources", "Waste/Chemicals", "Packaging/Reporting", "Suppliers", "Notes"]
    current_step = st.session_state["current_step"]
    st.sidebar.progress((current_step - 1) / 6)
    st.sidebar.write(f"Current Step: {current_step}/6 – {step_names[current_step]}")
    st.sidebar.subheader("Evaluation Focus Areas")
    st.sidebar.write("• Resource efficiency")
    st.sidebar.write("• Waste reduction")
    st.sidebar.write("• Ethical procurement")
