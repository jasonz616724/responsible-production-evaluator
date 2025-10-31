import streamlit as st
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import io

# --- 1. Third-Party Data (Per "scoring tool.docx") ---
def get_third_party_data(company_name, industry):
    if not company_name or not OPENAI_AVAILABLE:
        return {
            "penalties": False, 
            "penalties_details": "No third-party data fetched (missing company name or AI key)",
            "positive_news": "No third-party data fetched",
            "policy_updates": "No third-party data fetched"
        }
    
    prompt = f"""For {company_name} (industry: {industry}), extract ONLY the following per "scoring tool.docx" requirements:
    1. Environmental penalties (2023-2024): Violations related to sustainable resource use, waste management, or procurement (e.g., illegal disposal). Include authority/date if available.
    2. Positive news (2023-2024): Unlisted sustainability actions (e.g., recycling partnerships).
    3. Policy updates (2023-2024): Regional laws affecting sustainable production (e.g., extended producer responsibility).
    
    Sources: Environmental agencies, Bloomberg Green, Reuters Sustainability.
    Return ONLY valid JSON with keys: penalties (bool), penalties_details (str), positive_news (str), policy_updates (str). Use "No relevant data found" for empty fields."""
    
    response = get_ai_response(prompt, "Environmental data analyst specializing in 'scoring tool.docx' criteria.")
    try:
        data = json.loads(response) if response else {}
        return {
            "penalties": data.get("penalties", False),
            "penalties_details": data.get("penalties_details", "No relevant data found"),
            "positive_news": data.get("positive_news", "No relevant data found"),
            "policy_updates": data.get("policy_updates", "No relevant data found")
        }
    except json.JSONDecodeError:
        return {
            "penalties": False,
            "penalties_details": f"Invalid response: {response[:100]}...",
            "positive_news": "No valid third-party data found",
            "policy_updates": "No valid third-party data found"
        }

# --- 2. PDF Handling (Enhanced Extraction + User Confirmation) ---
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
        st.error("❌ Insufficient text for extraction (per 'scoring tool.docx'). Use a longer PDF.")
        return {}
    
    prompt = f"""Extract sustainability data from this PDF for {company_name} (industry: {industry}), per "scoring tool.docx" requirements.
    
    PDF Text (first 10,000 characters):
    {pdf_text[:10000]}
    
    REQUIRED DATA (map to "scoring tool.docx" criteria):
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
    
    response = get_ai_response(prompt, f"ESG extractor trained on 'scoring tool.docx'")
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
    """AI fills missing fields (per "scoring tool.docx" industry norms)"""
    if not OPENAI_AVAILABLE or not extracted_data:
        return extracted_data
    
    missing_fields = [k for k, v in extracted_data.items() if v is None]
    if not missing_fields:
        return extracted_data
    
    prompt = f"""For a {industry} company, fill missing sustainability data (per "scoring tool.docx" industry benchmarks).
    Extracted data (missing fields: {missing_fields}):
    {json.dumps(extracted_data, indent=2)}
    
    Rules:
    1. Use realistic {industry} averages (e.g., manufacturing: 35% renewable energy; textiles: 25% recycled materials).
    2. For booleans: Assume "False" for high-risk criteria (e.g., illegal_logging = False) and "True" for common practices (e.g., regular_emission_tests = True).
    3. Preserve existing non-null values.
    4. Return ONLY updated JSON. No extra text."""
    
    response = get_ai_response(prompt, f"Sustainability data analyst for {industry} sector.")
    try:
        return json.loads(response) if response else extracted_data
    except:
        st.warning("⚠️ AI could not fill missing fields. Using original extracted data.")
        return extracted_data

def render_pdf_confirmation_page(extracted_data, company_name, industry):
    """User confirms/edits extracted data before proceeding"""
    st.subheader(f"Review & Confirm Extracted Data (Company: {company_name})")
    st.write("Edit fields as needed (based on your report). Fields marked * are auto-filled by AI (per 'scoring tool.docx').")
    
    # Organize fields into action-based groups (no SDG mentions)
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
    
    # Initialize confirmed data with extracted values
    confirmed_data = extracted_data.copy()
    
    # Render each group with editable inputs
    for group_name, fields in field_groups.items():
        st.subheader(f"• {group_name}")
        col1, col2 = st.columns([1, 1], gap="small")
        for i, (field, label, field_type) in enumerate(fields):
            # Alternate columns for readability
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
    
    # Buttons: Confirm or Re-extract
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Confirm Data & Proceed"):
            # Update session state with confirmed data
            eval_data = st.session_state["eval_data"]
            # Map confirmed data to eval_data structure
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
            # Move to final notes (no manual steps needed if data is confirmed)
            st.session_state["current_step"] = 6
            st.rerun()
    
    with col2_btn:
        if st.button("Re-Extract from PDF"):
            st.session_state["current_step"] = 0  # Back to front page to re-upload
            st.rerun()

# --- 3. Page Config & OpenAI Client ---
st.set_page_config(page_title="SDG 12 Environmental Evaluator", layout="wide")

try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except KeyError:
    st.warning("⚠️ OPENAI_API_KEY missing. AI features (extraction, recommendations) disabled.")
    OPENAI_AVAILABLE = False
except Exception as e:
    st.error(f"⚠️ OpenAI Error: {str(e)}")
    OPENAI_AVAILABLE = False

# --- 4. Session State ---
if "eval_data" not in st.session_state:
    st.session_state["eval_data"] = {
        "company_name": "",
        "industry": "Manufacturing",
        "third_party": {
            "penalties": False, "penalties_details": "", "positive_news": "", "policy_updates": ""
        },
        # Action-based groups (no SDG labels)
        "12_2": {  # Energy & Material Management
            "renewable_share": None, "energy_retrofit": False, "energy_increase": False,
            "carbon_offsets_only": False, "recycled_water_ratio": None, "ghg_disclosure": False,
            "recycled_materials_pct": None, "illegal_logging": False
        },
        "12_3_4": {  # Loss & Chemical/Waste Management
            "loss_tracking_system": False, "loss_reduction_pct": None,
            "mrsl_zdhc_compliance": False, "regular_emission_tests": False,
            "hazardous_recovery_pct": None, "illegal_disposal": False
        },
        "12_5_6": {  # Packaging & Reporting
            "packaging_reduction_pct": None, "recycling_rate_pct": None,
            "sustainable_products_pct": None, "waste_disclosure_audit": False,
            "emission_plans": False, "annual_progress_disclosed": False, "no_goals": False,
            "high_carbon_assets_disclosed": False
        },
        "12_7": {  # Supplier & Procurement
            "esg_audited_suppliers_pct": None, "price_only_procurement": False,
            "supply_chain_transparency": False
        },
        "additional_notes": "",
        "target_scores": {}, "overall_score": 0, "rating": "", "other_positive_actions": ""
    }
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0  # 0: Front, 1: PDF Confirmation, 2-6: Manual Steps, 7: Report
if "pdf_extracted_text" not in st.session_state:
    st.session_state["pdf_extracted_text"] = ""
if "extracted_data" not in st.session_state:
    st.session_state["extracted_data"] = {}

# --- 5. Constants (Per "scoring tool.docx") ---
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

# --- 6. Core AI Functions ---
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

# FIXED: SyntaxError – Terminated string literal correctly
def identify_missing_fields(eval_data):
    """Identify missing data per "scoring tool.docx"""
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
        return "- Implemented employee training on sustainable production\n- Partnered with local recyclers for by-product reuse"
    
    prompt = f"""For {eval_data['company_name']} (industry: {eval_data['industry']}), identify 1-2 positive sustainability actions NOT listed in standard criteria (per "scoring tool.docx" "Others" category).
    
    Current data:
    - Energy: {eval_data['12_2']['renewable_share']}% renewable, {eval_data['12_2']['recycled_water_ratio']}% recycled water
    - Waste: {eval_data['12_3_4']['hazardous_recovery_pct']}% hazardous recovery, {eval_data['12_5_6']['recycling_rate_pct']}% recycling rate
    - Suppliers: {eval_data['12_7']['esg_audited_suppliers_pct']}% ESG-audited
    - Third-party news: {eval_data['third_party']['positive_news'][:200]}...
    
    Actions must be industry-relevant and tie to SDG 12. Return as bullet points (max 2)."""
    
    response = get_ai_response(prompt, f"Sustainability consultant specializing in 'scoring tool.docx'")
    return response.strip() if response else "- Implemented employee training on sustainable production\n- Partnered with local recyclers for by-product reuse"

# --- 7. Scoring Function ---
def calculate_scores(eval_data):
    scores = {sdg: 0 for sdg in SDG_MAX_SCORES.keys()}
    
    # SDG 12.2
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
    
    # SDG 12.3
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
    
    # SDG 12.4
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
    
    # SDG 12.5
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
    
    # SDG 12.6
    for field, desc, points, threshold in SDG_CRITERIA["12.6"]:
        value = eval_data["12_5_6"][field]
        if value:
            scores["12.6"] += points
    
    # SDG 12.7
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
    
    # "Others" Category
    scores["Others"] = min(10, len([l for l in eval_data["other_positive_actions"].split("\n") if l.strip()]) * 5)
    
    # Apply caps/floors
    for sdg in scores:
        scores[sdg] = max(0, min(scores[sdg], SDG_MAX_SCORES[sdg]))
    
    # Overall Rating
    overall = sum(scores.values())
    rating = "High Responsibility Enterprise (Low Risk)" if overall >=75 else \
             "Compliant but Requires Improvement (Moderate Risk)" if 60<=overall<75 else \
             "Potential Environmental Risk (High Risk)" if 40<=overall<60 else \
             "High Ethical Risk (Severe Risk)"
    
    return scores, overall, rating

# --- 8. Recommendations ---
def generate_recommendations(eval_data, target_scores, overall_score):
    if not OPENAI_AVAILABLE:
        return [
            f"Increase renewable energy to ≥50% (current: {eval_data['12_2']['renewable_share'] or 'Unknown'}%)—install 5MW solar panels by Q2 2025 (gains +7 points per 'scoring tool.docx').",
            f"Boost hazardous waste recovery to ≥90% (current: {eval_data['12_3_4']['hazardous_recovery_pct'] or 'Unknown'}%)—partner with a certified handler by Q1 2025 (gains +5 points).",
            f"Audit ≥80% of suppliers for ESG compliance (current: {eval_data['12_7']['esg_audited_suppliers_pct'] or 'Unknown'}%)—hire an auditor by Q3 2024 (gains +7 points)."
        ]
    
    prompt = f"""Generate 3 detailed recommendations for {eval_data['company_name']} (industry: {eval_data['industry']}) to improve sustainability scores (per "scoring tool.docx").
    
    Current Status:
    - Target scores (achieved/max): {json.dumps({k: f'{v}/{SDG_MAX_SCORES[k]}' for k, v in target_scores.items()}, indent=2)}
    - Overall score: {overall_score}/100
    - Low targets: {[k for k, v in target_scores.items() if v < SDG_MAX_SCORES[k]*0.5]}
    - Penalties: {eval_data['third_party']['penalties_details'][:150]}...
    
    Recommendations must:
    1. Link to "scoring tool.docx" criteria (e.g., "X to Y% for +Z points").
    2. Be industry-specific (e.g., manufacturing: solar panels; food: packaging reduction).
    3. Include time-bound steps and resources.
    4. Prioritize low-scoring areas first.
    
    Format as numbered bullets. No extra text."""
    
    response = get_ai_response(prompt, f"Sustainability consultant trained on 'scoring tool.docx'")
    recs = [l.strip() for l in response.split("\n") if l.strip() and l[0].isdigit()]
    while len(recs) < 3:
        recs.append(f"Increase recycled materials to ≥30% (current: {eval_data['12_2']['recycled_materials_pct'] or 'Unknown'}%)—source from {eval_data['industry']} recyclers by Q1 2025 (gains +5 points per 'scoring tool.docx').")
    return recs[:3]

# --- 9. Report Generation (Reveals SDG Links) ---
def generate_report(eval_data, target_scores, overall_score, rating, recommendations):
    report = [
        f"Sustainability Performance Report: {eval_data['company_name']}",
        "=" * len(report[0]),
        f"\nPrepared per 'scoring tool.docx' (KPMG 2024 & IFRS 2022)",
        "",
        "1. Executive Summary",
        f"- Industry: {eval_data['industry']}",
        f"- Overall Score: {overall_score}/100",
        f"- Overall Rating: {rating}",
        f"- Additional Notes: {eval_data['additional_notes'] or 'No additional notes provided'}",
        "",
        "2. Third-Party Data (Per 'scoring tool.docx')",
        f"- Environmental Penalties: {eval_data['third_party']['penalties_details']}",
        f"- Positive Sustainability News: {eval_data['third_party']['positive_news']}",
        f"- Relevant Policy Updates: {eval_data['third_party']['policy_updates']}",
        "",
        "3. Target-Wise Score Breakdown (SDG 12.2-12.7)",
    ]
    
    for sdg in target_scores:
        report.append(f"- SDG {sdg}: {target_scores[sdg]}/{SDG_MAX_SCORES[sdg]}")
    
    # Detailed Performance (Reveals SDG-action links)
    report.extend([
        "",
        "4. Detailed Performance (Tied to SDG 12 Targets)",
        "   SDG 12.2: Sustainable Management of Natural Resources",
        "   - Actions: Renewable energy, recycled water, recycled materials",
        f"   - Score: {target_scores['12.2']}/{SDG_MAX_SCORES['12.2']}",
        "",
        "   SDG 12.3: Reduce Food & Material Waste",
        "   - Actions: Loss-tracking systems, loss reduction",
        f"   - Score: {target_scores['12.3']}/{SDG_MAX_SCORES['12.3']}",
        "",
        "   SDG 12.4: Sound Chemical & Waste Management",
        "   - Actions: MRSL/ZDHC compliance, hazardous waste recovery",
        f"   - Score: {target_scores['12.4']}/{SDG_MAX_SCORES['12.4']}",
        "",
        "   SDG 12.5: Reduce, Reuse, Recycle Waste",
        "   - Actions: Packaging reduction, recycling rate, sustainable products",
        f"   - Score: {target_scores['12.5']}/{SDG_MAX_SCORES['12.5']}",
        "",
        "   SDG 12.6: Promote Sustainable Practices",
        "   - Actions: Emission goals, annual progress disclosure",
        f"   - Score: {target_scores['12.6']}/{SDG_MAX_SCORES['12.6']}",
        "",
        "   SDG 12.7: Sustainable Procurement",
        "   - Actions: ESG-audited suppliers, supply chain transparency",
        f"   - Score: {target_scores['12.7']}/{SDG_MAX_SCORES['12.7']}",
    ])
    
    # "Others" Category
    report.extend([
        "",
        "5. Additional Positive Actions (SDG 12 Alignment)",
        eval_data["other_positive_actions"] or "No additional actions identified.",
        "",
        "6. Actionable Recommendations",
    ])
    for i, rec in enumerate(recommendations, 1):
        report.append(f"   {i}. {rec}")
    
    # Data Sources
    report.extend([
        "",
        "7. Data Sources",
        "- User-confirmed PDF extraction (per 'scoring tool.docx')",
        "- Third-party data (environmental agencies, credible news)",
        "- AI analysis aligned with 'scoring tool.docx' criteria",
    ])
    
    return "\n".join(report)

# --- 10. UI Functions (Action-Based Labels, No SDG Mentions) ---
def render_front_page():
    st.title("🌱 Sustainable Production Evaluator")
    st.write("Evaluate performance per **'scoring tool.docx'** (Environmental Dimension of ESG)")
    
    # Aligned columns for PDF upload (priority) and manual input
    col1, col2 = st.columns([1.2, 0.8], gap="medium")  # PDF column wider (priority)
    
    with col1:
        st.subheader("Option 1: Upload ESG/Annual Report (PDF) – Recommended")
        if not PDF_AVAILABLE:
            st.info("⚠️ Install PyPDF2 first: pip install PyPDF2")
        else:
            company_name = st.text_input(
                "Company Name (required for extraction/third-party data)",
                value=st.session_state["eval_data"]["company_name"]
            )
            industry = st.selectbox(
                "Industry",
                ["Manufacturing", "Food & Beverage", "Textiles", "Chemicals", "Electronics", "Other"],
                index=["Manufacturing", "Food & Beverage", "Textiles", "Chemicals", "Electronics", "Other"].index(
                    st.session_state["eval_data"]["industry"]
                )
            )
            uploaded_file = st.file_uploader(
                "Upload Text-Based PDF (e.g., ESG report)",
                type="pdf",
                help="Auto-extracts data per 'scoring tool.docx' – no manual input needed"
            )
            
            if uploaded_file and company_name and st.button("Extract Data from PDF"):
                with st.spinner("Extracting text + fetching third-party data..."):
                    # Extract PDF text
                    pdf_text = extract_full_pdf_text(uploaded_file)
                    st.session_state["pdf_extracted_text"] = pdf_text
                    
                    # Extract SDG data
                    if OPENAI_AVAILABLE:
                        extracted_data = extract_sdg_data_from_pdf(pdf_text, company_name, industry)
                        # AI fills missing fields
                        filled_data = ai_fill_missing_fields(extracted_data, industry)
                        st.session_state["extracted_data"] = filled_data
                    else:
                        st.session_state["extracted_data"] = {}
                        st.warning("⚠️ AI disabled – manual confirmation will have empty fields.")
                    
                    # Update company/industry in session state
                    st.session_state["eval_data"]["company_name"] = company_name
                    st.session_state["eval_data"]["industry"] = industry
                    # Fetch third-party data
                    st.session_state["eval_data"]["third_party"] = get_third_party_data(company_name, industry)
                    
                    # Move to confirmation page
                    st.session_state["current_step"] = 1
                    st.rerun()
    
    with col2:
        st.subheader("Option 2: Manual Input – For PDF Failures")
        st.warning("⚠️ Use only if PDF upload/extraction fails (e.g., image-based PDFs).")
        company_name = st.text_input(
            "Company Name",
            value=st.session_state["eval_data"]["company_name"]
        )
        industry = st.selectbox(
            "Industry",
            ["Manufacturing", "Food & Beverage", "Textiles", "Chemicals", "Electronics", "Other"],
            index=0
        )
        
        if st.button("Start Manual Input"):
            st.session_state["eval_data"]["company_name"] = company_name
            st.session_state["eval_data"]["industry"] = industry
            st.session_state["eval_data"]["third_party"] = get_third_party_data(company_name, industry)
            st.session_state["current_step"] = 2  # Manual Step 1
            st.rerun()

def step_2_energy_materials():
    """Step 2: Energy & Material Management (no SDG label)"""
    st.subheader("Step 2/5: Energy & Material Management")
    eval_data = st.session_state["eval_data"]
    
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.caption("Energy Use")
        eval_data["12_2"]["renewable_share"] = st.number_input(
            "Renewable energy share (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_2"]["renewable_share"] or 0
        )
        eval_data["12_2"]["energy_retrofit"] = st.radio(
            "Full-scale energy retrofit implemented?",
            ["Yes", "No"],
            index=0 if eval_data["12_2"]["energy_retrofit"] else 1
        ) == "Yes"
        eval_data["12_2"]["energy_increase"] = st.radio(
            "Energy consumption up 2 consecutive years?",
            ["Yes", "No"],
            index=1 if eval_data["12_2"]["energy_increase"] else 0
        ) == "Yes"
    
    with col2:
        st.caption("Water & Materials")
        eval_data["12_2"]["recycled_water_ratio"] = st.number_input(
            "Recycled water ratio (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_2"]["recycled_water_ratio"] or 0
        )
        eval_data["12_2"]["recycled_materials_pct"] = st.number_input(
            "Recycled materials share (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_2"]["recycled_materials_pct"] or 0
        )
        eval_data["12_2"]["ghg_disclosure"] = st.radio(
            "Scope1-3 GHG disclosed + third-party verified?",
            ["Yes", "No"],
            index=0 if eval_data["12_2"]["ghg_disclosure"] else 1
        ) == "Yes"
    
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Back"):
            st.session_state["current_step"] = 0
            st.rerun()
    with col2_btn:
        if st.button("Proceed to Waste Management"):
            st.session_state["current_step"] = 3
            st.rerun()

def step_3_waste_chemicals():
    """Step 3: Waste & Chemical Management"""
    st.subheader("Step 3/5: Waste & Chemical Management")
    eval_data = st.session_state["eval_data"]
    
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.caption("Material Loss Control")
        eval_data["12_3_4"]["loss_tracking_system"] = st.radio(
            "Loss-tracking system established?",
            ["Yes", "No"],
            index=0 if eval_data["12_3_4"]["loss_tracking_system"] else 1
        ) == "Yes"
        eval_data["12_3_4"]["loss_reduction_pct"] = st.number_input(
            "Annual loss reduction (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_3_4"]["loss_reduction_pct"] or 0
        )
    
    with col2:
        st.caption("Chemical & Hazardous Waste")
        eval_data["12_3_4"]["mrsl_zdhc_compliance"] = st.radio(
            "Compliant with MRSL/ZDHC standards?",
            ["Yes", "No"],
            index=0 if eval_data["12_3_4"]["mrsl_zdhc_compliance"] else 1
        ) == "Yes"
        eval_data["12_3_4"]["hazardous_recovery_pct"] = st.number_input(
            "Hazardous waste recovery (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_3_4"]["hazardous_recovery_pct"] or 0
        )
    
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Back to Energy Management"):
            st.session_state["current_step"] = 2
            st.rerun()
    with col2_btn:
        if st.button("Proceed to Packaging & Reporting"):
            st.session_state["current_step"] = 4
            st.rerun()

def step_4_packaging_reporting():
    """Step 4: Packaging & Reporting"""
    st.subheader("Step 4/5: Packaging & Sustainability Reporting")
    eval_data = st.session_state["eval_data"]
    
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.caption("Packaging & Recycling")
        eval_data["12_5_6"]["packaging_reduction_pct"] = st.number_input(
            "Packaging weight reduction (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_5_6"]["packaging_reduction_pct"] or 0
        )
        eval_data["12_5_6"]["recycling_rate_pct"] = st.number_input(
            "Overall recycling rate (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_5_6"]["recycling_rate_pct"] or 0
        )
        eval_data["12_5_6"]["sustainable_products_pct"] = st.number_input(
            "Products with sustainable materials (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_5_6"]["sustainable_products_pct"] or 0
        )
    
    with col2:
        st.caption("Sustainability Reporting")
        eval_data["12_5_6"]["emission_plans"] = st.radio(
            "Clear 2030/2050 emission reduction goals?",
            ["Yes", "No"],
            index=0 if eval_data["12_5_6"]["emission_plans"] else 1
        ) == "Yes"
        eval_data["12_5_6"]["annual_progress_disclosed"] = st.radio(
            "Annual sustainability progress disclosed?",
            ["Yes", "No"],
            index=0 if eval_data["12_5_6"]["annual_progress_disclosed"] else 1
        ) == "Yes"
    
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Back to Waste Management"):
            st.session_state["current_step"] = 3
            st.rerun()
    with col2_btn:
        if st.button("Proceed to Supplier Management"):
            st.session_state["current_step"] = 5
            st.rerun()

def step_5_supplier_procurement():
    """Step 5: Supplier & Procurement"""
    st.subheader("Step 5/5: Supplier & Procurement Practices")
    eval_data = st.session_state["eval_data"]
    
    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        eval_data["12_7"]["esg_audited_suppliers_pct"] = st.number_input(
            "ESG-audited suppliers (%)",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_7"]["esg_audited_suppliers_pct"] or 0
        )
        eval_data["12_7"]["supply_chain_transparency"] = st.radio(
            "Supply chain transparency report published?",
            ["Yes", "No"],
            index=0 if eval_data["12_7"]["supply_chain_transparency"] else 1
        ) == "Yes"
    
    with col2:
        eval_data["12_7"]["price_only_procurement"] = st.radio(
            "Price-only procurement or outsourcing to high-emission regions?",
            ["Yes", "No"],
            index=1 if eval_data["12_7"]["price_only_procurement"] else 0
        ) == "Yes"
        st.caption("Third-Party Procurement Alerts")
        st.info(f"Policy Updates: {eval_data['third_party']['policy_updates'][:150]}...")
    
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Back to Packaging & Reporting"):
            st.session_state["current_step"] = 4
            st.rerun()
    with col2_btn:
        if st.button("Proceed to Additional Notes"):
            st.session_state["current_step"] = 6
            st.rerun()

def step_6_notes():
    """Step 6: Additional Notes (All input methods)"""
    st.subheader("Step 6/6: Additional Sustainability Notes")
    eval_data = st.session_state["eval_data"]
    
    eval_data["additional_notes"] = st.text_area(
        "Enter additional details (e.g., ongoing projects, future plans)",
        value=eval_data["additional_notes"],
        height=150,
        help="Examples: 'Installing 10MW wind farm in 2025', 'Targeting 100% ESG suppliers by 2026'"
    )
    
    if st.button("Generate Final Report (per 'scoring tool.docx')"):
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
    
    if st.button("Back"):
        # Return to last step (PDF confirmation or manual step 5)
        if st.session_state["extracted_data"]:
            st.session_state["current_step"] = 1
        else:
            st.session_state["current_step"] = 5
        st.rerun()

def render_report_page():
    eval_data = st.session_state["eval_data"]
    st.title(f"Sustainability Report: {eval_data['company_name']}")
    
    # Rating Banner
    rating_colors = {
        "High Responsibility Enterprise (Low Risk)": "#4CAF50",
        "Compliant but Requires Improvement (Moderate Risk)": "#FFD700",
        "Potential Environmental Risk (High Risk)": "#FFA500",
        "High Ethical Risk (Severe Risk)": "#DC143C"
    }
    st.markdown(
        f"""
        <div style="background-color:{rating_colors[eval_data['rating']]}; color:white; padding:15px; border-radius:8px; margin-bottom:20px;">
        <h3>Overall Rating: {eval_data['rating']}</h3>
        <h4>Total Score: {eval_data['overall_score']}/100 (per 'scoring tool.docx')</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Score Visualization
    st.subheader("Score Distribution (SDG 12 Targets)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sdgs = list(eval_data["target_scores"].keys())
    scores = [eval_data["target_scores"][sdg] for sdg in sdgs]
    max_scores = [SDG_MAX_SCORES[sdg] for sdg in sdgs]
    
    x = range(len(sdgs))
    width = 0.35
    bars1 = ax.bar([i-width/2 for i in x], scores, width, label="Achieved", color="#2196F3")
    bars2 = ax.bar([i+width/2 for i in x], max_scores, width, label="Maximum", color="#f0f0f0", alpha=0.7)
    
    ax.set_xlabel("SDG 12 Targets")
    ax.set_ylabel("Score")
    ax.set_title("SDG 12.2-12.7: Achieved vs. Maximum Score")
    ax.set_xticks(x)
    ax.set_xticklabels(sdgs)
    ax.legend()
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2., height+0.1, f"{height}", ha="center", va="bottom")
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Report Text
    st.subheader("Detailed Report (per 'scoring tool.docx')")
    st.text(st.session_state["report_text"])
    
    # Download
    st.download_button(
        label="📥 Download Report",
        data=st.session_state["report_text"],
        file_name=f"{eval_data['company_name']}_Sustainability_Report.txt",
        mime="text/plain"
    )
    
    # New Evaluation
    if st.button("Start New Evaluation"):
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

# --- 11. Main UI Flow ---
if st.session_state["current_step"] == 0:
    render_front_page()
elif st.session_state["current_step"] == 1:
    # PDF Data Confirmation Page (priority flow)
    render_pdf_confirmation_page(
        st.session_state["extracted_data"],
        st.session_state["eval_data"]["company_name"],
        st.session_state["eval_data"]["industry"]
    )
elif st.session_state["current_step"] == 2:
    step_2_energy_materials()  # Manual Step 1
elif st.session_state["current_step"] == 3:
    step_3_waste_chemicals()  # Manual Step 2
elif st.session_state["current_step"] == 4:
    step_4_packaging_reporting()  # Manual Step 3
elif st.session_state["current_step"] == 5:
    step_5_supplier_procurement()  # Manual Step 4
elif st.session_state["current_step"] == 6:
    step_6_notes()  # Final Notes (all flows)
elif st.session_state["current_step"] == 7:
    render_report_page()

# Progress Indicator (Manual Flow Only)
if 2 <= st.session_state["current_step"] <= 6 and not st.session_state["extracted_data"]:
    step_names = ["", "", "Energy/Materials", "Waste/Chemicals", "Packaging/Reporting", "Suppliers", "Notes"]
    current_step_name = step_names[st.session_state["current_step"]]
    progress = (st.session_state["current_step"] - 1) / 6
    st.sidebar.progress(progress)
    st.sidebar.write(f"Current Step: {st.session_state['current_step']}/6 – {current_step_name}")
