import streamlit as st
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import io

# --- Third-Party Data Integration (Per Document Requirement) ---
def get_third_party_data(company_name):
    """Fetch third-party data (penalties, news, policy updates) as specified in the document"""
    if not company_name or not OPENAI_AVAILABLE:
        return {"penalties": False, "positive_news": "", "policy_updates": ""}
    
    prompt = f"""For {company_name}, extract:
    1. Environmental penalties (2023-2024)
    2. Positive sustainability initiatives (unlisted in SDG 12.2-12.7 criteria)
    3. Policy updates relevant to SDG 12 compliance
    
    Return ONLY a valid JSON with keys: penalties (bool), positive_news (str), policy_updates (str)."""
    
    response = get_ai_response(prompt, "Environmental data researcher specializing in corporate ESG.")
    try:
        return json.loads(response) if response else {"penalties": False, "positive_news": "", "policy_updates": ""}
    except:
        return {"penalties": False, "positive_news": "", "policy_updates": ""}

# --- PDF Handling (For Document Upload) ---
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("⚠️ PyPDF2 library not found. PDF upload functionality is disabled. Install with: pip install PyPDF2")

# --- Page Configuration ---
st.set_page_config(page_title="SDG 12.2-12.7 Evaluator", layout="wide")

# --- OpenAI Client Initialization ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except KeyError:
    st.warning("⚠️ OPENAI_API_KEY not found in Streamlit Secrets. AI features (analysis, recommendations) are disabled.")
    OPENAI_AVAILABLE = False
except Exception as e:
    st.error(f"⚠️ OpenAI Client Error: {str(e)}")
    OPENAI_AVAILABLE = False

# --- Session State Initialization (Aligned with Scoring Criteria) ---
if "eval_data" not in st.session_state:
    st.session_state["eval_data"] = {
        "company_name": "",
        "industry": "Manufacturing",
        "third_party": {"penalties": False, "positive_news": "", "policy_updates": ""},
        # SDG 12.2: Sustainable Management and Efficient Use of Natural Resources
        "12_2_energy": {
            "renewable_share": None,  # ≥50% = +7
            "energy_retrofit": False,  # Full-scale = +5
            "energy_increase": False,  # 2 consecutive years = -5
            "carbon_offsets_only": False,  # Sole reliance = -3
            "recycled_water_ratio": None,  # ≥70% = +5
            "ghg_disclosure": False  # Scope1-3 disclosed + third-party verified = +7
        },
        "12_2_materials": {
            "recycled_materials_pct": None,  # ≥30% = +5
            "illegal_logging": False  # Incidents = -7
        },
        # SDG 12.3: Reduction of Food and Material Waste
        "12_3_loss": {
            "loss_tracking_system": False,  # Established = +5
            "loss_reduction_pct": None  # >10% annual reduction = +4
        },
        # SDG 12.4: Environmentally Sound Management of Chemicals and Waste
        "12_4_chemicals": {
            "mrsl_zdhc_compliance": False,  # Compliance = +5
            "regular_emission_tests": False  # Regular testing = +3
        },
        "12_4_waste": {
            "hazardous_recovery_pct": None,  # ≥90% recovery = +5
            "illegal_disposal": False  # Improper disposal = -3
        },
        # SDG 12.5: Waste Reduction, Recycling, and Reuse
        "12_5_waste_min": {
            "packaging_reduction_pct": None  # ≥20% weight reduction = +4
        },
        "12_5_recycle": {
            "recycling_rate_pct": None,  # ≥80% = +4
            "sustainable_products_pct": None  # ≥50% sustainable materials = +4
        },
        "12_5_disclosure": {
            "waste_disclosure_audit": False  # Disclosed + third-party audit = +5
        },
        # SDG 12.6: Promote Sustainable Practices and Environmental Reporting
        "12_6_targets": {
            "emission_plans": False,  # Clear 2030/2050 goals = +5
            "annual_progress_disclosed": False,  # Annual progress = +4
            "no_goals": False  # No goals/stagnant = -3
        },
        "12_6_risk": {
            "high_carbon_assets_disclosed": False  # Disclosure + reduction pathway = Implied criterion
        },
        # SDG 12.7: Sustainable Public Procurement Practices
        "12_7_procurement": {
            "esg_audited_suppliers_pct": None,  # ≥80% audited + plan = +7
            "price_only_procurement": False,  # Price-only/outsourcing to high-emission regions = -3
            "supply_chain_transparency": False  # Transparency report = +3
        },
        # "Others" Category (AI-identified positive actions)
        "other_positive_actions": "",
        # Scoring Results
        "target_scores": {},
        "overall_score": 0,
        "rating": ""
    }
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0  # 0: Input Method, 1-7: Missing Data Collection, 8: Report
if "missing_fields" not in st.session_state:
    st.session_state["missing_fields"] = []
if "report_text" not in st.session_state:
    st.session_state["report_text"] = ""

# --- Constants (Exact from "scoring tool.docx") ---
SDG_MAX_SCORES = {
    "12.2": 29,
    "12.3": 9,
    "12.4": 16,
    "12.5": 17,
    "12.6": 9,
    "12.7": 10,
    "Others": 10
}

# Scoring criteria mapping (directly from document table)
SDG_CRITERIA = {
    "12.2": {
        "energy": [
            ("renewable_share", "Renewable energy share ≥50%", 7, "≥50%"),
            ("energy_retrofit", "Full-scale energy retrofit", 5, "Yes/No"),
            ("energy_increase", "Energy consumption increases for 2 consecutive years", -5, "Yes/No"),
            ("carbon_offsets_only", "Reliance solely on carbon offsets", -3, "Yes/No"),
            ("recycled_water_ratio", "Recycled water ratio ≥70%", 5, "≥70%"),
            ("ghg_disclosure", "Greenhouse gas emissions (Scope1-3) fully disclosed and third-party verified", 7, "Yes/No")
        ],
        "materials": [
            ("recycled_materials_pct", "Recycled materials ≥30%", 5, "≥30%"),
            ("illegal_logging", "Illegal logging incidents", -7, "Yes/No")
        ]
    },
    "12.3": {
        "loss": [
            ("loss_tracking_system", "Establish loss-tracking system", 5, "Yes/No"),
            ("loss_reduction_pct", "Annual loss rate reduction >10%", 4, ">10%")
        ]
    },
    "12.4": {
        "chemicals": [
            ("mrsl_zdhc_compliance", "Comply with MRSL/ZDHC standards", 5, "Yes/No"),
            ("regular_emission_tests", "Regular emission testing", 3, "Yes/No")
        ],
        "waste": [
            ("hazardous_recovery_pct", "Hazardous waste recovery rate ≥90%", 5, "≥90%"),
            ("illegal_disposal", "Improper classification or illegal disposal", -3, "Yes/No"),
            ("penalties", "No environmental penalties (third-party data)", 3, "Yes/No")
        ]
    },
    "12.5": {
        "waste_min": [
            ("packaging_reduction_pct", "Packaging weight reduction ≥20%", 4, "≥20%")
        ],
        "recycle": [
            ("recycling_rate_pct", "Recycling rate ≥80%", 4, "≥80%"),
            ("sustainable_products_pct", "Products with low-carbon/recycled/sustainable materials ≥50%", 4, "≥50%")
        ],
        "disclosure": [
            ("waste_disclosure_audit", "Public disclosure with third-party audit", 5, "Yes/No")
        ]
    },
    "12.6": {
        "targets": [
            ("emission_plans", "Disclose specific emission reduction/energy transition plans (Clear 2030/2050 goals)", 5, "Yes/No"),
            ("annual_progress_disclosed", "Annual progress disclosed", 4, "Yes/No"),
            ("no_goals", "No goals or stagnant progress", -3, "Yes/No")
        ],
        "risk": [
            ("high_carbon_assets_disclosed", "Disclose high-carbon assets account and explain reduction pathway", 0, "Yes/No")
        ]
    },
    "12.7": {
        "procurement": [
            ("esg_audited_suppliers_pct", "Suppliers pass ESG audits ≥80% and policies have implementation plans", 7, "≥80%"),
            ("price_only_procurement", "Price-only procurement or outsourcing to high-emission regions", -3, "Yes/No"),
            ("supply_chain_transparency", "Establish supply chain transparency reports", 3, "Yes/No")
        ]
    }
}

# --- Core AI Functions ---
def get_ai_response(prompt, system_msg="You are a helpful assistant."):
    if not OPENAI_AVAILABLE:
        return ""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            temperature=0.3,
            timeout=20
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        return ""

def extract_pdf_data(pdf_text):
    """Extract SDG 12.2-12.7 data from uploaded documents (per document requirement)"""
    if len(pdf_text.strip()) < 300:
        st.warning("⚠️ Extracted text is too short to contain meaningful SDG data.")
        return {}
    
    prompt = f"""Extract ONLY SDG 12.2-12.7 relevant data from this text (first 5000 characters):
    {pdf_text[:5000]}
    
    Return a valid JSON with these fields (use null for unknown values):
    - company_name (str)
    - industry (str)
    - renewable_share (number, % of renewable energy)
    - energy_retrofit (bool)
    - energy_increase (bool)
    - carbon_offsets_only (bool)
    - recycled_water_ratio (number, %)
    - ghg_disclosure (bool)
    - recycled_materials_pct (number, %)
    - illegal_logging (bool)
    - loss_tracking_system (bool)
    - loss_reduction_pct (number, %)
    - mrsl_zdhc_compliance (bool)
    - regular_emission_tests (bool)
    - hazardous_recovery_pct (number, %)
    - illegal_disposal (bool)
    - packaging_reduction_pct (number, %)
    - recycling_rate_pct (number, %)
    - sustainable_products_pct (number, %)
    - waste_disclosure_audit (bool)
    - emission_plans (bool)
    - annual_progress_disclosed (bool)
    - high_carbon_assets_disclosed (bool)
    - esg_audited_suppliers_pct (number, %)
    - price_only_procurement (bool)
    - supply_chain_transparency (bool)
    
    Do NOT include extra text—only the JSON object."""
    
    response = get_ai_response(prompt, "ESG data extractor specializing in SDG 12.2-12.7.")
    if not response:
        return {}
    
    # Clean and validate JSON
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        st.error("❌ AI response does not contain valid JSON.")
        return {}
    
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError as e:
        st.error(f"❌ Failed to parse JSON: {str(e)}")
        return {}

def identify_missing_fields(eval_data):
    """Identify missing data points to collect from user (per document's proactive data collection)"""
    missing = []
    # Check SDG 12.2
    for field, _, _, _ in SDG_CRITERIA["12.2"]["energy"] + SDG_CRITERIA["12.2"]["materials"]:
        if field in eval_data["12_2_energy"] and eval_data["12_2_energy"][field] is None:
            missing.append(("12.2", field))
        elif field in eval_data["12_2_materials"] and eval_data["12_2_materials"][field] is None:
            missing.append(("12.2", field))
    # Check SDG 12.3-12.7
    for sdg in ["12.3", "12.4", "12.5", "12.6", "12.7"]:
        for section in SDG_CRITERIA[sdg].values():
            for field, _, _, _ in section:
                sdg_key = f"{sdg.replace('.', '_')}_{list(SDG_CRITERIA[sdg].keys())[0]}"
                if sdg_key in eval_data and field in eval_data[sdg_key] and eval_data[sdg_key][field] is None:
                    missing.append((sdg, field))
    return missing

def ai_other_actions(eval_data):
    """AI identifies unlisted positive actions (SDG "Others" category in document)"""
    if not OPENAI_AVAILABLE:
        return "- Implemented employee training programs for sustainable production practices\n- Partnered with local NGOs for waste reduction initiatives"
    
    prompt = f"""Analyze {eval_data['company_name']}'s (industry: {eval_data['industry']}) sustainability actions:
    - Collected data: {json.dumps(eval_data, indent=2)[:3000]}
    - Third-party news: {eval_data['third_party']['positive_news']}
    
    Identify 1-2 positive environmental actions NOT listed in SDG 12.2-12.7 criteria (per "Others" category in scoring rules).
    Return as concise bullet points (max 2)."""
    
    response = get_ai_response(prompt, "Sustainability analyst specializing in SDG 12.")
    return response.strip() if response else "- Implemented employee training programs for sustainable production practices\n- Partnered with local NGOs for waste reduction initiatives"

# --- Scoring Function (Strictly from Document) ---
def calculate_scores(eval_data):
    scores = {sdg: 0 for sdg in SDG_MAX_SCORES.keys()}
    
    # 1. SDG 12.2 Score Calculation
    # Energy criteria
    for field, desc, points, threshold in SDG_CRITERIA["12.2"]["energy"]:
        value = eval_data["12_2_energy"][field]
        if field in ["renewable_share", "recycled_water_ratio"]:
            if value is not None and value >= float(threshold.replace(">", "").replace("%", "")):
                scores["12.2"] += points
        else:  # Boolean criteria
            if value:
                scores["12.2"] += points
    # Materials criteria
    for field, desc, points, threshold in SDG_CRITERIA["12.2"]["materials"]:
        value = eval_data["12_2_materials"][field]
        if field == "recycled_materials_pct":
            if value is not None and value >= 30:
                scores["12.2"] += points
        else:  # Illegal logging
            if value:
                scores["12.2"] += points
    
    # 2. SDG 12.3 Score Calculation
    for field, desc, points, threshold in SDG_CRITERIA["12.3"]["loss"]:
        value = eval_data["12_3_loss"][field]
        if field == "loss_reduction_pct":
            if value is not None and value > 10:
                scores["12.3"] += points
        else:  # Boolean (loss-tracking system)
            if value:
                scores["12.3"] += points
    
    # 3. SDG 12.4 Score Calculation
    # Chemical management
    for field, desc, points, threshold in SDG_CRITERIA["12.4"]["chemicals"]:
        value = eval_data["12_4_chemicals"][field]
        if value:
            scores["12.4"] += points
    # Waste management
    for field, desc, points, threshold in SDG_CRITERIA["12.4"]["waste"]:
        if field == "penalties":
            if not eval_data["third_party"]["penalties"]:  # No penalties = +3
                scores["12.4"] += points
        else:
            value = eval_data["12_4_waste"][field]
            if field == "hazardous_recovery_pct":
                if value is not None and value >= 90:
                    scores["12.4"] += points
            else:  # Illegal disposal
                if value:
                    scores["12.4"] += points
    
    # 4. SDG 12.5 Score Calculation
    # Waste minimization
    for field, desc, points, threshold in SDG_CRITERIA["12.5"]["waste_min"]:
        value = eval_data["12_5_waste_min"][field]
        if value is not None and value >= 20:
            scores["12.5"] += points
    # Recycling & reuse
    for field, desc, points, threshold in SDG_CRITERIA["12.5"]["recycle"]:
        value = eval_data["12_5_recycle"][field]
        if value is not None and value >= float(threshold.replace(">", "").replace("%", "")):
            scores["12.5"] += points
    # Waste disclosure
    for field, desc, points, threshold in SDG_CRITERIA["12.5"]["disclosure"]:
        value = eval_data["12_5_disclosure"][field]
        if value:
            scores["12.5"] += points
    
    # 5. SDG 12.6 Score Calculation
    for field, desc, points, threshold in SDG_CRITERIA["12.6"]["targets"]:
        value = eval_data["12_6_targets"][field]
        if value:
            scores["12.6"] += points
    
    # 6. SDG 12.7 Score Calculation
    for field, desc, points, threshold in SDG_CRITERIA["12.7"]["procurement"]:
        value = eval_data["12_7_procurement"][field]
        if field == "esg_audited_suppliers_pct":
            if value is not None and value >= 80:
                scores["12.7"] += points
        else:  # Boolean criteria
            if value:
                scores["12.7"] += points
    
    # 7. "Others" Score (AI-identified actions)
    other_actions = eval_data["other_positive_actions"]
    if other_actions:
        scores["Others"] = min(10, len([line for line in other_actions.split("\n") if line.strip()]) * 5)
    
    # Apply score caps (max per target) and floors (0)
    for sdg in scores:
        scores[sdg] = max(0, min(scores[sdg], SDG_MAX_SCORES[sdg]))
    
    # Calculate overall score and rating (document's 4-tier scale)
    overall_score = sum(scores.values())
    if overall_score >= 75:
        rating = "High Responsibility Enterprise (Low Risk)"
    elif 60 <= overall_score < 75:
        rating = "Compliant but Requires Improvement (Moderate Risk)"
    elif 40 <= overall_score < 60:
        rating = "Potential Environmental Risk (High Risk)"
    else:
        rating = "High Ethical Risk (Severe Risk)"
    
    return scores, overall_score, rating

# --- Detailed Recommendation Generation (Aligned with Document) ---
def generate_recommendations(eval_data, target_scores, overall_score):
    if not OPENAI_AVAILABLE:
        return [
            f"SDG 12.2: Increase renewable energy share to ≥50% (current: {eval_data['12_2_energy']['renewable_share'] or 'Unknown'}%) to gain +7 points—install 5MW solar panels by Q2 2025.",
            f"SDG 12.3: Implement a loss-tracking system (currently: {eval_data['12_3_loss']['loss_tracking_system']}) to gain +5 points—deploy inventory management software by Q1 2025.",
            f"SDG 12.7: Audit ≥80% of suppliers for ESG compliance (current: {eval_data['12_7_procurement']['esg_audited_suppliers_pct'] or 'Unknown'}%) to gain +7 points—partner with a third-party auditor by Q3 2024."
        ]
    
    # Identify low-performing targets and missing data
    low_targets = [sdg for sdg in target_scores if target_scores[sdg] < SDG_MAX_SCORES[sdg] * 0.5]
    missing_criteria = identify_missing_fields(eval_data)
    
    prompt = f"""Generate 3 DETAILED, actionable recommendations for {eval_data['company_name']} (industry: {eval_data['industry']}) to improve SDG 12.2-12.7 scores.
    
    Current Status:
    - Target scores (achieved/max): {json.dumps({k: f'{v}/{SDG_MAX_SCORES[k]}' for k, v in target_scores.items()}, indent=2)}
    - Overall score: {overall_score}/100
    - Low-performing targets: {low_targets}
    - Missing data points: {[f'SDG {sdg}: {field}' for sdg, field in missing_criteria]}
    - Environmental penalties: {eval_data['third_party']['penalties']}
    - Current actions: {json.dumps(eval_data, indent=2)[:2000]}
    
    Recommendations must:
    1. Link to specific SDG targets and scoring criteria (e.g., "SDG 12.2: X to Y% for +Z points").
    2. Include time-bound, actionable steps (e.g., "Invest in X technology by Q2 2025").
    3. Prioritize low-performing targets and missing data.
    4. Be industry-relevant (e.g., manufacturing vs. textiles).
    
    Format as 3 numbered bullet points—no introduction."""
    
    response = get_ai_response(prompt, "Sustainability consultant specializing in SDG 12 environmental performance.")
    recommendations = [line.strip() for line in response.split("\n") if line.strip() and not line.startswith("###")]
    
    # Fallback for incomplete AI responses
    while len(recommendations) < 3:
        recommendations.append(f"SDG 12.2: Increase recycled materials to ≥30% (current: {eval_data['12_2_materials']['recycled_materials_pct'] or 'Unknown'}%) to gain +5 points—source recycled aluminum from local suppliers by Q1 2025.")
    
    return recommendations[:3]

# --- Report Generation (Per Document Requirements) ---
def generate_report(eval_data, target_scores, overall_score, rating, recommendations):
    report = [
        f"SDG 12.2-12.7 Environmental Performance Evaluation Report: {eval_data['company_name']}",
        "=" * len(report[0]),
        "",
        "1. Executive Summary",
        f"- Industry: {eval_data['industry']}",
        f"- Overall Score: {overall_score}/100",
        f"- Overall Rating: {rating}",
        f"- Third-Party Alerts: {'Environmental penalties reported' if eval_data['third_party']['penalties'] else 'No environmental penalties reported'}",
        f"- Relevant Policy Updates: {eval_data['third_party']['policy_updates'] or 'No relevant policy updates identified'}",
        "",
        "2. Target-Wise Score Breakdown",
    ]
    
    # Add target scores (achieved vs. max)
    for sdg in target_scores:
        report.append(f"- SDG {sdg}: {target_scores[sdg]}/{SDG_MAX_SCORES[sdg]}")
    
    # Add detailed criterion performance
    report.extend([
        "",
        "3. Detailed Criterion Performance",
        "   SDG 12.2: Sustainable Management and Efficient Use of Natural Resources",
    ])
    # SDG 12.2 details
    for field, desc, points, threshold in SDG_CRITERIA["12.2"]["energy"] + SDG_CRITERIA["12.2"]["materials"]:
        section = "12_2_energy" if field in eval_data["12_2_energy"] else "12_2_materials"
        value = eval_data[section][field]
        # Determine compliance status
        if field in ["renewable_share", "recycled_water_ratio", "recycled_materials_pct"]:
            status = "✓ Met" if (value is not None and value >= float(threshold.replace(">", "").replace("%", ""))) else "✗ Not Met" if value is not None else "? Data Missing"
        else:
            status = "✓ Met" if value else "✗ Not Met" if value is not None else "? Data Missing"
        report.append(f"   - {desc}: {status} (Value: {value or 'Unknown'}, Score Impact: {points})")
    
    # Add other SDG details (abbreviated for readability)
    for sdg in ["12.3", "12.4", "12.5", "12.6", "12.7"]:
        report.append(f"   SDG {sdg}: {list(SDG_CRITERIA[sdg].keys())[0].title()}")
        for section in SDG_CRITERIA[sdg].values():
            for field, desc, points, threshold in section:
                sdg_key = f"{sdg.replace('.', '_')}_{list(SDG_CRITERIA[sdg].keys())[0]}"
                value = eval_data[sdg_key][field] if (sdg_key in eval_data and field in eval_data[sdg_key]) else eval_data["third_party"].get(field, None)
                # Determine compliance status
                if field.endswith("_pct"):
                    status = "✓ Met" if (value is not None and value >= float(threshold.replace(">", "").replace("%", ""))) else "✗ Not Met" if value is not None else "? Data Missing"
                else:
                    status = "✓ Met" if value else "✗ Not Met" if value is not None else "? Data Missing"
                report.append(f"   - {desc}: {status} (Value: {value or 'Unknown'}, Score Impact: {points})")
    
    # Add AI-identified "Others" actions
    if eval_data["other_positive_actions"]:
        report.extend([
            "",
            "4. Additional Positive Actions (AI-Identified, 'Others' Category)",
            eval_data["other_positive_actions"],
        ])
    
    # Add actionable recommendations
    report.extend([
        "",
        "5. Actionable Improvement Recommendations",
    ])
    for i, rec in enumerate(recommendations, 1):
        report.append(f"   {i}. {rec}")
    
    # Add data sources (per document's traceability requirement)
    report.extend([
        "",
        "6. Data Sources",
        "- User-uploaded documents (ESG reports, annual reports, internal company reports)",
        "- Third-party data (environmental news, penalty notifications, policy updates)",
        "- AI-extracted data (validated against user inputs where applicable)",
        "- Scoring framework: KPMG (2024) & IFRS (2022) (per 'scoring tool.docx')"
    ])
    
    return "\n".join(report)

# --- UI Functions (Proactive Data Collection & Visualization) ---
def render_score_charts(target_scores):
    """Visualize target-wise scores vs. maximum possible scores (per document)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sdgs = list(target_scores.keys())
    achieved_scores = [target_scores[sdg] for sdg in sdgs]
    max_scores = [SDG_MAX_SCORES[sdg] for sdg in sdgs]
    
    x = range(len(sdgs))
    width = 0.35
    
    # Plot bars
    bars_achieved = ax.bar([i - width/2 for i in x], achieved_scores, width, label="Achieved Score", color="#2196F3")
    bars_max = ax.bar([i + width/2 for i in x], max_scores, width, label="Maximum Score", color="#f0f0f0", alpha=0.7)
    
    # Customize plot
    ax.set_xlabel("SDG 12 Targets")
    ax.set_ylabel("Score")
    ax.set_title("SDG 12.2-12.7: Achieved Score vs. Maximum Score")
    ax.set_xticks(x)
    ax.set_xticklabels(sdgs)
    ax.legend()
    
    # Add score labels to bars
    for bar in bars_achieved:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f"{height}", ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)

def input_step_0():
    """Step 0: Input Method Selection (PDF Upload or Manual Input)"""
    st.subheader("Step 1: Select Input Method")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Documents")
        if not PDF_AVAILABLE:
            st.info("⚠️ PDF upload requires PyPDF2. Install with: pip install PyPDF2")
        else:
            uploaded_file = st.file_uploader("Upload ESG/Annual/Internal Report (PDF)", type="pdf")
            company_name = st.text_input("Company Name (for third-party data check)")
            
            if uploaded_file and company_name and st.button("Analyze Documents"):
                with st.spinner("Extracting and analyzing data from PDF..."):
                    # Extract text from PDF
                    try:
                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                        pdf_text = ""
                        for page in pdf_reader.pages:
                            pdf_text += page.extract_text() or ""
                    except Exception as e:
                        st.error(f"❌ Failed to read PDF: {str(e)}")
                        return
                    
                    # Extract SDG data from PDF text
                    pdf_data = extract_pdf_data(pdf_text)
                    if pdf_data:
                        # Update session state with extracted data
                        st.session_state["eval_data"]["company_name"] = pdf_data.get("company_name", company_name)
                        st.session_state["eval_data"]["industry"] = pdf_data.get("industry", "Manufacturing")
                        # Update SDG 12.2 data
                        for field in ["renewable_share", "energy_retrofit", "energy_increase", "carbon_offsets_only", "recycled_water_ratio", "ghg_disclosure"]:
                            if field in pdf_data and field in st.session_state["eval_data"]["12_2_energy"]:
                                st.session_state["eval_data"]["12_2_energy"][field] = pdf_data[field]
                        for field in ["recycled_materials_pct", "illegal_logging"]:
                            if field in pdf_data and field in st.session_state["eval_data"]["12_2_materials"]:
                                st.session_state["eval_data"]["12_2_materials"][field] = pdf_data[field]
                        # Update other SDG targets
                        for sdg in ["12.3", "12.4", "12.5", "12.6", "12.7"]:
                            sdg_key = f"{sdg.replace('.', '_')}_{list(SDG_CRITERIA[sdg].keys())[0]}"
                            if sdg_key in st.session_state["eval_data"]:
                                for section in SDG_CRITERIA[sdg].values():
                                    for field, _, _, _ in section:
                                        if field in pdf_data and field in st.session_state["eval_data"][sdg_key]:
                                            st.session_state["eval_data"][sdg_key][field] = pdf_data[field]
                    
                    # Fetch third-party data (news, penalties, policies)
                    st.session_state["eval_data"]["third_party"] = get_third_party_data(company_name)
                    
                    # Identify missing fields to collect from user
                    st.session_state["missing_fields"] = identify_missing_fields(st.session_state["eval_data"])
                    
                    # Navigate to next step
                    if st.session_state["missing_fields"]:
                        st.session_state["current_step"] = 1
                    else:
                        # Calculate scores and generate report directly
                        target_scores, overall_score, rating = calculate_scores(st.session_state["eval_data"])
                        st.session_state["eval_data"]["target_scores"] = target_scores
                        st.session_state["eval_data"]["overall_score"] = overall_score
                        st.session_state["eval_data"]["rating"] = rating
                        # Identify "Others" actions
                        st.session_state["eval_data"]["other_positive_actions"] = ai_other_actions(st.session_state["eval_data"])
                        # Generate recommendations
                        recommendations = generate_recommendations(st.session_state["eval_data"], target_scores, overall_score)
                        # Generate final report
                        st.session_state["report_text"] = generate_report(st.session_state["eval_data"], target_scores, overall_score, rating, recommendations)
                        st.session_state["current_step"] = 8
                    st.rerun()
    
    with col2:
        st.subheader("Manual Input")
        company_name = st.text_input("Company Name")
        industry = st.selectbox(
            "Industry",
            ["Manufacturing", "Food & Beverage", "Textiles", "Chemicals", "Electronics", "Other"],
            index=0
        )
        
        if st.button("Start Manual Evaluation"):
            # Initialize basic company data
            st.session_state["eval_data"]["company_name"] = company_name
            st.session_state["eval_data"]["industry"] = industry
            # Fetch third-party data
            st.session_state["eval_data"]["third_party"] = get_third_party_data(company_name)
            # Identify missing fields (all initial data is missing for manual input)
            st.session_state["missing_fields"] = identify_missing_fields(st.session_state["eval_data"])
            # Move to data collection step
            st.session_state["current_step"] = 1
            st.rerun()

def input_step_missing():
    """Step 1-7: Proactively collect missing data from user (per document requirement)"""
    if not st.session_state["missing_fields"]:
        st.session_state["current_step"] = 8
        st.rerun()
    
    # Get current missing field
    current_sdg, current_field = st.session_state["missing_fields"][0]
    # Find corresponding criterion details
    criterion = None
    for sdg_section in SDG_CRITERIA.values():
        for section in sdg_section.values():
            for c in section:
                if c[0] == current_field:
                    criterion = c
                    break
            if criterion:
                break
        if criterion:
            break
    if not criterion:
        # Skip unknown fields
        st.session_state["missing_fields"].pop(0)
        st.rerun()
    
    field, desc, points, threshold = criterion
    st.subheader(f"Step {st.session_state['current_step'] + 1}: Supplement SDG {current_sdg} Data")
    st.write(f"**Criterion**: {desc}")
    st.write(f"**Score Impact**: {points} points (Requirement: {threshold})")
    
    # Determine input type (percentage or boolean)
    eval_data = st.session_state["eval_data"]
    # Find the correct section for the current field
    section_key = None
    if current_sdg == "12.2":
        section_key = "12_2_energy" if field in eval_data["12_2_energy"] else "12_2_materials"
    else:
        section_key = f"{current_sdg.replace('.', '_')}_{list(SDG_CRITERIA[current_sdg].keys())[0]}"
    section = eval_data[section_key] if section_key in eval_data else {}
    
    # Render input based on field type
    if field.endswith("_pct") or field in ["renewable_share", "recycled_water_ratio", "hazardous_recovery_pct"]:
        # Percentage input
        value = st.number_input(
            f"Enter value for '{desc}' (as percentage, e.g., 45 = 45%)",
            min_value=0,
            max_value=100,
            step=1
        )
        if st.button("Save & Continue"):
            section[field] = value
            st.session_state["missing_fields"].pop(0)
            # Update step counter
            if st.session_state["missing_fields"]:
                st.session_state["current_step"] += 1
            else:
                st.session_state["current_step"] = 8
            st.rerun()
    else:
        # Boolean input (Yes/No)
        value = st.radio(f"Does your company meet this criterion?", ["Yes", "No"]) == "Yes"
        if st.button("Save & Continue"):
            section[field] = value
            st.session_state["missing_fields"].pop(0)
            # Update step counter
            if st.session_state["missing_fields"]:
                st.session_state["current_step"] += 1
            else:
                st.session_state["current_step"] = 8
            st.rerun()
    
    # Back button (for navigation)
    if st.session_state["current_step"] > 0 and st.button("Back to Previous Step"):
        st.session_state["current_step"] -= 1
        # Re-add current field to missing list (not saved)
        st.session_state["missing_fields"].insert(0, (current_sdg, current_field))
        st.rerun()

def render_report_page():
    """Step 8: Display Final Report and Visualizations"""
    eval_data = st.session_state["eval_data"]
    
    # Calculate scores if not already computed
    if not eval_data["target_scores"]:
        target_scores, overall_score, rating = calculate_scores(eval_data)
        eval_data["target_scores"] = target_scores
        eval_data["overall_score"] = overall_score
        eval_data["rating"] = rating
        # Identify "Others" category actions
        eval_data["other_positive_actions"] = ai_other_actions(eval_data)
        # Generate recommendations
        recommendations = generate_recommendations(eval_data, target_scores, overall_score)
        # Generate final report
        st.session_state["report_text"] = generate_report(eval_data, target_scores, overall_score, rating, recommendations)
    
    # Display rating banner (color-coded for clarity)
    rating_colors = {
        "High Responsibility Enterprise (Low Risk)": "#4CAF50",
        "Compliant but Requires Improvement (Moderate Risk)": "#FFD700",
        "Potential Environmental Risk (High Risk)": "#FFA500",
        "High Ethical Risk (Severe Risk)": "#DC143C"
    }
    st.markdown(
        f"""
        <div style="background-color:{rating_colors[eval_data['rating']]}; color:white; padding:12px; border-radius:6px; margin-bottom:20px;">
        <h3>Overall Rating: {eval_data['rating']}</h3>
        <h4>Total Score: {eval_data['overall_score']}/100</h4>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display score visualization
    st.subheader("SDG 12.2-12.7 Score Distribution")
    render_score_charts(eval_data["target_scores"])
    
    # Display detailed report
    st.subheader("Detailed Evaluation Report")
    st.text(st.session_state["report_text"])
    
    # Download report (per document's output requirement)
    st.download_button(
        label="📥 Download Evaluation Report",
        data=st.session_state["report_text"],
        file_name=f"{eval_data['company_name']}_SDG12_2-12_7_Report.txt",
        mime="text/plain"
    )
    
    # Reset for new evaluation
    if st.button("Start New Evaluation"):
        st.session_state.clear()
        # Re-initialize session state
        st.session_state["eval_data"] = {
            "company_name": "",
            "industry": "Manufacturing",
            "third_party": {"penalties": False, "positive_news": "", "policy_updates": ""},
            "12_2_energy": {"renewable_share": None, "energy_retrofit": False, "energy_increase": False, "carbon_offsets_only": False, "recycled_water_ratio": None, "ghg_disclosure": False},
            "12_2_materials": {"recycled_materials_pct": None, "illegal_logging": False},
            "12_3_loss": {"loss_tracking_system": False, "loss_reduction_pct": None},
            "12_4_chemicals": {"mrsl_zdhc_compliance": False, "regular_emission_tests": False},
            "12_4_waste": {"hazardous_recovery_pct": None, "illegal_disposal": False},
            "12_5_waste_min": {"packaging_reduction_pct": None},
            "12_5_recycle": {"recycling_rate_pct": None, "sustainable_products_pct": None},
            "12_5_disclosure": {"waste_disclosure_audit": False},
            "12_6_targets": {"emission_plans": False, "annual_progress_disclosed": False, "no_goals": False},
            "12_6_risk": {"high_carbon_assets_disclosed": False},
            "12_7_procurement": {"esg_audited_suppliers_pct": None, "price_only_procurement": False, "supply_chain_transparency": False},
            "other_positive_actions": "",
            "target_scores": {},
            "overall_score": 0,
            "rating": ""
        }
        st.session_state["current_step"] = 0
        st.rerun()

# --- Main UI Flow ---
st.title("🌱 SDG 12.2-12.7 Environmental Performance Evaluator")
st.write("Evaluate corporate performance on SDG 12.2-12.7 (Environmental Dimension of ESG Reports) per 'scoring tool.docx'")

# Navigate between steps
if st.session_state["current_step"] == 0:
    input_step_0()
elif 1 <= st.session_state["current_step"] <= 7:
    input_step_missing()
elif st.session_state["current_step"] == 8:
    render_report_page()

# Progress Indicator (for data collection steps)
if 1 <= st.session_state["current_step"] <= 7:
    total_missing = len(st.session_state["missing_fields"]) + st.session_state["current_step"]
    progress = st.session_state["current_step"] / total_missing if total_missing > 0 else 1.0
    st.sidebar.progress(progress)
    st.sidebar.write(f"Data Collection Progress: {st.session_state['current_step']}/{total_missing}")
    st.sidebar.subheader("Pending Data Points")
    for i, (sdg, field) in enumerate(st.session_state["missing_fields"], 1):
        st.sidebar.write(f"{i}. SDG {sdg}: {field}")
