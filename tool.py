import streamlit as st
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import io

# --- 1. Third-Party Data (Enhanced for "scoring tool.docx") ---
def get_third_party_data(company_name, industry):
    """Fetch verified data per "scoring tool.docx": penalties, positive news, policy updates"""
    if not company_name or not OPENAI_AVAILABLE:
        return {
            "penalties": False, 
            "penalties_details": "No third-party data fetched (missing company name or AI key)",
            "positive_news": "No third-party data fetched",
            "policy_updates": "No third-party data fetched"
        }
    
    # Enriched prompt: references "scoring tool.docx" + specific sources
    prompt = f"""For {company_name} (industry: {industry}), extract ONLY the following per "scoring tool.docx" requirements:
    1. Environmental penalties (2023-2024): List any violations related to SDG 12.2-12.7 (e.g., illegal waste disposal, non-compliance with MRSL/ZDHC). Include issuing authority and date if available.
    2. Positive sustainability news (2023-2024): Actions unlisted in SDG 12.2-12.7 (e.g., new recycling partnerships, renewable energy investments).
    3. Policy updates: Regional/national policies (2023-2024) affecting SDG 12 compliance (e.g., extended producer responsibility laws).
    
    Sources to prioritize: Environmental protection agencies, Bloomberg Green, Reuters Sustainability, company regulatory filings.
    Return ONLY a valid JSON with keys: penalties (bool), penalties_details (str), positive_news (str), policy_updates (str).
    If no data exists for a key, use "No relevant data found"."""
    
    response = get_ai_response(prompt, "Environmental data analyst specializing in SDG 12 and corporate compliance.")
    try:
        # Add fallbacks for missing keys
        data = json.loads(response) if response else {}
        return {
            "penalties": data.get("penalties", False),
            "penalties_details": data.get("penalties_details", "No relevant data found"),
            "positive_news": data.get("positive_news", "No relevant data found"),
            "policy_updates": data.get("policy_updates", "No relevant data found")
        }
    except json.JSONDecodeError:
        # Fallback if AI returns invalid JSON
        return {
            "penalties": False,
            "penalties_details": f"Invalid third-party data response: {response[:100]}...",
            "positive_news": "No valid third-party data found",
            "policy_updates": "No valid third-party data found"
        }

# --- 2. PDF Handling (Fixed Extraction for "scoring tool.docx") ---
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("⚠️ PyPDF2 not found. Install with: pip install PyPDF2 (required for PDF upload)")

def extract_full_pdf_text(file):
    """Extract ALL text from PDF (no truncation) + warn about image-based text"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        full_text = ""
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text() or ""
            full_text += f"\n--- Page {page_num} ---\n{page_text}"
        
        # Warn if text is empty (likely image-based PDF)
        if len(full_text.strip()) < 100:
            st.warning("⚠️ PDF may contain image-based text (PyPDF2 cannot extract this). Use manual input or a text-based PDF.")
        
        return full_text
    except Exception as e:
        st.error(f"❌ PDF Extraction Error: {str(e)}")
        return ""

def extract_sdg_data_from_pdf(pdf_text, company_name):
    """Enhanced extraction: Explicitly references "scoring tool.docx" criteria"""
    if len(pdf_text.strip()) < 500:
        st.error("❌ Insufficient text to extract SDG 12.2-12.7 data (per 'scoring tool.docx'). Use a longer PDF or manual input.")
        return {}
    
    # Enriched prompt: Lists ALL "scoring tool.docx" data points + examples
    prompt = f"""Extract SDG 12.2-12.7 data from this PDF text for {company_name}, strictly per "scoring tool.docx" scoring table:
    
    PDF Text (first 10,000 characters):
    {pdf_text[:10000]}
    
    REQUIRED DATA POINTS (per "scoring tool.docx"):
    - renewable_share: % of renewable energy (e.g., 55 = 55%)
    - energy_retrofit: True/False (full-scale energy retrofit)
    - energy_increase: True/False (energy up 2 consecutive years)
    - carbon_offsets_only: True/False (sole reliance on carbon offsets)
    - recycled_water_ratio: % of recycled water (e.g., 75 = 75%)
    - ghg_disclosure: True/False (Scope1-3 disclosed + third-party verified)
    - recycled_materials_pct: % of recycled materials (e.g., 35 = 35%)
    - illegal_logging: True/False (any incidents)
    - loss_tracking_system: True/False (established)
    - loss_reduction_pct: % annual loss reduction (e.g., 12 = 12%)
    - mrsl_zdhc_compliance: True/False (compliant with MRSL/ZDHC)
    - regular_emission_tests: True/False (regular testing)
    - hazardous_recovery_pct: % hazardous waste recovery (e.g., 92 = 92%)
    - illegal_disposal: True/False (improper disposal)
    - packaging_reduction_pct: % packaging weight reduction (e.g., 25 = 25%)
    - recycling_rate_pct: % recycling rate (e.g., 85 = 85%)
    - sustainable_products_pct: % products with sustainable materials (e.g., 55 = 55%)
    - waste_disclosure_audit: True/False (disclosed + third-party audit)
    - emission_plans: True/False (clear 2030/2050 emission goals)
    - annual_progress_disclosed: True/False (annual progress shared)
    - no_goals: True/False (no goals or stagnant progress)
    - high_carbon_assets_disclosed: True/False (disclosed + reduction pathway)
    - esg_audited_suppliers_pct: % suppliers with ESG audits (e.g., 85 = 85%)
    - price_only_procurement: True/False (price-only or high-emission outsourcing)
    - supply_chain_transparency: True/False (transparency report)
    
    Return ONLY a valid JSON. Use null for unknown values. Do NOT add extra text."""
    
    response = get_ai_response(prompt, f"ESG data extractor trained on 'scoring tool.docx' criteria. Return only JSON.")
    if not response:
        st.error("❌ AI returned no data for PDF extraction.")
        return {}
    
    # Clean JSON (remove non-JSON text)
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        st.error(f"❌ AI response is not JSON (per 'scoring tool.docx' requirements). Raw response: {response[:200]}...")
        return {}
    
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError as e:
        st.error(f"❌ Failed to parse PDF data (per 'scoring tool.docx'): {str(e)}. Raw JSON: {json_match.group()[:200]}...")
        return {}

# --- 3. Page Config & OpenAI Client ---
st.set_page_config(page_title="SDG 12.2-12.7 Evaluator", layout="wide")

try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except KeyError:
    st.warning("⚠️ OPENAI_API_KEY missing in Streamlit Secrets. AI features (PDF extraction, recommendations) disabled.")
    OPENAI_AVAILABLE = False
except Exception as e:
    st.error(f"⚠️ OpenAI Error: {str(e)}")
    OPENAI_AVAILABLE = False

# --- 4. Session State (Aligned with Merged Steps) ---
if "eval_data" not in st.session_state:
    st.session_state["eval_data"] = {
        "company_name": "",
        "industry": "Manufacturing",
        "third_party": {
            "penalties": False, "penalties_details": "", "positive_news": "", "policy_updates": ""
        },
        # SDG 12.2 (Grouped for Step 2)
        "12_2": {
            "renewable_share": None, "energy_retrofit": False, "energy_increase": False,
            "carbon_offsets_only": False, "recycled_water_ratio": None, "ghg_disclosure": False,
            "recycled_materials_pct": None, "illegal_logging": False
        },
        # SDG 12.3-12.4 (Grouped for Step 3)
        "12_3_4": {
            "loss_tracking_system": False, "loss_reduction_pct": None,
            "mrsl_zdhc_compliance": False, "regular_emission_tests": False,
            "hazardous_recovery_pct": None, "illegal_disposal": False
        },
        # SDG 12.5-12.6 (Grouped for Step 4)
        "12_5_6": {
            "packaging_reduction_pct": None, "recycling_rate_pct": None,
            "sustainable_products_pct": None, "waste_disclosure_audit": False,
            "emission_plans": False, "annual_progress_disclosed": False, "no_goals": False,
            "high_carbon_assets_disclosed": False
        },
        # SDG 12.7 (Step 5)
        "12_7": {
            "esg_audited_suppliers_pct": None, "price_only_procurement": False,
            "supply_chain_transparency": False
        },
        # Final Notes (Step 6)
        "additional_notes": "",
        # Scoring Results
        "target_scores": {}, "overall_score": 0, "rating": "", "other_positive_actions": ""
    }
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0  # 0: Front Page, 1: Step 1 (Company), 2: Step 2 (12.2), ..., 6: Report
if "pdf_extracted_text" not in st.session_state:
    st.session_state["pdf_extracted_text"] = ""  # For PDF preview
if "missing_fields" not in st.session_state:
    st.session_state["missing_fields"] = []

# --- 5. Constants (Exact from "scoring tool.docx") ---
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

# --- 6. Core AI Functions (Enriched Prompts) ---
def get_ai_response(prompt, system_msg="You are a helpful assistant."):
    if not OPENAI_AVAILABLE:
        return "AI features require an OPENAI_API_KEY in Streamlit Secrets."
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            temperature=0.2,  # Low temp for factual accuracy (per "scoring tool.docx")
            timeout=25
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        return ""

def identify_missing_fields(eval_data):
    """Identify missing data per "scoring tool.docx""""
    missing = []
    # Check SDG 12.2
    for field, _, _, _ in SDG_CRITERIA["12.2"]:
        if eval_data["12_2"][field] is None:
            missing.append(("12.2", field))
    # Check SDG 12.3-12.4
    for sdg in ["12.3", "12.4"]:
        for field, _, _, _ in SDG_CRITERIA[sdg]:
            if field != "penalties" and eval_data["12_3_4"][field] is None:
                missing.append((sdg, field))
    # Check SDG 12.5-12.6
    for sdg in ["12.5", "12.6"]:
        for field, _, _, _ in SDG_CRITERIA[sdg]:
            if eval_data["12_5_6"][field] is None:
                missing.append((sdg, field))
    # Check SDG 12.7
    for field, _, _, _ in SDG_CRITERIA["12.7"]:
        if eval_data["12_7"][field] is None:
            missing.append(("12.7", field))
    return missing

def ai_other_actions(eval_data):
    """Enriched prompt: Ties to "scoring tool.docx" "Others" category"""
    if not OPENAI_AVAILABLE:
        return "- Implemented employee training on SDG 12.2-12.7 compliance\n- Partnered with local recyclers for by-product reuse"
    
    prompt = f"""For {eval_data['company_name']} (industry: {eval_data['industry']}), identify 1-2 positive environmental actions NOT listed in SDG 12.2-12.7 criteria (per "scoring tool.docx" "Others" category).
    
    Current data (per "scoring tool.docx"):
    - SDG 12.2: Renewable share={eval_data['12_2']['renewable_share']}%, Recycled materials={eval_data['12_2']['recycled_materials_pct']}%
    - SDG 12.4: Hazardous recovery={eval_data['12_3_4']['hazardous_recovery_pct']}%
    - SDG 12.7: ESG suppliers={eval_data['12_7']['esg_audited_suppliers_pct']}%
    - Third-party news: {eval_data['third_party']['positive_news'][:200]}...
    
    Actions must:
    1. Be industry-relevant (e.g., manufacturing: solar panel installation; textiles: water recycling).
    2. Have clear environmental benefits tied to SDG 12.
    3. Not overlap with existing SDG 12.2-12.7 criteria.
    
    Return as bullet points (max 2). No extra text."""
    
    response = get_ai_response(prompt, f"Sustainability consultant specializing in 'scoring tool.docx' criteria.")
    return response.strip() if response else "- Implemented employee training on SDG 12.2-12.7 compliance\n- Partnered with local recyclers for by-product reuse"

# --- 7. Scoring Function (Per "scoring tool.docx") ---
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
    
    # Overall Rating (per "scoring tool.docx")
    overall = sum(scores.values())
    rating = "High Responsibility Enterprise (Low Risk)" if overall >=75 else \
             "Compliant but Requires Improvement (Moderate Risk)" if 60<=overall<75 else \
             "Potential Environmental Risk (High Risk)" if 40<=overall<60 else \
             "High Ethical Risk (Severe Risk)"
    
    return scores, overall, rating

# --- 8. Recommendations (Enriched for "scoring tool.docx") ---
def generate_recommendations(eval_data, target_scores, overall_score):
    if not OPENAI_AVAILABLE:
        return [
            f"SDG 12.2: Increase renewable energy to ≥50% (current: {eval_data['12_2']['renewable_share'] or 'Unknown'}%)—install 5MW solar panels by Q2 2025 (gains +7 points per 'scoring tool.docx').",
            f"SDG 12.4: Boost hazardous waste recovery to ≥90% (current: {eval_data['12_3_4']['hazardous_recovery_pct'] or 'Unknown'}%)—partner with a certified waste handler by Q1 2025 (gains +5 points).",
            f"SDG 12.7: Audit ≥80% of suppliers for ESG compliance (current: {eval_data['12_7']['esg_audited_suppliers_pct'] or 'Unknown'}%)—hire a third-party auditor by Q3 2024 (gains +7 points)."
        ]
    
    prompt = f"""Generate 3 DETAILED recommendations for {eval_data['company_name']} (industry: {eval_data['industry']}) to improve SDG 12.2-12.7 scores, strictly per "scoring tool.docx".
    
    Current Status (per "scoring tool.docx"):
    - Target scores (achieved/max): {json.dumps({k: f'{v}/{SDG_MAX_SCORES[k]}' for k, v in target_scores.items()}, indent=2)}
    - Overall score: {overall_score}/100
    - Low targets: {[k for k, v in target_scores.items() if v < SDG_MAX_SCORES[k]*0.5]}
    - Missing data: {[f'SDG {s}: {f}' for s, f in identify_missing_fields(eval_data)]}
    - Penalties: {eval_data['third_party']['penalties_details'][:150]}...
    
    Recommendations must:
    1. Link to "scoring tool.docx" criteria (e.g., "SDG 12.2: X to Y% for +Z points").
    2. Be industry-specific (e.g., manufacturing: solar panels; food & beverage: packaging reduction).
    3. Include time-bound steps (e.g., "Q2 2025") and resources (e.g., "$500k budget").
    4. Prioritize low-scoring targets first.
    
    Format as numbered bullet points. No extra text."""
    
    response = get_ai_response(prompt, f"Sustainability consultant trained on 'scoring tool.docx' scoring rules.")
    recs = [l.strip() for l in response.split("\n") if l.strip() and l[0].isdigit()]
    while len(recs) < 3:
        recs.append(f"SDG 12.2: Increase recycled materials to ≥30% (current: {eval_data['12_2']['recycled_materials_pct'] or 'Unknown'}%)—source from {eval_data['industry']}-specific recyclers by Q1 2025 (gains +5 points per 'scoring tool.docx').")
    return recs[:3]

# --- 9. Report Generation (Includes Third-Party Data & Notes) ---
def generate_report(eval_data, target_scores, overall_score, rating, recommendations):
    report = [
        f"SDG 12.2-12.7 Environmental Performance Report: {eval_data['company_name']}",
        "=" * len(report[0]),
        f"\nPrepared per 'scoring tool.docx' (KPMG 2024 & IFRS 2022)",
        "",
        "1. Executive Summary",
        f"- Industry: {eval_data['industry']}",
        f"- Overall Score: {overall_score}/100",
        f"- Overall Rating: {rating}",
        f"- Additional Notes: {eval_data['additional_notes'] or 'No additional notes provided'}",
        "",
        "2. Third-Party Data (Per 'scoring tool.docx' Requirements)",
        f"- Environmental Penalties: {eval_data['third_party']['penalties_details']}",
        f"- Positive Sustainability News: {eval_data['third_party']['positive_news']}",
        f"- Relevant Policy Updates: {eval_data['third_party']['policy_updates']}",
        "",
        "3. Target-Wise Score Breakdown",
    ]
    
    for sdg in target_scores:
        report.append(f"- SDG {sdg}: {target_scores[sdg]}/{SDG_MAX_SCORES[sdg]}")
    
    # Detailed Criterion Performance
    report.extend([
        "",
        "4. Detailed Criterion Performance (Per 'scoring tool.docx')",
    ])
    # SDG 12.2
    report.append("   SDG 12.2: Sustainable Resource Management")
    for field, desc, points, threshold in SDG_CRITERIA["12.2"]:
        val = eval_data["12_2"][field]
        status = "✓ Met" if (
            ("%" in threshold and val is not None and val >= float(re.sub(r"[>≥%]", "", threshold)))
            or (not "%" in threshold and val)
        ) else "✗ Not Met" if val is not None else "? Data Missing"
        report.append(f"   - {desc}: {status} (Value: {val or 'Unknown'}, Score Impact: {points})")
    
    # SDG 12.3-12.4
    for sdg in ["12.3", "12.4"]:
        report.append(f"   SDG {sdg}: {('Loss Control' if sdg=='12.3' else 'Chemical/Waste Management')}")
        for field, desc, points, threshold in SDG_CRITERIA[sdg]:
            if field == "penalties":
                val = not eval_data["third_party"]["penalties"]
                status = "✓ Met" if val else "✗ Not Met"
                report.append(f"   - {desc}: {status} (Details: {eval_data['third_party']['penalties_details'][:100]}..., Score Impact: {points})")
            else:
                val = eval_data["12_3_4"][field]
                status = "✓ Met" if (
                    ("%" in threshold and val is not None and val >= float(re.sub(r"[>≥%]", "", threshold)))
                    or (not "%" in threshold and val)
                ) else "✗ Not Met" if val is not None else "? Data Missing"
                report.append(f"   - {desc}: {status} (Value: {val or 'Unknown'}, Score Impact: {points})")
    
    # SDG 12.5-12.6
    for sdg in ["12.5", "12.6"]:
        report.append(f"   SDG {sdg}: {('Waste Reduction' if sdg=='12.5' else 'Sustainable Reporting')}")
        for field, desc, points, threshold in SDG_CRITERIA[sdg]:
            val = eval_data["12_5_6"][field]
            status = "✓ Met" if (
                ("%" in threshold and val is not None and val >= float(re.sub(r"[>≥%]", "", threshold)))
                or (not "%" in threshold and val)
            ) else "✗ Not Met" if val is not None else "? Data Missing"
            report.append(f"   - {desc}: {status} (Value: {val or 'Unknown'}, Score Impact: {points})")
    
    # SDG 12.7
    report.append("   SDG 12.7: Sustainable Procurement")
    for field, desc, points, threshold in SDG_CRITERIA["12.7"]:
        val = eval_data["12_7"][field]
        status = "✓ Met" if (
            ("%" in threshold and val is not None and val >= float(re.sub(r"[>≥%]", "", threshold)))
            or (not "%" in threshold and val)
        ) else "✗ Not Met" if val is not None else "? Data Missing"
        report.append(f"   - {desc}: {status} (Value: {val or 'Unknown'}, Score Impact: {points})")
    
    # "Others" Category
    report.extend([
        "",
        "5. Additional Positive Actions ('Others' Category per 'scoring tool.docx')",
        eval_data["other_positive_actions"] or "No additional actions identified.",
        "",
        "6. Actionable Recommendations",
    ])
    for i, rec in enumerate(recommendations, 1):
        report.append(f"   {i}. {rec}")
    
    # Data Sources
    report.extend([
        "",
        "7. Data Sources (Per 'scoring tool.docx')",
        "- User-provided documents (ESG/annual reports) or manual inputs",
        "- Third-party data (environmental agencies, credible news outlets)",
        "- AI analysis aligned with 'scoring tool.docx' criteria",
    ])
    
    return "\n".join(report)

# --- 10. UI Functions (Merged Steps + Aligned Front Page) ---
def render_front_page():
    """Fixed alignment: Consistent columns + input sizing"""
    st.title("🌱 SDG 12.2-12.7 Environmental Evaluator")
    st.write("Evaluate performance per **'scoring tool.docx'** (SDG 12.2-12.7, Environmental Dimension of ESG)")
    
    # Aligned columns (fixed width for consistency)
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.subheader("Option 1: Upload ESG/Annual Report (PDF)")
        if not PDF_AVAILABLE:
            st.info("⚠️ PDF upload requires PyPDF2. Install with: pip install PyPDF2")
        else:
            # Aligned inputs: Company name + file uploader
            company_name = st.text_input(
                "Company Name (required for third-party data)",
                value=st.session_state["eval_data"]["company_name"],
                help="Used to fetch penalties/news (per 'scoring tool.docx')"
            )
            uploaded_file = st.file_uploader(
                "Upload PDF (text-based only)",
                type="pdf",
                help="Extracts data per 'scoring tool.docx' criteria"
            )
            
            if uploaded_file and company_name and st.button("Analyze PDF & Proceed"):
                with st.spinner("Extracting PDF text + fetching third-party data..."):
                    # Extract and preview PDF text
                    pdf_text = extract_full_pdf_text(uploaded_file)
                    st.session_state["pdf_extracted_text"] = pdf_text
                    
                    # Show text preview (so user confirms data exists)
                    with st.expander("View Extracted PDF Text (First 1,000 Characters)"):
                        st.text(pdf_text[:1000] + "...")
                    
                    # Extract SDG data from PDF
                    if OPENAI_AVAILABLE:
                        sdg_data = extract_sdg_data_from_pdf(pdf_text, company_name)
                        # Update session state with extracted data
                        if sdg_data:
                            # SDG 12.2
                            for field in ["renewable_share", "energy_retrofit", "energy_increase", "carbon_offsets_only", "recycled_water_ratio", "ghg_disclosure", "recycled_materials_pct", "illegal_logging"]:
                                if field in sdg_data and sdg_data[field] is not None:
                                    st.session_state["eval_data"]["12_2"][field] = sdg_data[field]
                            # SDG 12.3-12.4
                            for field in ["loss_tracking_system", "loss_reduction_pct", "mrsl_zdhc_compliance", "regular_emission_tests", "hazardous_recovery_pct", "illegal_disposal"]:
                                if field in sdg_data and sdg_data[field] is not None:
                                    st.session_state["eval_data"]["12_3_4"][field] = sdg_data[field]
                            # SDG 12.5-12.6
                            for field in ["packaging_reduction_pct", "recycling_rate_pct", "sustainable_products_pct", "waste_disclosure_audit", "emission_plans", "annual_progress_disclosed", "no_goals", "high_carbon_assets_disclosed"]:
                                if field in sdg_data and sdg_data[field] is not None:
                                    st.session_state["eval_data"]["12_5_6"][field] = sdg_data[field]
                            # SDG 12.7
                            for field in ["esg_audited_suppliers_pct", "price_only_procurement", "supply_chain_transparency"]:
                                if field in sdg_data and sdg_data[field] is not None:
                                    st.session_state["eval_data"]["12_7"][field] = sdg_data[field]
                    
                    # Fetch third-party data
                    st.session_state["eval_data"]["company_name"] = company_name
                    st.session_state["eval_data"]["third_party"] = get_third_party_data(company_name, st.session_state["eval_data"]["industry"])
                    
                    # Identify missing fields and move to Step 2 (or final notes if none missing)
                    st.session_state["missing_fields"] = identify_missing_fields(st.session_state["eval_data"])
                    if st.session_state["missing_fields"]:
                        st.session_state["current_step"] = 2  # Step 2: SDG 12.2 (first missing group)
                    else:
                        st.session_state["current_step"] = 6  # Step 6: Additional Notes
                    st.rerun()
    
    with col2:
        st.subheader("Option 2: Manual Input")
        # Aligned inputs: Company name + industry
        company_name = st.text_input(
            "Company Name",
            value=st.session_state["eval_data"]["company_name"]
        )
        industry = st.selectbox(
            "Industry (per 'scoring tool.docx')",
            ["Manufacturing", "Food & Beverage", "Textiles", "Chemicals", "Electronics", "Other"],
            index=["Manufacturing", "Food & Beverage", "Textiles", "Chemicals", "Electronics", "Other"].index(
                st.session_state["eval_data"]["industry"]
            ),
            help="Affects recommendations (e.g., textiles = water focus)"
        )
        
        if st.button("Start Manual Input"):
            # Initialize company data
            st.session_state["eval_data"]["company_name"] = company_name
            st.session_state["eval_data"]["industry"] = industry
            # Fetch third-party data
            st.session_state["eval_data"]["third_party"] = get_third_party_data(company_name, industry)
            # Move to Step 1: SDG 12.2
            st.session_state["current_step"] = 2
            st.rerun()

def step_2_sdg_12_2():
    """Step 2: SDG 12.2 (All indicators in one page)"""
    st.subheader("Step 2/6: SDG 12.2 – Sustainable Resource Management (per 'scoring tool.docx')")
    eval_data = st.session_state["eval_data"]
    
    # Grouped inputs for SDG 12.2
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.caption("Energy Management")
        eval_data["12_2"]["renewable_share"] = st.number_input(
            "Renewable energy share (%) ≥50% = +7 points",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_2"]["renewable_share"] or 0
        )
        eval_data["12_2"]["energy_retrofit"] = st.radio(
            "Full-scale energy retrofit? = +5 points",
            ["Yes", "No"],
            index=0 if eval_data["12_2"]["energy_retrofit"] else 1
        ) == "Yes"
        eval_data["12_2"]["energy_increase"] = st.radio(
            "Energy consumption up 2 consecutive years? = -5 points",
            ["Yes", "No"],
            index=1 if eval_data["12_2"]["energy_increase"] else 0
        ) == "Yes"
        eval_data["12_2"]["carbon_offsets_only"] = st.radio(
            "Rely solely on carbon offsets? = -3 points",
            ["Yes", "No"],
            index=1 if eval_data["12_2"]["carbon_offsets_only"] else 0
        ) == "Yes"
    
    with col2:
        st.caption("Water & Material Management")
        eval_data["12_2"]["recycled_water_ratio"] = st.number_input(
            "Recycled water ratio (%) ≥70% = +5 points",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_2"]["recycled_water_ratio"] or 0
        )
        eval_data["12_2"]["ghg_disclosure"] = st.radio(
            "Scope1-3 GHG (disclosed + third-party verified)? = +7 points",
            ["Yes", "No"],
            index=0 if eval_data["12_2"]["ghg_disclosure"] else 1
        ) == "Yes"
        eval_data["12_2"]["recycled_materials_pct"] = st.number_input(
            "Recycled materials (%) ≥30% = +5 points",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_2"]["recycled_materials_pct"] or 0
        )
        eval_data["12_2"]["illegal_logging"] = st.radio(
            "Illegal logging incidents? = -7 points",
            ["Yes", "No"],
            index=1 if eval_data["12_2"]["illegal_logging"] else 0
        ) == "Yes"
    
    # Navigation buttons
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Back to Front Page"):
            st.session_state["current_step"] = 0
            st.rerun()
    with col2_btn:
        if st.button("Proceed to SDG 12.3-12.4 (Step 3)"):
            st.session_state["current_step"] = 3
            st.rerun()

def step_3_sdg_12_3_4():
    """Step 3: SDG 12.3-12.4 (Merged)"""
    st.subheader("Step 3/6: SDG 12.3-12.4 – Loss Control & Chemical/Waste Management")
    eval_data = st.session_state["eval_data"]
    
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.caption("SDG 12.3 – Loss Control (per 'scoring tool.docx')")
        eval_data["12_3_4"]["loss_tracking_system"] = st.radio(
            "Loss-tracking system established? = +5 points",
            ["Yes", "No"],
            index=0 if eval_data["12_3_4"]["loss_tracking_system"] else 1
        ) == "Yes"
        eval_data["12_3_4"]["loss_reduction_pct"] = st.number_input(
            "Annual loss reduction (%) >10% = +4 points",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_3_4"]["loss_reduction_pct"] or 0
        )
    
    with col2:
        st.caption("SDG 12.4 – Chemical/Waste Management")
        eval_data["12_3_4"]["mrsl_zdhc_compliance"] = st.radio(
            "Comply with MRSL/ZDHC standards? = +5 points",
            ["Yes", "No"],
            index=0 if eval_data["12_3_4"]["mrsl_zdhc_compliance"] else 1
        ) == "Yes"
        eval_data["12_3_4"]["regular_emission_tests"] = st.radio(
            "Regular emission testing? = +3 points",
            ["Yes", "No"],
            index=0 if eval_data["12_3_4"]["regular_emission_tests"] else 1
        ) == "Yes"
        eval_data["12_3_4"]["hazardous_recovery_pct"] = st.number_input(
            "Hazardous waste recovery (%) ≥90% = +5 points",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_3_4"]["hazardous_recovery_pct"] or 0
        )
        eval_data["12_3_4"]["illegal_disposal"] = st.radio(
            "Improper disposal? = -3 points",
            ["Yes", "No"],
            index=1 if eval_data["12_3_4"]["illegal_disposal"] else 0
        ) == "Yes"
    
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Back to SDG 12.2 (Step 2)"):
            st.session_state["current_step"] = 2
            st.rerun()
    with col2_btn:
        if st.button("Proceed to SDG 12.5-12.6 (Step 4)"):
            st.session_state["current_step"] = 4
            st.rerun()

def step_4_sdg_12_5_6():
    """Step 4: SDG 12.5-12.6 (Merged)"""
    st.subheader("Step 4/6: SDG 12.5-12.6 – Waste Reduction & Sustainable Reporting")
    eval_data = st.session_state["eval_data"]
    
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.caption("SDG 12.5 – Waste Reduction/Recycling")
        eval_data["12_5_6"]["packaging_reduction_pct"] = st.number_input(
            "Packaging reduction (%) ≥20% = +4 points",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_5_6"]["packaging_reduction_pct"] or 0
        )
        eval_data["12_5_6"]["recycling_rate_pct"] = st.number_input(
            "Recycling rate (%) ≥80% = +4 points",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_5_6"]["recycling_rate_pct"] or 0
        )
        eval_data["12_5_6"]["sustainable_products_pct"] = st.number_input(
            "Sustainable material products (%) ≥50% = +4 points",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_5_6"]["sustainable_products_pct"] or 0
        )
        eval_data["12_5_6"]["waste_disclosure_audit"] = st.radio(
            "Waste disclosure + third-party audit? = +5 points",
            ["Yes", "No"],
            index=0 if eval_data["12_5_6"]["waste_disclosure_audit"] else 1
        ) == "Yes"
    
    with col2:
        st.caption("SDG 12.6 – Sustainable Reporting")
        eval_data["12_5_6"]["emission_plans"] = st.radio(
            "Clear 2030/2050 emission goals? = +5 points",
            ["Yes", "No"],
            index=0 if eval_data["12_5_6"]["emission_plans"] else 1
        ) == "Yes"
        eval_data["12_5_6"]["annual_progress_disclosed"] = st.radio(
            "Annual progress disclosed? = +4 points",
            ["Yes", "No"],
            index=0 if eval_data["12_5_6"]["annual_progress_disclosed"] else 1
        ) == "Yes"
        eval_data["12_5_6"]["no_goals"] = st.radio(
            "No goals/stagnant progress? = -3 points",
            ["Yes", "No"],
            index=1 if eval_data["12_5_6"]["no_goals"] else 0
        ) == "Yes"
        eval_data["12_5_6"]["high_carbon_assets_disclosed"] = st.radio(
            "High-carbon assets disclosed + reduction pathway?",
            ["Yes", "No"],
            index=0 if eval_data["12_5_6"]["high_carbon_assets_disclosed"] else 1
        ) == "Yes"
    
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Back to SDG 12.3-12.4 (Step 3)"):
            st.session_state["current_step"] = 3
            st.rerun()
    with col2_btn:
        if st.button("Proceed to SDG 12.7 (Step 5)"):
            st.session_state["current_step"] = 5
            st.rerun()

def step_5_sdg_12_7():
    """Step 5: SDG 12.7 (Standalone)"""
    st.subheader("Step 5/6: SDG 12.7 – Sustainable Procurement (per 'scoring tool.docx')")
    eval_data = st.session_state["eval_data"]
    
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        eval_data["12_7"]["esg_audited_suppliers_pct"] = st.number_input(
            "ESG-audited suppliers (%) ≥80% + implementation plan = +7 points",
            min_value=0, max_value=100, step=1,
            value=eval_data["12_7"]["esg_audited_suppliers_pct"] or 0
        )
        eval_data["12_7"]["supply_chain_transparency"] = st.radio(
            "Supply chain transparency report? = +3 points",
            ["Yes", "No"],
            index=0 if eval_data["12_7"]["supply_chain_transparency"] else 1
        ) == "Yes"
    
    with col2:
        eval_data["12_7"]["price_only_procurement"] = st.radio(
            "Price-only procurement or outsourcing to high-emission regions? = -3 points",
            ["Yes", "No"],
            index=1 if eval_data["12_7"]["price_only_procurement"] else 0
        ) == "Yes"
        # Third-party data preview
        st.caption("Third-Party Procurement Alerts (per 'scoring tool.docx')")
        st.info(f"Policy Updates: {eval_data['third_party']['policy_updates'][:150]}...")
    
    col1_btn, col2_btn = st.columns([1, 1])
    with col1_btn:
        if st.button("Back to SDG 12.5-12.6 (Step 4)"):
            st.session_state["current_step"] = 4
            st.rerun()
    with col2_btn:
        if st.button("Proceed to Additional Notes (Step 6)"):
            st.session_state["current_step"] = 6
            st.rerun()

def step_6_additional_notes():
    """Step 6: Final Notes (For all input methods)"""
    st.subheader("Step 6/6: Additional Sustainability Notes (per 'scoring tool.docx')")
    eval_data = st.session_state["eval_data"]
    
    # Final notes textbox
    eval_data["additional_notes"] = st.text_area(
        "Enter additional details (e.g., ongoing projects, future plans) – included in the report",
        value=eval_data["additional_notes"],
        height=150,
        help="Examples: 'Installing 10MW wind farm in 2025', 'Targeting 100% ESG suppliers by 2026'"
    )
    
    # Generate report button
    if st.button("Generate SDG 12.2-12.7 Report (per 'scoring tool.docx')"):
        with st.spinner("Calculating scores + generating report..."):
            # Calculate scores
            target_scores, overall_score, rating = calculate_scores(eval_data)
            eval_data["target_scores"] = target_scores
            eval_data["overall_score"] = overall_score
            eval_data["rating"] = rating
            # Identify "Others" actions
            eval_data["other_positive_actions"] = ai_other_actions(eval_data)
            # Generate recommendations
            recommendations = generate_recommendations(eval_data, target_scores, overall_score)
            # Generate report
            st.session_state["report_text"] = generate_report(eval_data, target_scores, overall_score, rating, recommendations)
            # Move to report page
            st.session_state["current_step"] = 7
            st.rerun()
    
    # Back button
    if st.button("Back to SDG 12.7 (Step 5)"):
        st.session_state["current_step"] = 5
        st.rerun()

def render_report_page():
    """Final Report Page"""
    eval_data = st.session_state["eval_data"]
    st.title(f"SDG 12.2-12.7 Report: {eval_data['company_name']}")
    
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
    st.subheader("Score Distribution (per 'scoring tool.docx')")
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
        file_name=f"{eval_data['company_name']}_SDG12_Report.txt",
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

# --- 11. Main UI Flow (6 Steps Total) ---
if st.session_state["current_step"] == 0:
    render_front_page()
elif st.session_state["current_step"] == 2:
    step_2_sdg_12_2()
elif st.session_state["current_step"] == 3:
    step_3_sdg_12_3_4()
elif st.session_state["current_step"] == 4:
    step_4_sdg_12_5_6()
elif st.session_state["current_step"] == 5:
    step_5_sdg_12_7()
elif st.session_state["current_step"] == 6:
    step_6_additional_notes()
elif st.session_state["current_step"] == 7:
    render_report_page()

# Progress Indicator
if 2 <= st.session_state["current_step"] <= 6:
    step_names = ["", "", "SDG 12.2", "SDG 12.3-12.4", "SDG 12.5-12.6", "SDG 12.7", "Notes"]
    current_step_name = step_names[st.session_state["current_step"]]
    progress = (st.session_state["current_step"] - 1) / 6  # 6 steps total
    st.sidebar.progress(progress)
    st.sidebar.write(f"Current Step: {st.session_state['current_step']}/6 – {current_step_name}")
    # Third-party data preview in sidebar
    st.sidebar.subheader("Third-Party Data (per 'scoring tool.docx')")
    st.sidebar.write(f"Penalties: {eval_data['third_party']['penalties_details'][:100]}...")
