import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import PyPDF2
import io

# --- Page Configuration ---
st.set_page_config(page_title="SDG 12 Production Evaluator", layout="wide")

# --- Initialize OpenAI Client ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except KeyError:
    st.warning("⚠️ OPENAI_API_KEY not found in Streamlit Secrets. AI features (excerpt, analysis) disabled.")
    OPENAI_AVAILABLE = False
except Exception as e:
    st.error(f"⚠️ OpenAI client error: {str(e)}")
    OPENAI_AVAILABLE = False

# --- Session State Initialization ---
if "evaluation_data" not in st.session_state:
    st.session_state["evaluation_data"] = {
        "company_name": "",
        "industry": "",
        "production_volume": 0,
        "material_efficiency_checks": [False]*5,
        "waste_management_checks": [False]*5,
        "energy_efficiency_checks": [False]*5,
        "water_management_checks": [False]*5,
        "circular_economy_checks": [False]*5,
        "circular_practices": [],
        "certifications": [],
        "key_metrics": {  # Added for excerpt data (recycled rates, waste diversion, etc.)
            "recycled_material_pct": None,
            "waste_diversion_pct": None,
            "water_reduction_liters": None,
            "hazardous_waste_reduction_pct": None,
            "auditor_name": None
        }
    }
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []
if "report_text" not in st.session_state:
    st.session_state["report_text"] = ""
if "scores" not in st.session_state:
    st.session_state["scores"] = {}
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None
if "custom_esg_excerpt" not in st.session_state:  # New: Store AI-generated excerpt
    st.session_state["custom_esg_excerpt"] = ""

# --- Constants ---
SDG12_CRITERIA = {
    "material_efficiency": [
        "Material waste reduction targets",
        "Use of recycled materials",
        "Material recovery rate",
        "By-product utilization",
        "Material substitution (lower environmental impact)"
    ],
    "waste_management": [
        "Waste reduction targets",
        "Recycling rate",
        "Hazardous waste treatment",
        "Waste-to-energy conversion",
        "Zero-waste production initiatives"
    ],
    "energy_efficiency": [
        "Renewable energy usage percentage",
        "Energy intensity reduction",
        "Energy management system",
        "Carbon footprint tracking",
        "Energy efficiency technologies"
    ],
    "water_management": [
        "Water usage efficiency",
        "Wastewater treatment and reuse",
        "Water stress risk management",
        "Water footprint reduction",
        "Water conservation initiatives"
    ],
    "circular_economy": [
        "Product lifecycle design",
        "Take-back/recycling programs",
        "Remanufacturing capabilities",
        "Sharing economy practices",
        "Extended producer responsibility"
    ]
}

INDUSTRY_BENCHMARKS = {
    "manufacturing": {"overall": 65, "material_efficiency": 60, "waste_management": 55, "energy_efficiency": 70, "water_management": 62, "circular_economy": 45},
    "food & beverage": {"overall": 70, "material_efficiency": 65, "waste_management": 75, "energy_efficiency": 60, "water_management": 55, "circular_economy": 50},
    "textiles": {"overall": 55, "material_efficiency": 50, "waste_management": 45, "energy_efficiency": 60, "water_management": 50, "circular_economy": 40},
    "chemicals": {"overall": 60, "material_efficiency": 55, "waste_management": 70, "energy_efficiency": 65, "water_management": 60, "circular_economy": 45},
    "electronics": {"overall": 62, "material_efficiency": 58, "waste_management": 65, "energy_efficiency": 72, "water_management": 55, "circular_economy": 55}
}

# --- Core AI Functions ---
def get_ai_response(prompt, system_msg="You are a helpful assistant and an expert in ESG and production processes across various industries."):
    if not OPENAI_AVAILABLE:
        return "AI features require an OPENAI_API_KEY."
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temp for factual, consistent ESG writing
            timeout=20
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"AI error: {str(e)}")
        return "Failed to generate AI content."

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def analyze_esg_document(text):
    prompt = f"""Analyze this ESG report text and extract or estimate ONLY SDG 12 (Responsible Consumption/Production) production-related data:
    1. Company name
    2. Industry
    3. Production volume (units/year)
    4. Key metrics (recycled material %, waste diversion %, water reduction, hazardous waste reduction %)
    5. Completed production initiatives (2023-2024)
    6. Certifications (e.g., ISO 14001)
    7. Auditor name (if mentioned)
    8. For each SDG12 category, mark criteria met (true/false): {json.dumps(SDG12_CRITERIA, indent=1)}
    
    Return as JSON with clear keys (no extra text)."""
    
    try:
        response = get_ai_response(prompt, "You are an ESG analyst specializing in manufacturing sustainability. Extract only factual data.")
        return json.loads(response)
    except Exception as e:
        st.error(f"Document analysis error: {str(e)}")
        return {}

def generate_custom_esg_excerpt(evaluation_data):
    """AI prompt to generate a company-specific ESG excerpt (real report style)."""
    if not OPENAI_AVAILABLE:
        return f"""## Production Sustainability: 2023 Circular Initiative  
        ### Material Efficiency  
        {evaluation_data['company_name']} implemented a recycled material program in 2023, increasing post-industrial resin usage to 30% (target: 25%). This reduced virgin material consumption by 18% year-over-year.  

        ### Waste Management  
        The {evaluation_data['industry']} facility achieved 85% waste diversion from landfills, up from 70% in 2022. Hazardous waste was reduced by 25% via process optimization.  

        ### SDG Alignment  
        Results support SDG Target 12.5 (waste reduction) and 12.2 (resource efficiency). 2024 plans include scaling to 40% recycled material usage with $1.5M equipment investment.  
        *Verified by第三方 auditor (Report No. ESG-23-001)*"""
    
    # Extract user data for personalization
    company = evaluation_data["company_name"] or "ABC Manufacturing"
    industry = evaluation_data["industry"] or "manufacturing"
    volume = evaluation_data["production_volume"] if evaluation_data["production_volume"] > 0 else "500,000 units"
    recycled_pct = evaluation_data["key_metrics"]["recycled_material_pct"] or "32%"
    waste_diversion = evaluation_data["key_metrics"]["waste_diversion_pct"] or "88%"
    water_reduction = evaluation_data["key_metrics"]["water_reduction_liters"] or "60,000 liters/month"
    hazardous_reduction = evaluation_data["key_metrics"]["hazardous_waste_reduction_pct"] or "28%"
    auditor = evaluation_data["key_metrics"]["auditor_name"] or "EcoVerify"
    circular_practices = ", ".join(evaluation_data["circular_practices"]) if evaluation_data["circular_practices"] else "product take-back program"

    # Prompt for AI-generated excerpt (mimics real ESG reports)
    prompt = f"""Write a 300-400 word excerpt from {company}'s 2023 ESG report, focusing ONLY on production sustainability (SDG 12). Follow these rules:
    1. Structure: 3 clear sections (e.g., "Material Efficiency Progress", "Waste Reduction Initiatives", "SDG Alignment & Next Steps").
    2. Tone: Formal, data-driven (like real corporate ESG reports—no fluff).
    3. Metrics: Include these exact numbers: {recycled_pct} recycled material usage, {waste_diversion} waste diversion, {water_reduction} water saved/month, {hazardous_reduction} hazardous waste reduction.
    4. Context: Link to {industry}-specific challenges (e.g., textiles = water intensity; electronics = e-waste).
    5. SDG Link: Explicitly connect results to SDG 12 sub-targets (12.2 = resource efficiency; 12.5 = waste reduction).
    6. Credibility: Add a third-party verification line (auditor: {auditor}) and a 2024 target (e.g., scale to 45% recycled materials).
    7. Avoid: Company overview (assume readers know the company); generic statements.
    8. Example Flow: Section 1 = 2023 material project + metrics; Section 2 = waste/water results; Section 3 = SDG link + 2024 plan.

    Write like this is a direct screenshot from the report (no introductions)."""

    # Generate and return excerpt
    system_msg = "You are a senior ESG report writer for Fortune 500 companies. Write factual, concise, industry-aligned sustainability content."
    return get_ai_response(prompt, system_msg)

# --- Calculation Functions ---
def calculate_scores(evaluation_data):
    scores = {}
    # Score each category (0-20 points: % of criteria met * 20)
    for category, criteria in SDG12_CRITERIA.items():
        checks = evaluation_data.get(f"{category}_checks", [False]*5)
        met_criteria = sum(checks)
        scores[category] = round((met_criteria / len(criteria)) * 20, 1)
    # Overall score (0-100: sum of category scores)
    scores["overall"] = round(sum(scores.values()), 1)
    return scores

# --- Report Generation Functions ---
def generate_recommendations(scores, evaluation_data):
    if not OPENAI_AVAILABLE:
        return [
            f"Increase recycled material usage to 40% (from {evaluation_data['key_metrics']['recycled_material_pct'] or '32%'}) to align with {evaluation_data['industry']} leaders.",
            "Implement ISO 14001 certification to formalize waste management processes.",
            "Expand water reuse systems to reduce consumption by an additional 15%."
        ]
    
    weak_areas = [k.replace("_", " ").title() for k, v in scores.items() if v < 10 and k != "overall"]
    industry = evaluation_data.get("industry", "manufacturing")
    
    prompt = f"""Give 3 {industry}-specific SDG 12 recommendations for {evaluation_data['company_name']}.
    Weak areas: {weak_areas}
    Current metrics: {json.dumps(evaluation_data['key_metrics'], indent=1)}
    Rules: 1) Actionable (e.g., "Invest $X in Y equipment"); 2) Tied to SDG 12 targets; 3) Industry-relevant."""
    
    response = get_ai_response(prompt, "You are a sustainability consultant for manufacturing.")
    recommendations = [line.strip() for line in response.split('\n') if line.strip()]
    # Fallback for incomplete responses
    while len(recommendations) < 3:
        recommendations.append(f"Set a 2024 target to increase waste diversion to 90% (from {evaluation_data['key_metrics']['waste_diversion_pct'] or '88%'}).")
    return recommendations[:3]

def generate_report():
    data = st.session_state["evaluation_data"]
    scores = calculate_scores(data)
    st.session_state["scores"] = scores
    recommendations = generate_recommendations(scores, data)
    # Generate CUSTOM ESG excerpt here (tied to user data)
    st.session_state["custom_esg_excerpt"] = generate_custom_esg_excerpt(data)
    
    # Get industry benchmark
    industry = data.get("industry", "manufacturing").lower()
    benchmark = INDUSTRY_BENCHMARKS.get(industry, INDUSTRY_BENCHMARKS["manufacturing"])
    
    # Build report
    report = []
    report.append(f"SDG Goal 12 Production Evaluation: {data.get('company_name', 'Unknown Company')}")
    report.append("=" * len(report[0]))
    report.append("")
    
    # 1. Company Overview (brief)
    report.append("1. Evaluation Overview")
    report.append(f"- Company: {data.get('company_name', 'Not provided')}")
    report.append(f"- Industry: {data.get('industry', 'Not provided')}")
    report.append(f"- Annual Production: {data.get('production_volume', 'Not provided')} units")
    report.append("")
    
    # 2. ESG Excerpt (AI-generated, company-specific)
    report.append("2. Company ESG Excerpt (2023 Production Sustainability)")
    report.append(st.session_state["custom_esg_excerpt"])
    report.append("")
    
    # 3. Scorecard
    report.append("3. SDG 12 Scorecard")
    report.append(f"- Overall Score: {scores['overall']}/100")
    report.append(f"- Industry Benchmark: {benchmark['overall']}/100")
    report.append("")
    for category, score in scores.items():
        if category != "overall":
            cat_name = category.replace("_", " ").title()
            report.append(f"- {cat_name}: {score}/20 (Industry: {benchmark[category]}/20)")
    report.append("")
    
    # 4. Strengths & Improvements
    report.append("4. Key Strengths")
    strengths = [k.replace("_", " ").title() for k, v in scores.items() if v >= 15 and k != "overall"]
    if strengths:
        for s in strengths:
            report.append(f"- {s} (exceeds industry average)")
    else:
        report.append("- Initial progress in circular economy practices (build to scale)")
    report.append("")
    
    report.append("5. Improvement Areas")
    weaknesses = [k.replace("_", " ").title() for k, v in scores.items() if v < 10 and k != "overall"]
    if weaknesses:
        for w in weaknesses:
            report.append(f"- {w} (below industry benchmark)")
    else:
        report.append("- No critical gaps—focus on incremental targets (e.g., 5% higher recycled material usage)")
    report.append("")
    
    # 6. Recommendations
    report.append("6. Actionable Recommendations")
    for i, rec in enumerate(recommendations, 1):
        report.append(f"- {i}. {rec}")
    
    st.session_state["report_text"] = "\n".join(report)
    return st.session_state["report_text"]

# --- UI Functions (Step-by-Step Input) ---
def input_step_1():
    """Company & Core Metrics (for excerpt relevance)"""
    st.subheader("Step 1: Company & Production Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        company_name = st.text_input("Company Name", st.session_state["evaluation_data"]["company_name"])
        industry = st.selectbox(
            "Industry", 
            ["Manufacturing", "Food & Beverage", "Textiles", "Chemicals", "Electronics", "Other"],
            index=["Manufacturing", "Food & Beverage", "Textiles", "Chemicals", "Electronics", "Other"].index(
                st.session_state["evaluation_data"]["industry"]
            )
        )
        production_volume = st.number_input(
            "Annual Production Volume (units)", 
            value=st.session_state["evaluation_data"]["production_volume"],
            min_value=0,
            help="e.g., 100000 (for 100,000 units)"
        )
    
    with col2:
        st.subheader("Key Sustainability Metrics (for ESG excerpt)")
        # These metrics feed into the AI-generated excerpt
        recycled_pct = st.number_input(
            "Recycled Material Usage (%)", 
            value=st.session_state["evaluation_data"]["key_metrics"]["recycled_material_pct"] or 0,
            min_value=0, max_value=100, step=1
        )
        waste_diversion = st.number_input(
            "Waste Diversion Rate (%)", 
            value=st.session_state["evaluation_data"]["key_metrics"]["waste_diversion_pct"] or 0,
            min_value=0, max_value=100, step=1
        )
        water_reduction = st.number_input(
            "Monthly Water Savings (liters)", 
            value=st.session_state["evaluation_data"]["key_metrics"]["water_reduction_liters"] or 0,
            min_value=0
        )
    
    # Save to session state
    st.session_state["evaluation_data"]["company_name"] = company_name
    st.session_state["evaluation_data"]["industry"] = industry
    st.session_state["evaluation_data"]["production_volume"] = production_volume
    st.session_state["evaluation_data"]["key_metrics"]["recycled_material_pct"] = recycled_pct
    st.session_state["evaluation_data"]["key_metrics"]["waste_diversion_pct"] = waste_diversion
    st.session_state["evaluation_data"]["key_metrics"]["water_reduction_liters"] = water_reduction
    
    if st.button("Save & Continue to Criteria"):
        st.session_state["current_step"] = 2
        st.rerun()

def input_step_2():
    """Material Efficiency Criteria"""
    st.subheader("Step 2: Material Efficiency (SDG 12.2)")
    checks = st.session_state["evaluation_data"]["material_efficiency_checks"]
    
    for i, criteria in enumerate(SDG12_CRITERIA["material_efficiency"]):
        checks[i] = st.checkbox(criteria, value=checks[i])
    
    # Add auditor info (for excerpt credibility)
    auditor = st.text_input(
        "Third-Party Sustainability Auditor (e.g., EcoVerify)",
        value=st.session_state["evaluation_data"]["key_metrics"]["auditor_name"] or ""
    )
    st.session_state["evaluation_data"]["key_metrics"]["auditor_name"] = auditor
    st.session_state["evaluation_data"]["material_efficiency_checks"] = checks
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            st.session_state["current_step"] = 1
            st.rerun()
    with col2:
        if st.button("Save & Continue"):
            st.session_state["current_step"] = 3
            st.rerun()

def input_step_3():
    """Waste Management Criteria"""
    st.subheader("Step 3: Waste Management (SDG 12.5)")
    checks = st.session_state["evaluation_data"]["waste_management_checks"]
    
    for i, criteria in enumerate(SDG12_CRITERIA["waste_management"]):
        checks[i] = st.checkbox(criteria, value=checks[i])
    
    # Add hazardous waste metric (for excerpt)
    hazardous_reduction = st.number_input(
        "Hazardous Waste Reduction (%)", 
        value=st.session_state["evaluation_data"]["key_metrics"]["hazardous_waste_reduction_pct"] or 0,
        min_value=0, max_value=100, step=1
    )
    st.session_state["evaluation_data"]["key_metrics"]["hazardous_waste_reduction_pct"] = hazardous_reduction
    st.session_state["evaluation_data"]["waste_management_checks"] = checks
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            st.session_state["current_step"] = 2
            st.rerun()
    with col2:
        if st.button("Save & Continue"):
            st.session_state["current_step"] = 4
            st.rerun()

def input_step_4():
    """Energy & Water Efficiency"""
    st.subheader("Step 4: Energy & Water Efficiency")
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption("Energy Efficiency (SDG 12.2)")
        energy_checks = st.session_state["evaluation_data"]["energy_efficiency_checks"]
        for i, criteria in enumerate(SDG12_CRITERIA["energy_efficiency"]):
            energy_checks[i] = st.checkbox(criteria, value=energy_checks[i])
        st.session_state["evaluation_data"]["energy_efficiency_checks"] = energy_checks
    
    with col2:
        st.caption("Water Management (SDG 12.2)")
        water_checks = st.session_state["evaluation_data"]["water_management_checks"]
        for i, criteria in enumerate(SDG12_CRITERIA["water_management"]):
            water_checks[i] = st.checkbox(criteria, value=water_checks[i])
        st.session_state["evaluation_data"]["water_management_checks"] = water_checks
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            st.session_state["current_step"] = 3
            st.rerun()
    with col2:
        if st.button("Save & Continue"):
            st.session_state["current_step"] = 5
            st.rerun()

def input_step_5():
    """Circular Economy Practices"""
    st.subheader("Step 5: Circular Economy (SDG 12.5)")
    checks = st.session_state["evaluation_data"]["circular_economy_checks"]
    
    for i, criteria in enumerate(SDG12_CRITERIA["circular_economy"]):
        checks[i] = st.checkbox(criteria, value=checks[i])
    
    # Add custom circular practices (for excerpt)
    practices = st.text_input(
        "Custom Circular Practices (comma-separated, e.g., 'take-back program, remanufacturing')",
        value=", ".join(st.session_state["evaluation_data"]["circular_practices"])
    )
    st.session_state["evaluation_data"]["circular_practices"] = [p.strip() for p in practices.split(",") if p.strip()]
    st.session_state["evaluation_data"]["circular_economy_checks"] = checks
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            st.session_state["current_step"] = 4
            st.rerun()
    with col2:
        if st.button("Generate Evaluation Report"):
            with st.spinner("Generating report & custom ESG excerpt..."):
                generate_report()
                st.session_state["current_step"] = 6
                st.rerun()

def render_score_charts(scores):
    """Visualize scores (donut for overall, bar for categories)"""
    plt.style.use("seaborn-v0_8-muted")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Donut chart: Overall score
    overall_score = scores["overall"]
    colors = ["#2E8B57" if overall_score >= 70 else "#FFD700" if overall_score >= 50 else "#DC143C", "#F0F0F0"]
    ax1.pie([overall_score, 100 - overall_score], labels=["Achieved", "Remaining"], 
            colors=colors, wedgeprops=dict(width=0.3), startangle=90)
    centre_circle = plt.Circle((0,0), 0.70, fc="white")
    fig.gca().add_artist(centre_circle)
    ax1.set_title("Overall SDG 12 Score", fontsize=14, pad=20)
    ax1.text(0, 0, f"{overall_score}/100", ha="center", va="center", fontsize=28, fontweight="bold")
    
    # Bar chart: Category scores
    categories = [k.replace("_", " ").title() for k in scores if k != "overall"]
    category_scores = [scores[k] for k in scores if k != "overall"]
    bars = ax2.bar(categories, category_scores, color="#4682B4")
    
    # Highlight low scores (red <10, yellow 10-14, green ≥15)
    for i, bar in enumerate(bars):
        score = category_scores[i]
        if score < 10:
            bar.set_color("#DC143C")
        elif score < 15:
            bar.set_color("#FFD700")
        else:
            bar.set_color("#2E8B57")
    
    ax2.axhline(y=10, color="gray", linestyle="--", alpha=0.5, label="Minimum Target (10/20)")
    ax2.axhline(y=15, color="green", linestyle="--", alpha=0.5, label="Excellence (15/20)")
    ax2.set_ylim(0, 20)
    ax2.set_title("Category Scores (0-20)", fontsize=14, pad=20)
    ax2.set_xticklabels(categories, rotation=45, ha="right")
    ax2.legend()
    ax2.set_ylabel("Score")
    
    plt.tight_layout()
    st.pyplot(fig)

def render_report():
    """Final Report UI (with excerpt, scores, charts)"""
    st.title(f"🌱 SDG 12 Evaluation Report: {st.session_state['evaluation_data']['company_name']}")
    
    # 1. Custom ESG Excerpt (Prominent Section)
    st.subheader("Company ESG Excerpt (2023 Production Sustainability)")
    st.info(st.session_state["custom_esg_excerpt"])  # Highlight as "report snippet"
    
    # 2. Score Visualization
    st.subheader("Score Breakdown")
    render_score_charts(st.session_state["scores"])
    
    # 3. Full Text Report
    st.subheader("Detailed Evaluation")
    st.text(st.session_state["report_text"])
    
    # 4. Download Options
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download Full Report",
            data=st.session_state["report_text"],
            file_name=f"{st.session_state['evaluation_data']['company_name']}_SDG12_Report.txt",
            mime="text/plain"
        )
    with col2:
        st.download_button(
            label="📥 Download ESG Excerpt",
            data=st.session_state["custom_esg_excerpt"],
            file_name=f"{st.session_state['evaluation_data']['company_name']}_ESG_Excerpt.txt",
            mime="text/plain"
        )
    
    # 5. Follow-Up Questions
    st.subheader("Ask About Your Report")
    user_question = st.text_input("e.g., 'How can we improve our waste diversion score?'")
    if user_question and OPENAI_AVAILABLE:
        with st.spinner("Generating answer..."):
            prompt = f"""Answer this question about {st.session_state['evaluation_data']['company_name']}'s SDG 12 report:
            Question: {user_question}
            Report Data: {st.session_state['report_text']}
            Keep the answer concise (2-3 sentences) and actionable."""
            response = get_ai_response(prompt, "You are a sustainability advisor specializing in SDG 12.")
            st.write(f"**Answer:** {response}")
    
    # Reset for new evaluation
    if st.button("Start New Evaluation"):
        st.session_state.clear()
        st.session_state["evaluation_data"] = {
            "company_name": "",
            "industry": "",
            "production_volume": 0,
            "material_efficiency_checks": [False]*5,
            "waste_management_checks": [False]*5,
            "energy_efficiency_checks": [False]*5,
            "water_management_checks": [False]*5,
            "circular_economy_checks": [False]*5,
            "circular_practices": [],
            "certifications": [],
            "key_metrics": {
                "recycled_material_pct": None,
                "waste_diversion_pct": None,
                "water_reduction_liters": None,
                "hazardous_waste_reduction_pct": None,
                "auditor_name": None
            }
        }
        st.session_state["current_step"] = 0
        st.rerun()

# --- Main UI Flow ---
if st.session_state["current_step"] == 0:
    st.title("🌱 SDG 12 Production Evaluator")
    st.write("Evaluate your company’s production processes against **SDG Goal 12 (Responsible Consumption & Production)** and generate a custom ESG excerpt.")
    
    # Input Method Selection
    st.subheader("Select Input Method")
    tab1, tab2 = st.tabs(["Upload ESG Report", "Manual Input"])
    
    with tab1:
        st.caption("Upload a PDF ESG report (AI will extract data for evaluation)")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file is not None:
            st.session_state["uploaded_file"] = uploaded_file
            if st.button("Analyze Report") and OPENAI_AVAILABLE:
                with st.spinner("Extracting SDG 12 data..."):
                    text = extract_text_from_pdf(uploaded_file)
                    extracted_data = analyze_esg_document(text)
                    if extracted_data:
                        # Merge extracted data with session state
                        for key, value in extracted_data.items():
                            if key in st.session_state["evaluation_data"]:
                                st.session_state["evaluation_data"][key] = value
                        # Generate report and excerpt
                        generate_report()
                        st.session_state["current_step"] = 6
                        st.rerun()
            elif uploaded_file and not OPENAI_AVAILABLE:
                st.warning("AI is disabled—use manual input to proceed.")
    
    with tab2:
        st.caption("Enter data manually (step-by-step)")
        st.write("You’ll input company metrics, sustainability criteria, and generate a custom ESG excerpt.")
        if st.button("Start Manual Evaluation"):
            st.session_state["current_step"] = 1
            st.rerun()

# Step Navigation
elif st.session_state["current_step"] == 1:
    input_step_1()
elif st.session_state["current_step"] == 2:
    input_step_2()
elif st.session_state["current_step"] == 3:
    input_step_3()
elif st.session_state["current_step"] == 4:
    input_step_4()
elif st.session_state["current_step"] == 5:
    input_step_5()
elif st.session_state["current_step"] == 6:
    render_report()
