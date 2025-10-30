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
    st.warning("⚠️ OPENAI_API_KEY not found in Streamlit Secrets. AI features disabled.")
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
        "circular_practices": [],
        "material_efficiency_checks": [False]*5,
        "waste_management_checks": [False]*5,
        "energy_efficiency_checks": [False]*5,
        "water_management_checks": [False]*5,
        "circular_economy_checks": [False]*5
    }
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0
if "report_text" not in st.session_state:
    st.session_state["report_text"] = ""
if "scores" not in st.session_state:
    st.session_state["scores"] = {}
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None
if "custom_esg_excerpt" not in st.session_state:
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

# Simplified industry benchmarks (direct 0-100 scale)
INDUSTRY_BENCHMARKS = {
    "manufacturing": 65,
    "food & beverage": 70,
    "textiles": 55,
    "chemicals": 60,
    "electronics": 62
}

# --- Core AI Functions ---
def get_ai_response(prompt, system_msg="You are a helpful assistant."):
    if not OPENAI_AVAILABLE:
        return "AI features require an OPENAI_API_KEY."
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            temperature=0.3,
            timeout=20
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"AI error: {str(e)}")
        return "Failed to generate AI response."

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def analyze_esg_document(text):
    prompt = f"""Analyze this ESG report text and extract SDG 12 production data:
    - Company name
    - Industry
    - Production volume
    - For each category, mark which criteria are met (true/false):
    {json.dumps(SDG12_CRITERIA, indent=2)}
    
    Return as JSON only."""
    
    try:
        response = get_ai_response(prompt, "You are an ESG analyst specializing in SDG Goal 12.")
        return json.loads(response)
    except Exception as e:
        st.error(f"Document analysis error: {str(e)}")
        return {}

def ai_generate_mock_esg(evaluation_data):
    """Generate company-specific ESG excerpt"""
    if not OPENAI_AVAILABLE:
        return """## 2023 Production Sustainability Initiative

### Material Efficiency
Our facility achieved a 22% reduction in virgin material usage through recycled input integration (38% of total materials), exceeding our 15% target.

### Waste & Resource Management
- 91% waste diversion from landfills (up from 78% in 2022)
- 31% reduction in hazardous waste
- 45,000 liters/month water savings via closed-loop systems

### SDG Alignment
Results support SDG Target 12.2 (resource efficiency) and 12.5 (waste reduction). 2024 plans include scaling to 45% recycled materials with $2.8M investment.

*Verified by EcoVerify (Report No. EV-23-7842)*"""
    
    company = evaluation_data.get("company_name", "Sustainable Manufacturing Inc.")
    industry = evaluation_data.get("industry", "manufacturing")
    
    # Calculate simple achievement percentages
    material_pct = int((sum(evaluation_data["material_efficiency_checks"]) / 5) * 100)
    waste_pct = int((sum(evaluation_data["waste_management_checks"]) / 5) * 100)
    
    prompt = f"""Generate a 300-word ESG excerpt for {company} ({industry}) focusing on SDG 12 production efforts.
    Include:
    1. Material efficiency results with {material_pct}% achievement metric
    2. Waste reduction with {waste_pct}% diversion rate
    3. Specific SDG 12 targets (12.2, 12.5)
    4. 2024 goals with investment figure
    5. Third-party verification
    Do not include company overview."""
    
    return get_ai_response(prompt, "You are an ESG report writer specializing in sustainable production.")

# --- Score Calculation (No Weighting - Direct 0-100) ---
def calculate_scores(evaluation_data):
    scores = {}
    
    # Each category has 5 criteria = 20 points per category (5×20=100 total)
    for category in SDG12_CRITERIA.keys():
        checks = evaluation_data.get(f"{category}_checks", [False]*5)
        met_criteria = sum(checks)
        # 20 points per category (each criterion = 4 points: 20/5=4)
        scores[category] = met_criteria * 4
    
    # Overall score = sum of all category scores (0-100)
    scores["overall"] = sum(scores.values())
    
    return scores

# --- Report Generation ---
def generate_recommendations(scores, evaluation_data):
    if not OPENAI_AVAILABLE:
        return [
            "Increase recycled material usage in production processes.",
            "Implement formal energy tracking systems.",
            "Develop product take-back programs for circular economy."
        ]
    
    weak_areas = [k.replace("_", " ").title() for k, v in scores.items() if v < 10 and k != "overall"]
    industry = evaluation_data.get("industry", "manufacturing")
    
    prompt = f"Give 3 {industry}-specific SDG 12 recommendations. Weak areas: {weak_areas}"
    response = get_ai_response(prompt, "You are a sustainability consultant.")
    return [line.strip() for line in response.split('\n') if line.strip()][:3]

def generate_report():
    data = st.session_state["evaluation_data"]
    scores = calculate_scores(data)
    st.session_state["scores"] = scores
    recommendations = generate_recommendations(scores, data)
    st.session_state["custom_esg_excerpt"] = ai_generate_mock_esg(data)
    
    # Get industry benchmark
    industry = data.get("industry", "manufacturing").lower()
    benchmark = INDUSTRY_BENCHMARKS.get(industry, 60)
    
    # Build report
    report = []
    report.append(f"SDG Goal 12 Evaluation: {data.get('company_name', 'Unknown Company')}")
    report.append("=" * len(report[0]))
    report.append("")
    
    # Overview
    report.append("1. Overview")
    report.append(f"- Company: {data.get('company_name', 'Not provided')}")
    report.append(f"- Industry: {data.get('industry', 'Not provided')}")
    report.append(f"- Production Volume: {data.get('production_volume', 'Not provided')}")
    report.append("")
    
    # ESG Excerpt
    report.append("2. ESG Report Excerpt (Production Sustainability)")
    report.append(st.session_state["custom_esg_excerpt"])
    report.append("")
    
    # Scorecard (0-100 scale)
    report.append("3. SDG 12 Scorecard")
    report.append(f"- Overall Score: {scores['overall']}/100")
    report.append(f"- Industry Benchmark: {benchmark}/100")
    report.append("")
    for category, score in scores.items():
        if category != "overall":
            cat_name = category.replace("_", " ").title()
            report.append(f"- {cat_name}: {score}/20")  # 20 points per category
    report.append("")
    
    # Strengths & Improvements
    report.append("4. Key Strengths")
    strengths = [k.replace("_", " ").title() for k, v in scores.items() if v >= 15 and k != "overall"]
    if strengths:
        for s in strengths:
            report.append(f"- Strong performance in {s}")
    else:
        report.append("- Identify initial sustainability practices to build upon")
    report.append("")
    
    report.append("5. Improvement Areas")
    weaknesses = [k.replace("_", " ").title() for k, v in scores.items() if v < 10 and k != "overall"]
    if weaknesses:
        for w in weaknesses:
            report.append(f"- {w} requires attention")
    else:
        report.append("- Maintain current practices and set higher targets")
    report.append("")
    
    # Recommendations
    report.append("6. Recommendations")
    for i, rec in enumerate(recommendations, 1):
        report.append(f"- {i}. {rec}")
    
    st.session_state["report_text"] = "\n".join(report)
    return st.session_state["report_text"]

# --- UI Functions ---
def render_score_charts(scores):
    # Create visualizations with 0-100 scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Donut chart for overall score (0-100)
    overall_score = scores['overall']
    ax1.pie([overall_score, 100 - overall_score], 
            labels=['Achieved', 'Remaining'], 
            colors=['#4CAF50', '#f0f0f0'], 
            wedgeprops=dict(width=0.3))
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    ax1.set_title('Overall SDG 12 Score')
    ax1.text(0, 0, f'{overall_score}/100', ha='center', va='center', fontsize=24)
    
    # Bar chart for category scores (0-20 each)
    categories = [k.replace("_", " ").title() for k in scores if k != "overall"]
    category_scores = [scores[k] for k in scores if k != "overall"]
    
    ax2.bar(categories, category_scores, color='#2196F3')
    ax2.axhline(y=10, color='orange', linestyle='--', label='Minimum Target')
    ax2.axhline(y=15, color='green', linestyle='--', label='Excellent')
    ax2.set_ylim(0, 20)
    ax2.set_title('Category Scores (0-20)')
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

def input_step_1():
    st.subheader("Step 1: Company Information")
    col1, col2 = st.columns(2)
    with col1:
        company_name = st.text_input("Company Name", st.session_state["evaluation_data"]["company_name"])
    with col2:
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
        min_value=0
    )
    
    if st.button("Save & Continue"):
        st.session_state["evaluation_data"]["company_name"] = company_name
        st.session_state["evaluation_data"]["industry"] = industry
        st.session_state["evaluation_data"]["production_volume"] = production_volume
        st.session_state["current_step"] = 2
        st.rerun()

# Step 2-6: Criteria Input (unchanged structure, simplified scoring logic)
def input_step_2():
    st.subheader("Step 2: Material Efficiency")
    checks = st.session_state["evaluation_data"]["material_efficiency_checks"]
    for i, criteria in enumerate(SDG12_CRITERIA["material_efficiency"]):
        checks[i] = st.checkbox(criteria, value=checks[i])
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
    st.subheader("Step 3: Waste Management")
    checks = st.session_state["evaluation_data"]["waste_management_checks"]
    for i, criteria in enumerate(SDG12_CRITERIA["waste_management"]):
        checks[i] = st.checkbox(criteria, value=checks[i])
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
    st.subheader("Step 4: Energy Efficiency")
    checks = st.session_state["evaluation_data"]["energy_efficiency_checks"]
    for i, criteria in enumerate(SDG12_CRITERIA["energy_efficiency"]):
        checks[i] = st.checkbox(criteria, value=checks[i])
    st.session_state["evaluation_data"]["energy_efficiency_checks"] = checks
    
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
    st.subheader("Step 5: Water Management")
    checks = st.session_state["evaluation_data"]["water_management_checks"]
    for i, criteria in enumerate(SDG12_CRITERIA["water_management"]):
        checks[i] = st.checkbox(criteria, value=checks[i])
    st.session_state["evaluation_data"]["water_management_checks"] = checks
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            st.session_state["current_step"] = 4
            st.rerun()
    with col2:
        if st.button("Save & Continue"):
            st.session_state["current_step"] = 6
            st.rerun()

def input_step_6():
    st.subheader("Step 6: Circular Economy Practices")
    checks = st.session_state["evaluation_data"]["circular_economy_checks"]
    for i, criteria in enumerate(SDG12_CRITERIA["circular_economy"]):
        checks[i] = st.checkbox(criteria, value=checks[i])
    
    practices = st.text_input(
        "Additional circular practices (optional)", 
        ", ".join(st.session_state["evaluation_data"]["circular_practices"])
    )
    st.session_state["evaluation_data"]["circular_practices"] = [p.strip() for p in practices.split(",") if p.strip()]
    st.session_state["evaluation_data"]["circular_economy_checks"] = checks
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            st.session_state["current_step"] = 5
            st.rerun()
    with col2:
        if st.button("Generate Report"):
            generate_report()
            st.session_state["current_step"] = 7
            st.rerun()

def render_report():
    st.subheader("SDG Goal 12 Evaluation Report")
    st.text(st.session_state["report_text"])
    
    st.subheader("Score Visualization")
    render_score_charts(st.session_state["scores"])
    
    # Download
    st.download_button(
        label="Download Report",
        data=st.session_state["report_text"],
        file_name=f"{st.session_state['evaluation_data']['company_name']}_sdg12_evaluation.txt"
    )
    
    # Follow-up
    user_question = st.text_input("Ask a question about the report...")
    if user_question and OPENAI_AVAILABLE:
        with st.spinner("Generating answer..."):
            response = get_ai_response(user_question, f"Answer about this report: {st.session_state['report_text']}")
            st.write(f"**Answer:** {response}")

# --- Main UI ---
st.title("🌱 SDG Goal 12 Production Evaluator")
st.write("Evaluate production processes against SDG 12 (Responsible Consumption and Production)")

if st.session_state["current_step"] == 0:
    st.subheader("Select Input Method")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload ESG Report")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file and st.button("Analyze Document") and OPENAI_AVAILABLE:
            with st.spinner("Analyzing..."):
                text = extract_text_from_pdf(uploaded_file)
                result = analyze_esg_document(text)
                if result:
                    st.session_state["evaluation_data"] = result
                    st.session_state["current_step"] = 7
                    generate_report()
                    st.rerun()
    
    with col2:
        st.subheader("Manual Input")
        if st.button("Start Evaluation"):
            st.session_state["current_step"] = 1
            st.rerun()
    
    with st.expander("View Sample ESG Excerpt"):
        st.text("""## 2023 Production Sustainability Initiative

### Material Efficiency
Our facility achieved a 22% reduction in virgin material usage through recycled input integration (38% of total materials), exceeding our 15% target.

### Waste & Resource Management
- 91% waste diversion from landfills (up from 78% in 2022)
- 31% reduction in hazardous waste
- 45,000 liters/month water savings via closed-loop systems

### SDG Alignment
Results support SDG Target 12.2 (resource efficiency) and 12.5 (waste reduction). 2024 plans include scaling to 45% recycled materials with $2.8M investment.

*Verified by EcoVerify (Report No. EV-23-7842)*""")

# Step navigation
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
    input_step_6()
elif st.session_state["current_step"] == 7:
    render_report()
    if st.button("Start New Evaluation"):
        st.session_state.clear()
        st.session_state["evaluation_data"] = {
            "company_name": "",
            "industry": "",
            "production_volume": 0,
            "circular_practices": [],
            "material_efficiency_checks": [False]*5,
            "waste_management_checks": [False]*5,
            "energy_efficiency_checks": [False]*5,
            "water_management_checks": [False]*5,
            "circular_economy_checks": [False]*5
        }
        st.session_state["current_step"] = 0
        st.rerun()

# Progress indicator
if 1 <= st.session_state["current_step"] <= 6:
    st.sidebar.progress(st.session_state["current_step"] / 6)
    st.sidebar.write(f"Step {st.session_state['current_step']}/6")
