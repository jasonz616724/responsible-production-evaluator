import streamlit as st
import json
import re  # New: For cleaning AI responses
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import io

# Handle PDF functionality with fallback
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("⚠️ PyPDF2 library not found. PDF upload disabled. Install with: pip install PyPDF2")

# --- Page Configuration ---
st.set_page_config(page_title="SDG 12 Production Evaluator", layout="wide")

# --- Initialize OpenAI Client ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except KeyError:
    st.warning("⚠️ OPENAI_API_KEY not found. AI features disabled.")
    OPENAI_AVAILABLE = False
except Exception as e:
    st.error(f"⚠️ OpenAI client error: {str(e)}")
    OPENAI_AVAILABLE = False

# --- Session State Initialization ---
if "evaluation_data" not in st.session_state:
    st.session_state["evaluation_data"] = {
        "company_name": "",
        "industry": "Manufacturing",
        "production_volume": 0,
        "circular_practices": [],
        "material_efficiency_checks": [False]*5,
        "waste_management_checks": [False]*5,
        "energy_efficiency_checks": [False]*5,
        "water_management_checks": [False]*5,
        "circular_economy_checks": [False]*5,
        "project_details": {
            "project_name": "",
            "timeframe": "2023 Q1-Q4",
            "investment": 0,
            "impact_metrics": {}
        }
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
            temperature=0.2,
            timeout=20
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"AI error: {str(e)}")
        return ""  # Return empty string to trigger fallback

def extract_text_from_pdf(file):
    if not PDF_AVAILABLE:
        return ""
    
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
        # Validate extracted text quality
        if len(text.strip()) < 500:  # Too short to contain meaningful data
            st.warning("⚠️ Extracted text is very short. May not contain enough information.")
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def analyze_esg_document(text):
    """Improved document analysis with JSON validation and error handling"""
    if len(text.strip()) < 300:
        st.error("❌ Not enough text extracted from PDF to analyze. Please use manual input.")
        return {}

    prompt = f"""Analyze this ESG report text and extract SDG 12 production data.
    RETURN ONLY A VALID JSON OBJECT (NO EXPLANATIONS, NO MARKDOWN).
    
    Required fields:
    - company_name (string)
    - industry (string)
    - production_volume (number, or 0 if unknown)
    - material_efficiency_checks (array of 5 booleans)
    - waste_management_checks (array of 5 booleans)
    - energy_efficiency_checks (array of 5 booleans)
    - water_management_checks (array of 5 booleans)
    - circular_economy_checks (array of 5 booleans)
    - project_details (object with project_name, timeframe, investment)
    
    Criteria for checks (map each to true/false if met):
    {json.dumps(SDG12_CRITERIA, indent=2)}
    """
    
    try:
        # Get AI response with strict JSON instruction
        response = get_ai_response(
            prompt, 
            "You are an ESG data extractor. Return ONLY valid JSON. No extra text."
        )
        
        if not response:
            st.error("❌ AI returned empty response. Could not analyze document.")
            return {}

        # Clean response: Remove any non-JSON prefix/suffix (common in AI outputs)
        # Use regex to find the first { and last } to extract valid JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            st.error(f"❌ AI response is not valid JSON. Raw response: {response[:200]}...")
            return {}
        
        clean_json = json_match.group()
        
        # Parse JSON with validation
        result = json.loads(clean_json)
        
        # Validate required fields exist
        required_fields = ["company_name", "industry", "material_efficiency_checks", 
                          "waste_management_checks", "energy_efficiency_checks",
                          "water_management_checks", "circular_economy_checks"]
        for field in required_fields:
            if field not in result:
                st.warning(f"⚠️ Missing field '{field}' in analysis. Using defaults.")
                result[field] = "" if field in ["company_name", "industry"] else [False]*5
        
        return result
    
    except json.JSONDecodeError as e:
        st.error(f"❌ Invalid JSON from AI: {str(e)}. Raw response: {clean_json[:200]}...")
        return {}
    except Exception as e:
        st.error(f"❌ Document analysis failed: {str(e)}")
        return {}

def ai_generate_mock_esg(evaluation_data):
    if not OPENAI_AVAILABLE:
        return """## 3.2 Circular Production Initiative: Plastic Waste Reduction Program  
        *Facility: Southeast Asia Manufacturing Hub | Timeframe: Jan-Dec 2023*  

        ### Project Overview  
        The plastic waste reduction program targeted post-production scrap and packaging waste, part of our 2025 commitment to 50% recycled material usage across product lines.  

        ### Key Achievements  
        - Implemented in-line recycling systems for polypropylene scrap, diverting 142 metric tons (MT) from landfills—equivalent to 3.2 million plastic bottles  
        - Achieved 38% recycled content in core product packaging (2023 target: 30%), reducing virgin plastic procurement by 22%  
        - Partnered with local recyclers to establish closed-loop collection, resulting in 91% waste diversion from facility operations (up from 78% in 2022)  

        ### Environmental Impact  
        - CO₂e reduction: 287 tons (avoided from virgin plastic production and waste transport)  
        - Water savings: 45,000 liters/month through reduced cleaning needs in recycling process  

        ### SDG Alignment  
        Directly supports SDG Target 12.5 (substantially reduce waste generation by 2030) and Target 12.2 (sustainable management of natural resources).  

        ### 2024 Expansion  
        $2.8M investment approved to scale systems to European facilities, targeting 45% recycled content and 95% waste diversion.  

        *Verified by SGS (Report Ref: ESG-23-1472 | Audit Date: Jan 2024)*"""
    
    company = evaluation_data.get("company_name", "Sustainable Manufacturing Inc.")
    industry = evaluation_data.get("industry", "manufacturing").lower()
    project_name = evaluation_data["project_details"].get("project_name") or "Circular Material Optimization"
    timeframe = evaluation_data["project_details"].get("timeframe") or "2023"
    investment = evaluation_data["project_details"].get("investment") or 2800000
    
    material_met = sum(evaluation_data["material_efficiency_checks"])
    waste_met = sum(evaluation_data["waste_management_checks"])
    recycled_pct = 30 + (material_met * 5)
    waste_diversion = 75 + (waste_met * 3)
    
    industry_context = {
        "manufacturing": "machinery components",
        "food & beverage": "packaging and processing waste",
        "textiles": "cotton scrap and dye waste",
        "chemicals": "by-product reprocessing",
        "electronics": "e-waste component recovery"
    }.get(industry, "production waste")
    
    prompt = f"""Write a 350-word ESG report excerpt for {company} about their {project_name} in {timeframe}.  
    Structure: Project header, overview, key achievements (3 bullets with {recycled_pct}% recycled content, {waste_diversion}% waste diversion), environmental impact, SDG alignment, 2024 plans with ${investment:,} investment, third-party verification.  
    Tone: Factual, technical. Focus on {industry_context}."""
    
    return get_ai_response(prompt, "Senior ESG report writer: Generate excerpts for real corporate reports.")

# --- Score Calculation ---
def calculate_scores(evaluation_data):
    scores = {}
    for category in SDG12_CRITERIA.keys():
        checks = evaluation_data.get(f"{category}_checks", [False]*5)
        scores[category] = sum(checks) * 4
    scores["overall"] = sum(scores.values())
    return scores

# --- Report Generation ---
def generate_recommendations(scores, evaluation_data):
    if not OPENAI_AVAILABLE:
        return [
            f"Expand recycled material sourcing to reach {min(50, 38 + 10)}% by 2024.",
            "Implement real-time waste tracking to identify reduction opportunities.",
            "$280K allocated for employee training on circular practices."
        ]
    
    weak_areas = [k.replace("_", " ").title() for k, v in scores.items() if v < 10 and k != "overall"]
    industry = evaluation_data.get("industry", "manufacturing")
    
    prompt = f"3 {industry}-specific SDG 12 recommendations. Weak areas: {weak_areas}. Include investment figures."
    response = get_ai_response(prompt, "Sustainability consultant for industrial projects.")
    return [line.strip() for line in response.split('\n') if line.strip()][:3]

def generate_report():
    data = st.session_state["evaluation_data"]
    scores = calculate_scores(data)
    st.session_state["scores"] = scores
    recommendations = generate_recommendations(scores, data)
    st.session_state["custom_esg_excerpt"] = ai_generate_mock_esg(data)
    
    industry = data.get("industry", "manufacturing").lower()
    benchmark = INDUSTRY_BENCHMARKS.get(industry, 60)
    
    report = [
        f"SDG Goal 12 Evaluation: {data.get('company_name', 'Unknown Company')}",
        "=" * len(report[0]),
        "",
        "1. Overview",
        f"- Company: {data.get('company_name', 'Not provided')}",
        f"- Industry: {data.get('industry', 'Not provided')}",
        f"- Focus Project: {data['project_details'].get('project_name') or 'Circular Production Initiative'}",
        "",
        "2. ESG Report Excerpt (Production Sustainability)",
        st.session_state["custom_esg_excerpt"],
        "",
        "3. SDG 12 Scorecard",
        f"- Overall Score: {scores['overall']}/100",
        f"- Industry Benchmark: {benchmark}/100",
        ""
    ]
    
    for category, score in scores.items():
        if category != "overall":
            report.append(f"- {category.replace('_', ' ').title()}: {score}/20")
    
    report.extend([
        "",
        "4. Project Recommendations"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        report.append(f"- {i}. {rec}")
    
    st.session_state["report_text"] = "\n".join(report)
    return st.session_state["report_text"]

# --- UI Functions ---
def render_score_charts(scores):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    overall_score = scores['overall']
    ax1.pie([overall_score, 100 - overall_score], 
            labels=['Achieved', 'Remaining'], 
            colors=['#4CAF50', '#f0f0f0'], 
            wedgeprops=dict(width=0.3))
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    ax1.set_title('Overall SDG 12 Score')
    ax1.text(0, 0, f'{overall_score}/100', ha='center', va='center', fontsize=24)
    
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
    st.subheader("Step 1: Company & Project Basics")
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
    
    with col2:
        st.caption("Focus Project (for ESG excerpt)")
        project_name = st.text_input(
            "Project Name", 
            st.session_state["evaluation_data"]["project_details"]["project_name"] or "Circular Production Initiative"
        )
        timeframe = st.text_input(
            "Timeframe", 
            st.session_state["evaluation_data"]["project_details"]["timeframe"]
        )
    
    if st.button("Save & Continue"):
        st.session_state["evaluation_data"]["company_name"] = company_name
        st.session_state["evaluation_data"]["industry"] = industry
        st.session_state["evaluation_data"]["project_details"]["project_name"] = project_name
        st.session_state["evaluation_data"]["project_details"]["timeframe"] = timeframe
        st.session_state["current_step"] = 2
        st.rerun()

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
    st.subheader("Step 4: Energy & Water Efficiency")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Energy Efficiency")
        checks = st.session_state["evaluation_data"]["energy_efficiency_checks"]
        for i, criteria in enumerate(SDG12_CRITERIA["energy_efficiency"]):
            checks[i] = st.checkbox(criteria, value=checks[i])
        st.session_state["evaluation_data"]["energy_efficiency_checks"] = checks
    
    with col2:
        st.caption("Water Management")
        checks = st.session_state["evaluation_data"]["water_management_checks"]
        for i, criteria in enumerate(SDG12_CRITERIA["water_management"]):
            checks[i] = st.checkbox(criteria, value=checks[i])
        st.session_state["evaluation_data"]["water_management_checks"] = checks
    
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
    st.subheader("Step 5: Circular Economy & Investment")
    checks = st.session_state["evaluation_data"]["circular_economy_checks"]
    for i, criteria in enumerate(SDG12_CRITERIA["circular_economy"]):
        checks[i] = st.checkbox(criteria, value=checks[i])
    st.session_state["evaluation_data"]["circular_economy_checks"] = checks
    
    investment = st.number_input(
        "Project Investment ($)",
        value=st.session_state["evaluation_data"]["project_details"]["investment"] or 2800000,
        min_value=0,
        step=100000
    )
    st.session_state["evaluation_data"]["project_details"]["investment"] = investment
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            st.session_state["current_step"] = 4
            st.rerun()
    with col2:
        if st.button("Generate Report"):
            generate_report()
            st.session_state["current_step"] = 6
            st.rerun()

def render_report():
    st.subheader(f"🌱 SDG 12 Evaluation: {st.session_state['evaluation_data']['company_name']}")
    
    st.subheader("ESG Report Excerpt (Production Project)")
    st.info(st.session_state["custom_esg_excerpt"])
    
    st.subheader("Score Breakdown")
    render_score_charts(st.session_state["scores"])
    
    st.subheader("Full Evaluation")
    st.text(st.session_state["report_text"])
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Full Report",
            data=st.session_state["report_text"],
            file_name=f"{st.session_state['evaluation_data']['company_name']}_sdg12_report.txt"
        )
    with col2:
        st.download_button(
            label="Download ESG Excerpt",
            data=st.session_state["custom_esg_excerpt"],
            file_name=f"{st.session_state['evaluation_data']['company_name']}_esg_excerpt.txt"
        )
    
    if st.button("Start New Evaluation"):
        st.session_state.clear()
        st.session_state["evaluation_data"] = {
            "company_name": "",
            "industry": "Manufacturing",
            "production_volume": 0,
            "circular_practices": [],
            "material_efficiency_checks": [False]*5,
            "waste_management_checks": [False]*5,
            "energy_efficiency_checks": [False]*5,
            "water_management_checks": [False]*5,
            "circular_economy_checks": [False]*5,
            "project_details": {
                "project_name": "",
                "timeframe": "2023 Q1-Q4",
                "investment": 0,
                "impact_metrics": {}
            }
        }
        st.session_state["current_step"] = 0
        st.rerun()

# --- Main UI ---
st.title("🌱 SDG 12 Production Evaluator")
st.write("Generate ESG report excerpts focused on specific sustainability projects aligned with SDG 12.")

if st.session_state["current_step"] == 0:
    st.subheader("Select Input Method")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload ESG Report")
        if not PDF_AVAILABLE:
            st.info("PDF upload requires PyPDF2. Install with: pip install PyPDF2")
        else:
            uploaded_file = st.file_uploader("Upload PDF", type="pdf")
            if uploaded_file and st.button("Analyze Document") and OPENAI_AVAILABLE:
                with st.spinner("Analyzing..."):
                    text = extract_text_from_pdf(uploaded_file)
                    if not text:
                        st.error("❌ Could not extract text from PDF. Please try manual input.")
                    else:
                        result = analyze_esg_document(text)
                        if result:
                            st.session_state["evaluation_data"] = result
                            industry_options = ["Manufacturing", "Food & Beverage", "Textiles", "Chemicals", "Electronics", "Other"]
                            if st.session_state["evaluation_data"].get("industry") not in industry_options:
                                st.session_state["evaluation_data"]["industry"] = "Manufacturing"
                            st.session_state["current_step"] = 6
                            generate_report()
                            st.rerun()
                        else:
                            st.warning("⚠️ Could not analyze document. Switching to manual input.")
                            st.session_state["current_step"] = 1
                            st.rerun()
    
    with col2:
        st.subheader("Manual Input")
        st.write("Enter project details to generate a realistic ESG excerpt.")
        if st.button("Start Evaluation"):
            st.session_state["current_step"] = 1
            st.rerun()
    
    with st.expander("View Sample ESG Excerpt"):
        st.text("""## 3.2 Circular Production Initiative: Plastic Waste Reduction Program  
        *Facility: Southeast Asia Manufacturing Hub | Timeframe: Jan-Dec 2023*  

        ### Project Overview  
        The plastic waste reduction program targeted post-production scrap and packaging waste, part of our 2025 commitment to 50% recycled material usage across product lines.  

        ### Key Achievements  
        - Implemented in-line recycling systems for polypropylene scrap, diverting 142 metric tons (MT) from landfills—equivalent to 3.2 million plastic bottles  
        - Achieved 38% recycled content in core product packaging (2023 target: 30%), reducing virgin plastic procurement by 22%  
        - Partnered with local recyclers to establish closed-loop collection, resulting in 91% waste diversion from facility operations (up from 78% in 2022)  

        ### Environmental Impact  
        - CO₂e reduction: 287 tons (avoided from virgin plastic production and waste transport)  
        - Water savings: 45,000 liters/month through reduced cleaning needs in recycling process  

        ### SDG Alignment  
        Directly supports SDG Target 12.5 (substantially reduce waste generation by 2030) and Target 12.2 (sustainable management of natural resources).  

        ### 2024 Expansion  
        $2.8M investment approved to scale systems to European facilities, targeting 45% recycled content and 95% waste diversion.  

        *Verified by SGS (Report Ref: ESG-23-1472 | Audit Date: Jan 2024)*""")

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

if 1 <= st.session_state["current_step"] <= 5:
    st.sidebar.progress(st.session_state["current_step"] / 5)
    st.sidebar.write(f"Step {st.session_state['current_step']}/5")
