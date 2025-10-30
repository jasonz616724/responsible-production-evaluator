import streamlit as st
import json
from openai import OpenAI
import numpy as np

# --- Page Configuration (Chatbot Style) ---
st.set_page_config(page_title="SDG 12 Chatbot Evaluator", layout="centered")
st.title("🌱 SDG 12 Chatbot Evaluator")
st.subheader("I’ll ask 5 quick questions to assess your SDG 12 performance → Get a free feedback report!")

# --- Initialize OpenAI Client (for recommendations/excerpts) ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    OPENAI_AVAILABLE = True
except KeyError:
    st.warning("⚠️ OPENAI_API_KEY not found. AI features (recommendations) disabled.")
    OPENAI_AVAILABLE = False
except Exception as e:
    st.error(f"⚠️ OpenAI error: {str(e)}")
    OPENAI_AVAILABLE = False

# --- Session State Initialization (Track Chat Progress & Data) ---
if "chat_state" not in st.session_state:
    st.session_state["chat_state"] = {
        "round": 1,  # Tracks 5 question rounds
        "data": {
            "company_name": "",
            "industry": "",
            "resource_efficiency": {"renewable_energy_pct": 0, "water_reuse_pct": 0},
            "sustainable_production": {"recycled_material_pct": 0, "waste_intensity_pct": 0},
            "circular_economy": {"takeback_program_pct": 0, "packaging_sustainable_pct": 0},
            "total_score": 0,
            "dimension_scores": {},
            "recommendations": [],
            "feedback_excerpt": ""
        },
        "completed": False  # Whether 5 rounds are done
    }

# --- SDG 12 Scoring Framework (Simplified for Chatbot) ---
DIMENSIONS = [
    {
        "id": "resource_efficiency",
        "name": "Resource Efficiency (SDG 12.2)",
        "weight": 0.3,
        "actions": [
            {"name": "renewable_energy_pct", "calc": lambda x: 10 if x >=50 else 5 if x >=30 else 0},
            {"name": "water_reuse_pct", "calc": lambda x: 10 if x >=70 else 5 if x >=40 else 0}
        ],
        "max_subtotal": 20
    },
    {
        "id": "sustainable_production",
        "name": "Sustainable Production (SDG 12.3)",
        "weight": 0.3,
        "actions": [
            {"name": "recycled_material_pct", "calc": lambda x: 10 if x >=40 else 5 if x >=20 else 0},
            {"name": "waste_intensity_pct", "calc": lambda x: 10 if x <=20 else 5 if x <=40 else 0}
        ],
        "max_subtotal": 20
    },
    {
        "id": "circular_economy",
        "name": "Circular Economy (SDG 12.5)",
        "weight": 0.4,
        "actions": [
            {"name": "takeback_program_pct", "calc": lambda x: 10 if x >=50 else 5 if x >=20 else 0},
            {"name": "packaging_sustainable_pct", "calc": lambda x: 10 if x >=80 else 5 if x >=50 else 0}
        ],
        "max_subtotal": 20
    }
]

# --- Core Functions ---
def calculate_score():
    """Calculate SDG 12 score using collected chat data"""
    data = st.session_state["chat_state"]["data"]
    dimension_scores = {}
    total_score = 0

    for dim in DIMENSIONS:
        # Calculate subtotal for each dimension
        dim_data = data[dim["id"]]
        action_subtotal = sum([action["calc"](dim_data[action["name"]]) for action in dim["actions"]])
        
        # Weighted score (total = 100)
        weighted_score = round(action_subtotal * dim["weight"], 1)
        dimension_scores[dim["id"]] = {
            "name": dim["name"],
            "subtotal": action_subtotal,
            "weighted_score": weighted_score,
            "max_weighted": round(dim["max_subtotal"] * dim["weight"], 1)
        }
        total_score += weighted_score

    # Update session state with scores
    data["dimension_scores"] = dimension_scores
    data["total_score"] = min(100, round(total_score, 1))  # Cap at 100
    st.session_state["chat_state"]["data"] = data

def generate_feedback():
    """Generate industry-specific recommendations and feedback excerpt"""
    data = st.session_state["chat_state"]["data"]
    company_name = data["company_name"]
    industry = data["industry"]
    scores = data["dimension_scores"]

    # Identify strengths/areas for improvement
    strengths = [v["name"] for v in scores.values() if v["weighted_score"] >= v["max_weighted"] * 0.7]
    improvements = [v["name"] for v in scores.values() if v["weighted_score"] < v["max_weighted"] * 0.5]

    # Fallback if AI is unavailable
    fallback_recs = [
        f"1. Increase renewable energy to 50% by 2026 (SDG 12.2) – Cuts carbon emissions and aligns with {industry} sector trends.",
        f"2. Boost recycled material use to 40% (SDG 12.3) – Reduces virgin material reliance and lowers production costs.",
        f"3. Expand product take-back programs to 50% of lines (SDG 12.5) – Enhances circularity and customer loyalty."
    ]

    # AI-generated recommendations (industry-specific)
    if OPENAI_AVAILABLE:
        prompt = f"""Generate 3 SDG 12 recommendations for {company_name} ({industry} industry).
        Strengths: {strengths if strengths else ['Resource efficiency']}
        Improvements: {improvements if improvements else ['Circular economy']}
        Include SDG 12 target, measurable goal, and impact. Number them (1., 2., 3.)."""
        
        response = OpenAI().chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        recs = [line.strip() for line in response.choices[0].message.content.split("\n") if line.strip() and line[0].isdigit()]
        data["recommendations"] = recs if recs else fallback_recs
    else:
        data["recommendations"] = fallback_recs

    # Generate feedback excerpt
    data["feedback_excerpt"] = f"""
    # SDG 12 Performance Feedback Report
    ## Company: {company_name} | Industry: {industry}
    ## Total Score: {data['total_score']}/100

    ### 1. Dimension Breakdown
    """
    for dim_id, dim_data in scores.items():
        data["feedback_excerpt"] += f"- {dim_data['name']}: {dim_data['weighted_score']}/{dim_data['max_weighted']}\n"
    
    data["feedback_excerpt"] += f"""
    ### 2. Key Strengths
    {', '.join(strengths) if strengths else 'No major strengths identified – focus on recommendations below.'}

    ### 3. Areas for Improvement
    {', '.join(improvements) if improvements else 'All dimensions are on track – aim for higher targets!'}

    ### 4. Actionable Recommendations
    """
    for rec in data["recommendations"]:
        data["feedback_excerpt"] += f"- {rec}\n"
    
    data["feedback_excerpt"] += f"""
    ### 5. Next Steps
    1. Review your scores against {industry} sector benchmarks (e.g., average renewable energy use: 35%).
    2. Prioritize 1-2 recommendations (e.g., start with sustainable packaging if scores are low).
    3. Reassess in 6 months to track progress toward SDG 12 targets.
    """
    st.session_state["chat_state"]["data"] = data

# --- Chatbot Logic (5 Rounds) ---
chat = st.session_state["chat_state"]

# Round 1: Company Name & Industry
if chat["round"] == 1 and not chat["completed"]:
    st.write("**Round 1/5: Let’s start with basics!**")
    col1, col2 = st.columns(2)
    with col1:
        company_name = st.text_input("What’s your company name?", key="round1_name")
    with col2:
        industries = ["Manufacturing", "Retail", "Transport", "Food & Beverage", "Construction", "Tourism", "Healthcare", "Other"]
        industry = st.selectbox("What industry are you in?", industries, key="round1_industry")
    
    if st.button("Next →", key="round1_next"):
        if company_name.strip() and industry:
            chat["data"]["company_name"] = company_name.strip()
            chat["data"]["industry"] = industry
            chat["round"] = 2
            st.rerun()
        else:
            st.warning("Please enter both company name and industry!")

# Round 2: Resource Efficiency (Renewable Energy + Water Reuse)
elif chat["round"] == 2 and not chat["completed"]:
    st.write(f"**Round 2/5: Hi {chat['data']['company_name']}! Let’s talk resources.**")
    st.write("(Enter percentages as numbers, e.g., 40 = 40%)")
    col1, col2 = st.columns(2)
    with col1:
        renewable_pct = st.number_input(
            "What % of your energy comes from renewable sources (solar/wind)?",
            min_value=0, max_value=100, key="round2_renewable"
        )
    with col2:
        water_pct = st.number_input(
            "What % of your water is reused/recycled (vs. total consumption)?",
            min_value=0, max_value=100, key="round2_water"
        )
    
    if st.button("Next →", key="round2_next"):
        chat["data"]["resource_efficiency"]["renewable_energy_pct"] = renewable_pct
        chat["data"]["resource_efficiency"]["water_reuse_pct"] = water_pct
        chat["round"] = 3
        st.rerun()

# Round 3: Sustainable Production (Recycled Materials + Waste Intensity)
elif chat["round"] == 3 and not chat["completed"]:
    st.write("**Round 3/5: Let’s dive into production.**")
    st.write("(Enter percentages as numbers, e.g., 30 = 30%)")
    col1, col2 = st.columns(2)
    with col1:
        recycled_pct = st.number_input(
            "What % of your production materials are recycled/upcycled?",
            min_value=0, max_value=100, key="round3_recycled"
        )
    with col2:
        waste_intensity = st.number_input(
            "How does your waste intensity compare to the industry average? (Enter %: 20 = 20% below average, 50 = 50% above)",
            min_value=0, max_value=200, key="round3_waste"
        )
    
    if st.button("Next →", key="round3_next"):
        chat["data"]["sustainable_production"]["recycled_material_pct"] = recycled_pct
        chat["data"]["sustainable_production"]["waste_intensity_pct"] = waste_intensity
        chat["round"] = 4
        st.rerun()

# Round 4: Circular Economy (Take-Back Programs + Sustainable Packaging)
elif chat["round"] == 4 and not chat["completed"]:
    st.write("**Round 4/5: Almost there! Let’s talk circularity.**")
    st.write("(Enter percentages as numbers, e.g., 25 = 25%)")
    col1, col2 = st.columns(2)
    with col1:
        takeback_pct = st.number_input(
            "What % of your product lines have a take-back/recycling program?",
            min_value=0, max_value=100, key="round4_takeback"
        )
    with col2:
        packaging_pct = st.number_input(
            "What % of your packaging is sustainable (recyclable/compostable)?",
            min_value=0, max_value=100, key="round4_packaging"
        )
    
    if st.button("Next →", key="round4_next"):
        chat["data"]["circular_economy"]["takeback_program_pct"] = takeback_pct
        chat["data"]["circular_economy"]["packaging_sustainable_pct"] = packaging_pct
        chat["round"] = 5
        st.rerun()

# Round 5: Confirmation & Score Calculation
elif chat["round"] == 5 and not chat["completed"]:
    st.write("**Round 5/5: Let’s confirm your data!**")
    
    # Show summary of collected data
    st.write("**Your Responses:**")
    st.write(f"- Company: {chat['data']['company_name']} | Industry: {chat['data']['industry']}")
    st.write(f"- Renewable energy: {chat['data']['resource_efficiency']['renewable_energy_pct']}%")
    st.write(f"- Water reuse: {chat['data']['resource_efficiency']['water_reuse_pct']}%")
    st.write(f"- Recycled materials: {chat['data']['sustainable_production']['recycled_material_pct']}%")
    st.write(f"- Waste intensity vs. industry: {chat['data']['sustainable_production']['waste_intensity_pct']}%")
    st.write(f"- Product take-back: {chat['data']['circular_economy']['takeback_program_pct']}%")
    st.write(f"- Sustainable packaging: {chat['data']['circular_economy']['packaging_sustainable_pct']}%")
    
    if st.button("Calculate My SDG 12 Score!", key="round5_calc"):
        # Calculate score and generate feedback
        calculate_score()
        generate_feedback()
        chat["completed"] = True
        st.rerun()

# Post-Completion: Show Results & Export Report
elif chat["completed"]:
    data = chat["data"]
    st.success("🎉 Great! Here’s your SDG 12 performance feedback:")
    
    # 1. Score Overview
    st.subheader(f"Your Total SDG 12 Score: {data['total_score']}/100")
    score_color = "#2E8B57" if data["total_score"] >= 70 else "#FFD700" if data["total_score"] >= 50 else "#DC143C"
    st.markdown(f"<h3 style='color:{score_color}'>{['Needs Improvement', 'Good', 'Excellent'][0 if data['total_score']<50 else 1 if data['total_score']<70 else 2]}</h3>", unsafe_allow_html=True)
    
    # 2. Dimension Breakdown
    st.subheader("📊 Dimension Breakdown")
    for dim_id, dim_data in data["dimension_scores"].items():
        st.write(f"- {dim_data['name']}: {dim_data['weighted_score']}/{dim_data['max_weighted']}")
    
    # 3. Recommendations
    st.subheader("💡 Actionable Recommendations")
    for i, rec in enumerate(data["recommendations"], 1):
        st.write(f"{i}. {rec}")
    
    # 4. Download Feedback Report
    st.subheader("📥 Download Your Full Feedback Report")
    st.download_button(
        label="Download TXT Report",
        data=data["feedback_excerpt"],
        file_name=f"{data['company_name']}_SDG12_Feedback.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    # Reset Chat
    if st.button("Evaluate Another Company", use_container_width=True):
        st.session_state["chat_state"] = {
            "round": 1,
            "data": {
                "company_name": "",
                "industry": "",
                "resource_efficiency": {"renewable_energy_pct": 0, "water_reuse_pct": 0},
                "sustainable_production": {"recycled_material_pct": 0, "waste_intensity_pct": 0},
                "circular_economy": {"takeback_program_pct": 0, "packaging_sustainable_pct": 0},
                "total_score": 0,
                "dimension_scores": {},
                "recommendations": [],
                "feedback_excerpt": ""
            },
            "completed": False
        }
        st.rerun()
