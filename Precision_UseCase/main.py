import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import joblib
from datetime import datetime, date

# ----------------------------
# Page and Styling Configuration
# ----------------------------
# Set page configuration
st.set_page_config(
    page_title="xTRACE - Explainable AI (xAI) Trafficking Risk Assessment & Case Evaluation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Color palette from the company guidelines
colors = {
    "primary": {
        "yellow": "#ffcb05",
        "black": "#000000",
        "blue": "#0000ff", 
        "navy": "#031091",
        "green": "#648e37",
        "light_green": "#7fa35c"
    },
    "greys": {
        "dark_grey": "#333333",
        "grey": "#666666",
        "medium_grey": "#999999",
        "light_grey": "#cccccc",
        "very_light_grey": "#e6e6e6",
        "almost_white": "#f7f7f7"
    }
}

# Custom CSS to ensure visibility with dark theme
st.markdown(f"""
    <style>
    /* Set header colors */
    body {{
        background-color: {colors["greys"]["almost_white"]};
    }}
    .stApp {{
        background-color: {colors["greys"]["almost_white"]};
    }}
    .main .block-container {{
        padding-top: 2rem;
        background-color: {colors["greys"]["almost_white"]};
    }}
    h1, h2, h3 {{
        color: {colors["primary"]["navy"]} !important;
    }}
    
    /* Make labels more visible */
    label, .stTextInput label, .stTextArea label, .stNumberInput label, .stSelectbox label, .stDateInput label {{
        color: {colors["primary"]["navy"]} !important;
        font-weight: bold !important;
        font-size: 1rem !important;
    }}
    

    
    /* Risk indicators */
    .risk-high {{
        color: white;
        background-color: #FF4B4B;
        padding: 0.3rem 0.6rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }}
    .risk-medium {{
        color: black;
        background-color: {colors["primary"]["yellow"]};
        padding: 0.3rem 0.6rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }}
    .risk-low {{
        color: white;
        background-color: {colors["primary"]["green"]};
        padding: 0.3rem 0.6rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }}
    
    /* Output section styling */
    .output-container {{
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid {colors["primary"]["yellow"]};
        margin: 1rem 0;
    }}
    
    /* Make section dividers yellow */
    hr {{
        border-top: 2px solid {colors["primary"]["yellow"]};
    }}
    
    /* Make dataframe headers more visible */
    .dataframe thead th {{
        background-color: {colors["primary"]["navy"]};
        color: white !important;
    }}
    
    /* Styling for the app title */
    .app-title {{
        color: {colors["primary"]["navy"]};
        background-color: {colors["greys"]["almost_white"]};
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 20px;
    }}
    
    /* Footer styling */
    .footer {{
        color: {colors["greys"]["light_grey"]};
        text-align: center;
        padding: 10px;
        font-size: 0.8rem;
    }}

    /* Side by side score containers */
    .score-container {{
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
        background-color: white;
    }}
    /* Force text visibility in alerts with !important */
    div[data-baseweb="notification"] div {{
        color: black !important;
    }}

    /* Target specifically the alert content */
    div[data-baseweb="notification"] [data-testid="stMarkdownContainer"] p {{
        color: black !important;
        font-weight: bold !important;
    }}


    /* Target success, info, warning and error boxes directly */
    div[data-testid="stCaptionContainer"] {{
        color: black !important;
    }}


    /* Style input boxes with white background and navy text */
    input, textarea, [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] input,
    div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea, 
    div[data-testid="stDateInput"] input, div[data-baseweb="base-input"] input {{
        background-color: white !important;
        color: {colors["primary"]["navy"]} !important;
        border: 1px solid {colors["greys"]["light_grey"]} !important;
    }}

    /* Style select boxes and other form controls */
    [data-baseweb="select"] [data-baseweb="popover"] div,
    div[data-testid="stSelectbox"] div[aria-expanded="false"],
    div[data-baseweb="select"] div {{
        background-color: white !important;
        color: {colors["primary"]["navy"]} !important;
    }}

    /* Style date picker and other specialized inputs */
    .stDateInput > div[data-baseweb="input"] {{
        background-color: white !important;
    }}
    .stDateInput > div[data-baseweb="input"] input {{
        color: {colors["primary"]["navy"]} !important;
    }}
    
    /* Checkbox styling */
    div[data-testid="stCheckbox"] label {{
        color: {colors["primary"]["navy"]} !important;
        font-weight: bold !important;
    }}
    
    /* Style the checkbox itself */
    div[data-testid="stCheckbox"] div[role="checkbox"] {{
        background-color: white !important;
        border: 2px solid {colors["primary"]["navy"]} !important;
    }}
    
    /* Style the checkbox when checked */
    div[data-testid="stCheckbox"] div[data-checked="true"] {{
        background-color: {colors["primary"]["navy"]} !important;
    }}
    
    /* Style the checkbox icon when checked */
    div[data-testid="stCheckbox"] svg {{
        color: white !important;
        fill: white !important;
    }}
    
    /* Style the slider track */
    div[data-testid="stSlider"] div[role="slider"] div:first-child {{
        background-color: {colors["greys"]["medium_grey"]};
    }}
    
    /* Style the slider knob */
    div[data-testid="stSlider"] div[role="slider"] div:nth-child(2) {{
        background-color: {colors["primary"]["yellow"]} !important;
        border: 2px solid {colors["primary"]["navy"]} !important;
    }}
    
    /* Style the slider active track */
    div[data-testid="stSlider"] div[role="slider"] div:nth-child(3) {{
        background-color: {colors["primary"]["navy"]} !important;
    }}


    /* Force text visibility in alerts with !important */
    div[data-baseweb="notification"] div {{
        color: navy !important;
    }}

    /* Target specifically the alert content */
    div[data-baseweb="notification"] [data-testid="stMarkdownContainer"] p {{
        color: navy !important;
        font-weight: bold !important;
    }}

    /* Style dataframe cells */
    div[data-testid="stDataFrame"] div[data-testid="stVerticalBlock"] div {{
        color: navy !important;
    }}

    /* Target success, info, warning and error boxes directly */
    div[data-testid="stCaptionContainer"] {{
        color: navy !important;
    }}

    /* Additional catchall for any text in the app */
    .stTextArea, .stMarkdown, .stText, p, span, div {{
        color: navy;
    }}


    /* Button styling */
    .stButton>button {{
        background-color: {colors["greys"]["almost_white"]};
        color: white;
        font-weight: bold;
        border: 2px solid {colors["primary"]["navy"]};
    }}
    .stButton>button:hover {{
        background-color: {colors["primary"]["blue"]};
        border: 2px solid {colors["primary"]["navy"]};
    }}

    


    div[data-baseweb="popover"] {{
  background-color: white !important;
}}
   div[data-baseweb="popover"] ul li,
div[data-baseweb="menu"] {{
  background-color: white !important;
  color: navy !important;
}}

div[data-baseweb="popover"] ul li:hover,
div[data-baseweb="menu"] div:hover {{
  background-color: light_grey !important;
}}

[data-baseweb="menu"] [role="option"] {{
  background-color: white !important;
  color: navy !important;
}}

[data-baseweb="select"] * {{
  color: navy !important;
}}



/* Target the top navigation/header bar */
    header[data-testid="stHeader"] {{
        background-color: {colors["greys"]["almost_white"]};
        color: white;
    }}
    
    /* Style the header buttons and icons */
    header[data-testid="stHeader"] button, 
    header[data-testid="stHeader"] svg {{
        color: white !important;
    }}
    
    /* Style the top-right menu in the header */
    header[data-testid="stHeader"] [data-testid="stToolbar"] {{
        background-color: {colors["greys"]["almost_white"]};
        color: white;
    }}

    /* Style the hamburger menu button */
    button[data-testid="baseButton-headerNoPadding"] svg {{
        fill: white !important;
    }}
    
    /* Style the dropdown menus from the header */
    header button[aria-expanded="true"] + div {{
        background-color: {colors["greys"]["almost_white"]};
        color: white;
    }}
    
    /* If you have a sidebar, match its header color */
    [data-testid="stSidebar"] [data-testid="stHeader"] {{
        background-color: {colors["greys"]["almost_white"]};
    }}

    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Utility Functions and Model Loading
# ----------------------------
def load_logo():
    """Attempt to load the logo; if not found, return None."""
    try:
        return Image.open("logo.png")
    except Exception:
        return None

def calculate_d_score(score, sponsor_age, past_sponsorships, past_denials, 
                      criminal_history, known_route, network_affiliation, prior_trafficking):
    """Calculate the duplication risk score based on provided factors."""
    d_score = score
    if past_sponsorships != 0:
        d_score += (past_sponsorships * 5)
    if past_denials != 0:
        d_score = 87
    if 18 <= sponsor_age <= 28 or sponsor_age > 75:
        d_score *= 1.2
    if criminal_history: 
        d_score *= 1.5
    if known_route and network_affiliation and prior_trafficking:
        return 98
    if known_route:
        d_score = 87
    if network_affiliation:
        d_score = 89
    if prior_trafficking:
        d_score = 92
    return min(int(d_score), 100)

# Load supporting files and model
MODEL_PATH = "models/sar_model.pkl"
with open('fig_hast.txt', 'r') as f:
    f_data = f.read().split(',')

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception:
    model_loaded = False

data = pd.read_excel(r"synthetic data for ACF precision forum demo -april 14 2025 -acb.xlsx")

states = [
    "California",
    "Texas",
    "Florida",
    "New York",
    "Illinois",
    "Pennsylvania",
    "Ohio",
    "Georgia",
    "North Carolina",
    "Michigan"
]

cities = [
    "New York City",
    "Los Angeles",
    "Chicago",
    "Houston",
    "Phoenix",
    "Philadelphia",
    "San Antonio",
    "San Diego",
    "Dallas",
    "San Jose"
]

streets = [
    "Main Street",
    "Broadway",
    "Elm Street",
    "Maple Avenue",
    "Oak Street",
    "Pine Street",
    "Cedar Avenue",
    "5th Avenue",
    "Sunset Boulevard",
    "Wall Street"
]

zip_codes = [
    "10001",  # New York, NY
    "90001",  # Los Angeles, CA
    "60601",  # Chicago, IL
    "77001",  # Houston, TX
    "85001",  # Phoenix, AZ
    "19101",  # Philadelphia, PA
    "78201",  # San Antonio, TX
    "92101",  # San Diego, CA
    "75201",  # Dallas, TX
    "95101"   # San Jose, CA
]

# ----------------------------
# Session State Initialization
# ----------------------------
if 'sponsor_name' not in st.session_state:
    st.session_state.sponsor_name = 'john paul'

if 'sponsor_staddress' not in st.session_state:
    st.session_state.sponsor_staddress = 'Broadway'

if 'sponsor_city' not in st.session_state:
    st.session_state.sponsor_city = 'Houston'

if 'sponsor_states' not in st.session_state:
    st.session_state.sponsor_states = 'Florida'

if 'sponsor_zip' not in st.session_state:
    st.session_state.sponsor_zip = '92101'

if 'sponsor_phone' not in st.session_state:
    st.session_state.sponsor_phone = '651515919111'

if 'sponsor_email' not in st.session_state:
    st.session_state.sponsor_email = 'johnpaul@hotmail.com'

if 'sponsor_suite' not in st.session_state:
    st.session_state.sponsor_suite = ''

if 'sponsor_ssn' not in st.session_state:
    st.session_state.sponsor_ssn = '51511981818'

if 'calculated_dob' not in st.session_state:
    st.session_state.calculated_dob = 'today'

# Initialize fingure_data as an empty DataFrame so that .empty attribute can be used
if 'fingure_data' not in st.session_state:
    st.session_state.fingure_data = pd.DataFrame()

# ----------------------------
# Header Section
# ----------------------------
with st.container():
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        logo = load_logo()
        if logo:
            st.image(logo, width=200)
        else:
            st.write("Logo Not Found")
    with col_title:
        st.markdown("<h1 class='app-title'>xTRACE - Explainable AI (xAI) Trafficking Risk Assessment & Case Evaluation</h1>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------
# Sponsor Information Section
# ----------------------------
st.markdown("<hr><h2>Sponsor Information</h2><hr>", unsafe_allow_html=True)
with st.container():
    # Row 1: Basic Information
    # row1 = st.columns(2)
    # with row1[0]:
    sponsor_name = st.text_input("*Full Name", value=st.session_state.sponsor_name)
    # with row1[1]:
        
    
    # Row 2: Contact Details
    row2 = st.columns(2)
    with row2[0]:
        sponsor_staddress = st.text_input("*Street address", value=st.session_state.sponsor_staddress)
    with row2[1]:
        sponsor_suite = st.text_input("Apt/Suite/Other", value=st.session_state.sponsor_suite)
    
    row5 = st.columns(3)
    with row5[0]:
        sponsor_city = st.text_input("*City", value=st.session_state.sponsor_city)
    with row5[1]:
        sponsor_states = st.selectbox("*states", states)
    with row5[2]:
        sponsor_zip = st.text_input("*Zip Code", value=st.session_state.sponsor_zip)

    row6 = st.columns(2)
    with row6[0]:
        sponsor_phone = st.text_input("*Phone", value=st.session_state.sponsor_phone)
    with row6[1]:
        sponsor_email = st.text_input("*Email", value=st.session_state.sponsor_email)

    # Row 3: Identifier and Date of Birth
    row3 = st.columns(2)
    with row3[0]:
        sponsor_ssn = st.text_input("SSN (if available)", value=st.session_state.sponsor_ssn)
    with row3[1]:
        min_date = date(1900, 1, 1)
        max_date = date.today()
        sponsor_age = 0
        today = date.today()
        if sponsor_age!=0:
            # Calculate approximate DOB based on age
            st.session_state.calculated_dob = date(today.year - sponsor_age, today.month, today.day)
        
        sponsor_dob = st.date_input("DOB (YYYY/MM/DD)", value=st.session_state.calculated_dob, key="sponsor_dob", min_value=min_date, max_value=max_date)
    
    today = date.today()
    if sponsor_dob:
        age_cal = today.year - sponsor_dob.year - ((today.month, today.day) < (sponsor_dob.month, sponsor_dob.day))
    else:
        age_cal = 35
    
    # Row 4: File Upload for Fingerprint
    row4 = st.container()
    sponsor_fingerprint_file = st.file_uploader("Upload Fingerprint File", type=['txt', 'pdf', 'png', 'jpg'], key="sponsor_fingerprint_file")
    sponsor_fingerprint = 0
    if sponsor_fingerprint_file is not None and "sponsor_loaded" not in st.session_state:
        # Process the file (placeholder logic)
        sponsor_fingerprint = np.random.choice(f_data)
        st.write("Fingerprint hash:", sponsor_fingerprint)
        fingure_data = data[data['fingerprint_hash'] == sponsor_fingerprint]
        st.session_state.fingure_data = fingure_data
        st.session_state.sponsor_name = fingure_data[['first_name', 'last_name']].agg(' '.join, axis=1).iloc[0]
        st.session_state.sponsor_staddress = np.random.choice(streets)
        st.session_state.sponsor_city = np.random.choice(cities)
        st.session_state.sponsor_states = sponsor_states
        st.session_state.sponsor_zip = np.random.choice(zip_codes)
        st.session_state.sponsor_phone = fingure_data['phone'].iloc[0]
        st.session_state.sponsor_email = fingure_data['email'].iloc[0]
        st.session_state.sponsor_ssn = fingure_data['ssn'].iloc[0]
        st.session_state.sponsor_loaded = True
        st.success("Fingerprint processed! Please refresh the page for updated values.")
        st.rerun()
    
    # Row 5: Additional Sponsor Details
    row5 = st.columns(2)
    with row5[0]:
        sponsor_gender = st.selectbox("Gender", ["Male", "Female"], key="sponsor_gender")
        sponsor_age = st.slider("Age", 18, 115, age_cal, key="sponsor_age")
    with row5[1]:
        family_ties = st.selectbox("Family Ties Status", ["Verified", "Unverified", "Unknown"], key="family_ties")
        criminal_history = st.checkbox("Criminal History", value=False, key="criminal_history")
    
    # Row 6: Risk Factors
    row6 = st.columns(2)
    with row6[0]:
        past_sponsorships = st.slider("Past Sponsorships", 0, 5, value=0, key="past_sponsorships")
    with row6[1]:
        past_denials = st.slider("Past Denials", 0, 3, value=0, key="past_denials")
    
    row7 = st.columns(2)
    with row7[0]:
        financial_status = st.selectbox("Financial Status", ["Low", "Medium", "High"], index=0, key="financial_status")
    with row7[1]:
        st.markdown("### Trafficking Risk Indicators")
        prior_trafficking = st.checkbox("Prior Trafficking History", value=False, key="prior_trafficking",
                                        help="Check if sponsor has any prior trafficking history")
        network_affiliation = st.checkbox("Network Affiliation", value=False, key="network_affiliation",
                                        help="Check if sponsor has known affiliations with trafficking networks")
        known_route = st.checkbox("Known Trafficking Route", value=False, key="known_route",
                                        help="Check if sponsor is associated with known trafficking routes")

# ----------------------------
# Child Information Section
# ----------------------------
st.markdown("<hr><h2>Child Information</h2><hr>", unsafe_allow_html=True)
with st.container():
    child_cols = st.columns(3)
    with child_cols[0]:
        child_age = st.slider("Child Age", 0, 17, key="child_age")
    with child_cols[1]:
        child_gender = st.selectbox("Gender", ["Male", "Female"], key="child_gender")
    with child_cols[2]:
        child_country = st.selectbox("Country of Origin",
                                      ["Honduras", "Guatemala", "El Salvador", "Mexico", 
                                       "Nicaragua", "Costa Rica", "Panama", "Other"],
                                      key="child_country")

st.markdown("---")

# ----------------------------
# Output Section: Risk Assessment and Recommendations
# ----------------------------
if sponsor_name and sponsor_email and sponsor_phone and sponsor_zip and sponsor_states and sponsor_city and sponsor_staddress:
    st.markdown("<hr><h2>Vetting Results</h2><hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="output-container">
        <p style="font-size: 1.6rem;">Based on the data provided, below are the risk scores calculated by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    record = {
        "Age": child_age,
        "Gender": child_gender,
        "Country_of_Origin": child_country,
        "Family_Ties_Status": family_ties,
        "Criminal_History": int(criminal_history),
        "Past_Sponsorships": past_sponsorships,
        "Past_Denials": past_denials,
        "Financial_Status": financial_status,
        "Prior_Trafficking_History": int(prior_trafficking),
        "Trafficking_Network_Affiliation": int(network_affiliation),
        "Known_Trafficking_Route": int(known_route)
    }
    
    sponsor_data = {
        "Name": sponsor_name,
        "Country": 'usa',
        "Address": sponsor_staddress,
        "Phone": sponsor_phone,
        "SSN": sponsor_ssn,
        "DOB": sponsor_dob,
        "Fingerprint": sponsor_fingerprint,
        "Age": sponsor_age,
        "Gender": sponsor_gender
    }
    
    # Calculate risk using the imported risk model
    from sentry_lite.risk_model import predict_risk
    score = predict_risk(record, model)
    score = min(max(score, 0), 100)
    
    # Adjust score with additional risk factors
    score = calculate_d_score(score, sponsor_age, past_sponsorships, past_denials,
                              criminal_history, known_route, network_affiliation, prior_trafficking)
    d_score = np.random.randint(30, 85)
    if len(st.session_state.fingure_data) > 1:
        score += 20
    score = min(score,100)
    # Determine risk levels for sponsor and duplication (child) scores
    if score > 85:
        sponsor_risk_class = "risk-high"
        sponsor_risk_level = "HIGH RISK"
    elif score > 60:
        sponsor_risk_class = "risk-medium"
        sponsor_risk_level = "MEDIUM RISK"
    else:
        sponsor_risk_class = "risk-low"
        sponsor_risk_level = "LOW RISK"
    
    if d_score > 85:
        duplication_risk_class = "risk-high"
        duplication_risk_level = "HIGH RISK"
    elif d_score > 60:
        duplication_risk_class = "risk-medium"
        duplication_risk_level = "MEDIUM RISK"
    else:
        duplication_risk_class = "risk-low"
        duplication_risk_level = "LOW RISK"
    
    # Display the scores side by side
    score_cols = st.columns(2)
    with score_cols[0]:
        st.markdown("<h3>Sponsor Risk Score</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="score-container">
            <h1>{score}%</h1>
            <span class="{sponsor_risk_class}">{sponsor_risk_level}</span>
        </div>
        """, unsafe_allow_html=True)
    with score_cols[1]:
        st.markdown("<h3>Child Risk Score</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="score-container">
            <h1>{d_score}%</h1>
            <span class="{duplication_risk_class}">{duplication_risk_level}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f'<hr style="border-top: 1px solid {colors["primary"]["yellow"]};">', unsafe_allow_html=True)
    
    # Display potential duplicate sponsor information
    st.markdown("<hr><h3>Sponsor Duplicates</h3><hr>", unsafe_allow_html=True)
    sponsors = [
        {
            "sponsor_name": f"John {sponsor_name}",
            "phone": "555-123-4567",
            "address": "123 Main Street, Honduras, CA 94321"
        },
        {
            "sponsor_name": f"{sponsor_name} Rodriguez",
            "phone": "555-987-6543",
            "address": "456 Oak Avenue, Guatemala, NY 10001"
        },
        {
            "sponsor_name": f"{sponsor_name} Chen",
            "phone": "555-246-8135",
            "address": "789 Maple Boulevard, Mexico, TX 75001"
        }
    ]
    if not st.session_state.fingure_data.empty:
        sponsors_df = st.session_state.fingure_data[['first_name', 'last_name', 'phone', 'email']]
    else:
        sponsors_df = pd.DataFrame(sponsors)
    sponsors_df.index = sponsors_df.index + 1
    st.table(sponsors_df)
    
    # Final recommendation based on risk score thresholds
    if score > 85:
        st.markdown(
            '<div class="risk-high" style="font-size: 1.3rem; padding: 1rem;"><strong>HIGH RISK DETECTED</strong> - Manual review required before proceeding</div>',
            unsafe_allow_html=True)
    elif score > 60 or d_score > 60:
        st.markdown(
            '<div class="risk-medium" style="font-size: 1.3rem; padding: 1rem;"><strong>MEDIUM RISK DETECTED</strong> - Additional verification recommended</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="risk-low" style="font-size: 1.3rem; padding: 1rem;"><strong>LOW RISK PROFILE</strong> - Processing can proceed with standard protocols</div>',
            unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
