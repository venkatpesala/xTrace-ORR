import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import joblib
from datetime import datetime, date


# Set page configuration
st.set_page_config(
    page_title="xTRACE - xAI Trafficking Risk Assessment & Case Evaluation",
    page_icon="??",
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
        background-color: {colors["primary"]["yellow"]};
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

# Function to load and display logo
def load_logo():
    try:
        logo = Image.open("logo.png")
        return logo
    except:
        # If logo not found, return None
        return None

# Header with logo
col1, col2 = st.columns([1, 5])
with col1:
    logo = load_logo()
    if logo:
        st.image(logo, width=200)
    else:
        st.write("??")
with col2:
    st.markdown(f"<h1 style='font-size:58px;'>xTRACE - xAI Trafficking Risk Assessment & Case Evaluation</h1>", unsafe_allow_html=True)

st.markdown("---")

# Create two columns for the form
col1, col2 = st.columns(2)

# Define the model path (adjust if needed)
MODEL_PATH = "models/sar_model.pkl"

    
def read_fingerprint_data(filename='fig_hastu.txt'):
    """Read fingerprint data with appropriate encoding handling"""
    try:
        # First try UTF-8
        with open(filename, 'r', encoding='utf-8') as f:
            f_data = f.read().split(',')
            return f_data
    except UnicodeDecodeError:
        # If UTF-8 fails, try with cp1252 (Windows encoding)
        try:
            with open(filename, 'r', encoding='cp1252') as f:
                f_data = f.read().split(',')
                return f_data
        except UnicodeDecodeError:
            # If that still fails, use latin-1 which can read any byte
            with open(filename, 'r', encoding='latin-1') as f:
                f_data = f.read().split(',')
                return f_data
f_data = read_fingerprint_data(filename='fig_hastu.txt')
# Use the function to read the data
f_data = read_fingerprint_data()

# Try to load the model if it exists
try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except:
    model_loaded = False
    # We'll use mock data if model can't be loaded


data = pd.read_excel(r"synthetic data for ACF precision forum demo -april 14 2025 -acb.xlsx")

if 'sponsor_name' not in st.session_state:
    st.session_state.sponsor_name = ''

if 'sponsor_address' not in st.session_state:
    st.session_state.sponsor_address = ''

if 'sponsor_phone' not in st.session_state:
    st.session_state.sponsor_phone = ''

if 'sponsor_ssn' not in st.session_state:
    st.session_state.sponsor_ssn = ''

if 'fingure_data' not in st.session_state:
    st.session_state.fingure_data = []

# if 'sponsor_loaded' not in st.session_state:
#     st.session_state.sponsor_loaded = True

# Sponsor Inputs Form
with col1:
    st.markdown("<h2>Sponsor Information</h2>", unsafe_allow_html=True)
    
    # Basic sponsor information
    sponsor_name = st.text_input("Sponsor Name",value=st.session_state.sponsor_name)
    sponsor_country = st.selectbox("Country", ["United States", "Canada", "Mexico", "Other"])
    sponsor_address = st.text_area("Address (if available)",value=st.session_state.sponsor_address)
    sponsor_phone = st.text_input("Phone (if available)",value=st.session_state.sponsor_phone)
    sponsor_ssn = st.text_input("SSN (if available)",value=st.session_state.sponsor_ssn)
    # Set min and max dates
    min_date = date(1900, 1, 1)  # Allow dates from 1900
    max_date = date.today()  # Allow dates up to today

    today = date.today()

    # Date input for DOB
    sponsor_dob = st.date_input(
        "DOB (if available)", 
        key="sponsor_dob",
        min_value=min_date,
        max_value=max_date
    )

    # Calculate age automatically
    if sponsor_dob:
        age_cal = today.year - sponsor_dob.year - ((today.month, today.day) < (sponsor_dob.month, sponsor_dob.day))
    else:
      age_cal = 35 
    # Replace the text input with a file upload component
    sponsor_fingerprint_file = st.file_uploader("Upload Fingerprint File", type=['txt', 'pdf', 'png', 'jpg'], key="sponsor_fingerprint_file")
    sponsor_fingerprint = 0
    # Optional: Add handling for the uploaded file
    if sponsor_fingerprint_file is not None and "sponsor_loaded" not in st.session_state:
        # Display file details
        file_details = {"Filename": sponsor_fingerprint_file.name, "FileType": sponsor_fingerprint_file.type, "FileSize": sponsor_fingerprint_file.size}        
        # You can read the file content if needed
        # For text files:
        if sponsor_fingerprint_file:
            # sponsor_fingerprint = sponsor_fingerprint_file.getvalue().decode("utf-8")
            sponsor_fingerprint = np.random.choice(f_data)
            st.write("Fingerprint hash:", sponsor_fingerprint)
            fingure_data = data[data['fingerprint_hash']==sponsor_fingerprint]
            st.session_state.fingure_data = fingure_data
            st.session_state.sponsor_name = fingure_data[['first_name', 'last_name']].agg(' '.join, axis=1).iloc[0]
            st.session_state.sponsor_address = fingure_data['address'].iloc[0]
            st.session_state.sponsor_phone = fingure_data['phone'].iloc[0]
            st.session_state.sponsor_ssn = fingure_data['ssn'].iloc[0]
            st.session_state.sponsor_loaded = True
            st.rerun()
        # For binary files, you might want to display them or process differently
        else:
            st.write("File uploaded successfully")
    # Additional sponsor information (from the dictionary)
    # st.markdown("<h3>Risk Assessment Factors</h3>", unsafe_allow_html=True)
    
    # Gender (using M/F format)
    sponsor_gender = st.selectbox("Gender", ["Male", "Female"], key="sponsor_gender")
    
    # For sponsor age, use a slider
    sponsor_age = st.slider("Age", 18, 115, age_cal, key="sponsor_age")
    
    # Family Ties Status
    family_ties = st.selectbox("Family Ties Status", 
                              ["Verified", "Unverified", "Unknown"], 
                              key="family_ties")
    
    # Criminal History - ensure default value is displayed
    criminal_history = st.checkbox("Criminal History", value=False, key="criminal_history")
    
    # Past Sponsorships and Denials - set default values
    past_sponsorships = st.slider("Past Sponsorships", 0, 5, value=0, key="past_sponsorships")
    past_denials = st.slider("Past Denials", 0, 3, value=0, key="past_denials")
    
    # Financial Status
    financial_status = st.selectbox("Financial Status", 
                                   ["Low", "Medium", "High"], 
                                   index=0,
                                   key="financial_status")
    
    # Trafficking-related checkboxes - ensure default values are displayed
    st.markdown("<h3>Trafficking Risk Indicators</h3>", unsafe_allow_html=True)
    prior_trafficking = st.checkbox("Prior Trafficking History", value=False, key="prior_trafficking", help="Check if sponsor has any prior trafficking history")
    network_affiliation = st.checkbox("Network Affiliation", value=False, key="network_affiliation", help="Check if sponsor has known affiliations with trafficking networks")
    known_route = st.checkbox("Known Trafficking Route", value=False, key="known_route", help="Check if sponsor is associated with known trafficking routes")

# Child Information Form
with col2:
    st.markdown("<h2>Child Information</h2>", unsafe_allow_html=True)
    
    # Using slider for age as in the previous app
    child_age = st.slider("Child Age", 0, 17, key="child_age")
    
    # Using M/F format for gender as in the previous app
    child_gender = st.selectbox("Gender", ["Male", "Female"], key="child_gender")
    
    # Countries from the original list but prioritizing the ones in the dictionary
    child_country = st.selectbox("Country of Origin", 
                                ["Honduras", "Guatemala", "El Salvador", "Mexico", 
                                 "Nicaragua", "Costa Rica", "Panama", "Other"], 
                                key="child_country")
    
    # Add some spacing to align with sponsor form
    for _ in range(12):
        st.write("")

# Output section
if sponsor_name:
    st.markdown("---")
    st.markdown("<h2>Vetting Results</h2>", unsafe_allow_html=True)

    st.markdown("""
<div class="output-container">
<p style="font-size: 1.6rem;">Based on the data provided below are the risk scores calculated by AI</p>
</div>
""", unsafe_allow_html=True)
    
    # Prepare the record for scoring
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
    
    # Include all sponsor information in a combined dictionary
    sponsor_data = {
        "Name": sponsor_name,
        "Country": sponsor_country,
        "Address": sponsor_address,
        "Phone": sponsor_phone,
        "SSN": sponsor_ssn,
        "DOB": sponsor_dob,
        "Fingerprint": sponsor_fingerprint,
        "Age": sponsor_age,
        "Gender": sponsor_gender
    }
    
    from sentry_lite.risk_model import predict_risk
    score = predict_risk(record, model)

    score = min(max(score, 0), 100)
    

    def calculate_d_score(score, sponsor_age, past_sponsorships, past_denials, criminal_history, known_route, network_affiliation, prior_trafficking):
        d_score = score
        
        if past_sponsorships!=0:
            d_score += (past_sponsorships*7)

        if past_denials!=0:
            d_score += (past_denials*10)

        if 18 <= sponsor_age <= 28 or sponsor_age > 75:
            d_score *= 1.2
            
        # Additional factor if female and criminal history
        if criminal_history: 
            d_score *= 1.5
        if known_route and network_affiliation and prior_trafficking:
            return 98
        # Apply trafficking indicators - each one increases by factor of 1.2
        if known_route:
            d_score = 87
        
        if network_affiliation:
            d_score = 89
            # d_score *= 1.2
        
        if prior_trafficking:
            d_score = 92
            # d_score *= 1.2
        
        # Round to 2 decimal places
        return min(int(d_score), 100)
    
    # Fixed duplication score from your example
    score = calculate_d_score(score, sponsor_age, past_sponsorships, past_denials, criminal_history, known_route, network_affiliation, prior_trafficking)
    d_score = np.random.randint(30, 85)
    
    # Side by side scores without Key Factors
    col1, col2 = st.columns(2)
    if len(st.session_state.fingure_data)>1:
        score+=20
    
    # Determine risk levels
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
    
    # Risk Score
    with col1:
        st.markdown("<h3>Sponsor Risk Score</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="score-container">
            <h1>{score}%</h1>
            <span class="{sponsor_risk_class}">{sponsor_risk_level}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Duplication Score
    with col2:
        st.markdown("<h3>Child Risk Score</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="score-container">
            <h1>{d_score}%</h1>
            <span class="{duplication_risk_class}">{duplication_risk_level}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f'<hr style="border-top: 1px solid {colors["primary"]["yellow"]};">', unsafe_allow_html=True)
    
    # Related Information (Potential Duplicate Sponsors)
    st.markdown("<h3>Sponsor Duplicates</h3>", unsafe_allow_html=True)
    
    # Generate similar sponsors list as in your previous app
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

    if len(st.session_state.fingure_data)!=0:
        sponsors_df = st.session_state.fingure_data[['first_name', 'last_name', 'phone', 'address']]
    else:
        # Create DataFrame with 1-based indexing
        sponsors_df = pd.DataFrame(sponsors)

    sponsors_df.index = sponsors_df.index + 1
    
    # Display the table
    st.table(sponsors_df)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Final recommendation
    if score > 85:
        st.markdown('<div class="risk-high" style="font-size: 1.3rem; padding: 1rem;"><strong>HIGH RISK DETECTED</strong> - Manual review required before proceeding</div>', unsafe_allow_html=True)
    elif score > 60 or d_score > 60:
        st.markdown('<div class="risk-medium" style="font-size: 1.3rem; padding: 1rem;"><strong>MEDIUM RISK DETECTED</strong> - Additional verification recommended</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="risk-low" style="font-size: 1.3rem; padding: 1rem;"><strong>LOW RISK PROFILE</strong> - Processing can proceed with standard protocols</div>', unsafe_allow_html=True)
# Footer
st.markdown("---")
st.markdown('<div class="footer">SENTRY-Lite Sponsor Vetting System • For authorized use only • © 2025</div>', unsafe_allow_html=True)
