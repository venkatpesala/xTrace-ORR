import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import plotly.express as px

# -------------------------------
# Page configuration and styling
# -------------------------------

st.set_page_config(
    page_title="xTRACE - Explainable AI (xAI) Trafficking Risk Assessment & Case Evaluation", 
    page_icon="??",
    layout="wide"
)

def set_page_style(page):
    if page == 1:  # Blue theme
        primary_color = "#2196F3"
        secondary_color = "#90CAF9"
        text_color = "#0D47A1"
        background_color = "#E3F2FD"
    elif page == 2:  # Orange theme
        primary_color = "#FF7E33"
        secondary_color = "#FFB380"
        text_color = "#7F3F19"
        background_color = "#FFF2E6"
    elif page == 3:  # Purple theme
        primary_color = "#7E33FF"
        secondary_color = "#B380FF"
        text_color = "#3F197F"
        background_color = "#F2E6FF"
    elif page == 4:  # Yellow theme
        primary_color = "#FFC107"
        secondary_color = "#FFE082"
        text_color = "#7F6003"
        background_color = "#FFFDE7"
    
    greys = {
        "dark_grey": "#333333",
        "grey": "#666666",
        "medium_grey": "#999999",
        "light_grey": "#cccccc",
        "very_light_grey": "#e6e6e6",
        "almost_white": "#f7f7f7"
    }
    
    st.markdown(f"""
    <style>
    body {{
        background-color: {background_color} !important;
    }}
    .stApp {{
        background-color: {background_color} !important;
    }}
    .main .block-container {{
        padding-top: 2rem;
        background-color: {background_color} !important;
    }}
    h1, h2, h3, h4, h5 {{
        color: {text_color} !important;
    }}
    label, .stTextInput label, .stTextArea label, .stNumberInput label, 
    .stSelectbox label, .stDateInput label {{
        color: {text_color} !important;
        font-weight: bold !important;
        font-size: 1rem !important;
    }}
    .stTextArea, .stMarkdown, .stText, p, span, div {{
        color: {text_color} !important;
    }}
    .stButton > button {{
        background-color: {primary_color} !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.5rem 1rem !important;
        font-weight: bold !important;
    }}
    .stButton > button:hover {{
        background-color: {secondary_color} !important;
        color: {text_color} !important;
    }}
    .stProgress > div > div > div {{
        background-color: {primary_color} !important;
    }}
    .dataframe {{
        border: 2px solid {primary_color} !important;
    }}
    .dataframe th {{
        background-color: {primary_color} !important;
        color: white !important;
        font-weight: bold !important;
        padding: 8px !important;
    }}
    .dataframe td {{
        padding: 8px !important;
        border-bottom: 1px solid {secondary_color} !important;
        background-color: white !important;
        color: {text_color} !important;
    }}
    .risk-high {{
        background-color: #FF4B4B !important;
        color: white !important;
        padding: 0.5rem !important;
        border-radius: 4px !important;
        font-weight: bold !important;
        text-align: center !important;
    }}
    .risk-medium {{
        background-color: #FFC107 !important;
        color: black !important;
        padding: 0.5rem !important;
        border-radius: 4px !important;
        font-weight: bold !important;
        text-align: center !important;
    }}
    .risk-low {{
        background-color: #4CAF50 !important;
        color: white !important;
        padding: 0.5rem !important;
        border-radius: 4px !important;
        font-weight: bold !important;
        text-align: center !important;
    }}
    .page-header {{
        background-color: {primary_color} !important;
        color: white !important;
        padding: 10px !important;
        border-radius: 5px !important;
        text-align: center !important;
        margin-bottom: 20px !important;
        font-weight: bold !important;
    }}
    </style>
    """, unsafe_allow_html=True)

def load_logo():
    try:
        return Image.open("logo.png")
    except Exception:
        return None

# -------------------------------
# Data Processing and Prediction
# -------------------------------

def process_excel_file(file_path):
    df = pd.read_excel(file_path)
    
    orange_columns = ["ID", "first_name", "last_name", "dob", "email", "phone", 
                      "sponsor_id_hash", "fingerprint_hash", "ssn"]
    
    purple_columns = ["is_duplicate", "county", "high_trafficking", 
                      "Sponsor Registration", "FBI Fingerprint (Galton)", 
                      "Orange-IAM", "Purple-Vetting", "UAC Portal", "ICE", 
                      "CBP", "ATIMS", "DHS Payment", "Local Welfare Services"]
    
    available_orange = [col for col in orange_columns if col in df.columns]
    available_purple = [col for col in purple_columns if col in df.columns]
    
    orange_data = df[available_orange].copy() if available_orange else pd.DataFrame()
    purple_data = df[available_purple].copy() if available_purple else pd.DataFrame()
    
    return orange_data, purple_data

# Load a model if available
MODEL_PATH = "models/sar_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception:
    model_loaded = False

def predict_sponser_risk(record, model_loaded=None):
    try:
        from sentry_lite.risk_model import predict_risk
        base_score = predict_risk(record, model)
    except Exception:
        base_score = 30 

    if record.get("is_duplicate", False):
        base_score += 5
    if record.get("high_trafficking", False):
        base_score += 20
    if not record.get("Sponsor Registration", True):
        base_score += 10
    if not record.get("FBI Fingerprint (Galton)", True):
        base_score += 10
    if not record.get("Purple-Vetting", True):
        base_score += 10        
    if record.get("ICE", False):
        base_score += 15        
    if record.get("CBP", False):
        base_score += 10
    if not record.get("UAC Portal", True):
        base_score += 8
    if not record.get("Orange-IAM", True):
        base_score += 5
    if not record.get("ATIMS", True):
        base_score += 5
    
    return min(base_score, 100)

# -------------------------------
# Session State Initialization
# -------------------------------

if "page" not in st.session_state:
    st.session_state.page = 1

if "data_sources" not in st.session_state:
    st.session_state.data_sources = {
        'acf_data': False,
        'ice_data': False,
        'dhs_data': False,
        'uacs_data': False
    }

if "orange_data" not in st.session_state or "purple_data" not in st.session_state:
    st.session_state.orange_data, st.session_state.purple_data = process_excel_file("full_canonicalization_dataset_script_output.xlsx")

# -------------------------------
# Navigation: Sidebar & Buttons
# -------------------------------

# Sidebar navigation using a radio button; labels are mapped from page numbers.
page_labels = {
    1: "Step 1: Data Fetching",
    2: "Step 2: Sponsor Information",
    3: "Step 3: System Checks",
    4: "Step 4: Analysis Results"
}
selected_page = st.sidebar.radio(
    "Navigation", 
    options=[1, 2, 3, 4],
    index=st.session_state.page - 1,
    format_func=lambda x: page_labels[x]
)
st.session_state.page = selected_page  # update state if the user chooses via sidebar

def render_navigation_buttons(prev_page, next_page):
    """Renders the Previous and Next buttons in two columns.
       Updates st.session_state.page without requiring a manual refresh.
    """
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous"):
            if st.session_state.page > 1:
                st.session_state.page -= 1
    with col2:
        if st.button("Next"):
            if st.session_state.page < 4:
                st.session_state.page += 1  # This directly updates the session state on the first click
                st.rerun()  # This will force a rerun to apply the change immediately


# -------------------------------
# Dashboard Header
# -------------------------------

col1, col2 = st.columns([1, 5])
with col1:
    logo = load_logo()
    if logo:
        st.image(logo, width=200)
    else:
        st.write("??")
with col2:
    st.markdown(
        "<h1 style='font-size:70px;'>xTRACE - Explainable AI (xAI) Trafficking Risk Assessment & Case Evaluation</h1>", 
        unsafe_allow_html=True
    )

st.markdown("---")  # horizontal separator

# -------------------------------
# Page-wise Content Definitions
# -------------------------------

def data_fetching_page():
    st.markdown('<div class="page-header">DATA FETCHING</div>', unsafe_allow_html=True)
    st.write("Select the data sources you want to use for analysis:")
    
    with st.form("data_sources_form"):
        col1, col2 = st.columns(2)
        with col1:
            acf_data = st.checkbox("ACF Data", value=st.session_state.data_sources['acf_data'])
            ice_data = st.checkbox("ICE Data", value=st.session_state.data_sources['ice_data'])
        with col2:
            dhs_data = st.checkbox("DHS Data", value=st.session_state.data_sources['dhs_data'])
            uacs_data = st.checkbox("UACs Data", value=st.session_state.data_sources['uacs_data'])
        submit = st.form_submit_button("Fetch Data")
        if submit:
            st.session_state.data_sources = {
                'acf_data': acf_data,
                'ice_data': ice_data,
                'dhs_data': dhs_data,
                'uacs_data': uacs_data
            }
            st.success("Data fetched successfully!")
            st.session_state.page = 2  # move to next step

def orange_data_page():
    st.markdown('<div class="page-header">SPONSOR INFORMATION</div>', unsafe_allow_html=True)
    st.write("Review the sponsor data extracted from selected sources:")
    
    orange_data = st.session_state.orange_data
    st.dataframe(orange_data, use_container_width=True)
    
    st.subheader("Detailed Sponsor Information")
    selected_id = st.selectbox("Select a sponsor to view details:", orange_data['ID'].tolist())
    selected_row = orange_data[orange_data['ID'] == selected_id].iloc[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **ID:** {selected_row['ID']}  
        **Name:** {selected_row['first_name']} {selected_row['last_name']}  
        **DOB:** {selected_row['dob']}  
        **Email:** {selected_row['email']}  
        """)
    with col2:
        st.markdown(f"""
        **Phone:** {selected_row['phone']}  
        **SSN:** {selected_row['ssn']}  
        **ID Hash:** {selected_row['sponsor_id_hash']}  
        **Fingerprint Hash:** {selected_row['fingerprint_hash']}  
        """)
    
    render_navigation_buttons(prev_page=1, next_page=3)

def purple_data_page():
    st.markdown('<div class="page-header">SYSTEM CHECKS</div>', unsafe_allow_html=True)
    st.write("Review the system verification data and risk factors:")
    
    purple_data = st.session_state.purple_data
    orange_data = st.session_state.orange_data  # for duplicate lookup
    st.dataframe(purple_data, use_container_width=True)
    
    st.subheader("System Verification Status")
    sponsor_idx = st.selectbox(
        "Select sponsor index to view system checks:",
        range(len(purple_data)),
        format_func=lambda x: f"Sponsor {x+1}"
    )
    selected_row = purple_data.iloc[sponsor_idx]
    
    system_checks = [
        "Sponsor Registration", "FBI Fingerprint (Galton)", "Orange-IAM", 
        "Purple-Vetting", "UAC Portal", "ICE", "CBP", "ATIMS", 
        "DHS Payment", "Local Welfare Services"
    ]
    
    cols = st.columns(2)
    for i, check in enumerate(system_checks):
        col = cols[i % 2]
        status = "Yes" if selected_row[check] else "No"
        # For ICE and CBP, reverse the logic (True is bad)
        if check in ["ICE", "CBP"]:
            status = "No" if selected_row[check] else "Yes"
        status_class = "status-pass" if status == "Yes" else "status-fail"
        col.markdown(f"**{check}:** <span class='{status_class}'>{status}</span>", unsafe_allow_html=True)
    
    st.subheader("Risk Indicators")
    cols2 = st.columns(2)
    with cols2[0]:
        if selected_row["is_duplicate"]:
            st.markdown("<div class='risk-high'>DUPLICATE DETECTED</div>", unsafe_allow_html=True)
            dup_fields = ["first_name", "last_name", "dob", "email", "phone", "ssn", "sponsor_id_hash", "fingerprint_hash"]
            duplicate_mask = pd.Series([False] * len(orange_data))
            for field in dup_fields:
                duplicate_mask = duplicate_mask | (orange_data[field] == orange_data.iloc[sponsor_idx][field])
            duplicate_mask &= (orange_data['ID'] != orange_data.iloc[sponsor_idx]['ID'])
            duplicates = orange_data[duplicate_mask]
            if not duplicates.empty:
                st.markdown("**Duplicate Details:**")
                st.dataframe(duplicates, use_container_width=True)
            else:
                st.markdown("No duplicate details found.")
        else:
            st.markdown("<div class='risk-low'>NO DUPLICATES</div>", unsafe_allow_html=True)
    
    with cols2[1]:
        if selected_row["high_trafficking"]:
            st.markdown("<div class='risk-high'>HIGH TRAFFICKING</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='risk-low'>LOW TRAFFICKING</div>", unsafe_allow_html=True)
    
    render_navigation_buttons(prev_page=2, next_page=4)

def heatmap_generator(results_df):
    # Extract state and clean county name
    results_df['State'] = results_df['County'].apply(lambda x: x.split(',')[1].strip() if ',' in x else 'Unknown')

    # State name to abbreviation mapping
    state_mapping = {
        'Nevada': 'NV',
        'New Mexico': 'NM',
        'Texas': 'TX',
        'California': 'CA',
        'Florida': 'FL',
        'Georgia': 'GA',
        'Arizona': 'AZ'
    }
    # Apply the mapping to get state abbreviations
    results_df['State_Abbr'] = results_df['State'].map(state_mapping)

    # Calculate average risk score by state
    state_risk = results_df.groupby(['State', 'State_Abbr'])['Risk Score'].mean().reset_index()

    # Create choropleth map with full state names in hover text
    fig = px.choropleth(
        state_risk,
        locations='State_Abbr',  # Still use abbreviations for mapping
        locationmode='USA-states',
        color='Risk Score',
        scope="usa",
        color_continuous_scale="Reds",
        range_color=(state_risk['Risk Score'].min(), state_risk['Risk Score'].max()),
        labels={'Risk Score': 'Average Risk Score', 'State': 'State Name'},
        hover_name='State',  # This will show the full state name as the primary hover text
        hover_data={'State_Abbr': False, 'Risk Score': True}  # Hide State_Abbr, keep Risk Score
    )

    fig.update_layout(
        title_text='State Risk Score Heat Map',
        geo=dict(
            showcoastlines=True,
            projection_type='albers usa'
        ),
        height=800,
        width=900
    )
    return fig

def results_page():
    st.markdown('<div class="page-header">ANALYSIS RESULTS</div>', unsafe_allow_html=True)
    st.write("Risk assessment for sponsors:")
    
    orange_data = st.session_state.orange_data
    purple_data = st.session_state.purple_data
    results = []
    
    for i in range(len(orange_data)):
        sponsor = orange_data.iloc[i].to_dict()
        system_checks = purple_data.iloc[i].to_dict()
        risk_score = predict_sponser_risk(system_checks, model_loaded)
        if risk_score >= 70:
            risk_level = "HIGH RISK"
        elif risk_score > 40:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "LOW RISK"
        results.append({
            "ID": sponsor["ID"],
            "First Name": sponsor["first_name"],
            "Last Name": sponsor["last_name"],
            "County": system_checks.get("county", ""),
            "Is Duplicate": "Yes" if system_checks.get("is_duplicate", False) else "No",
            "High Trafficking": "Yes" if system_checks.get("high_trafficking", False) else "No",
            "Risk Score": risk_score,
            "Risk Level": risk_level
        })
    
    results_df = pd.DataFrame(results)
    
    def highlight_risk(val):
        # Check if val is numeric (Risk Score)
        if isinstance(val, (int, float)):  # Check if it's a number
            if val >= 70:
                return 'background-color: #FF4B4B; color: white; font-weight: bold'
            elif val >= 40:
                return 'background-color: #FFC107; color: black; font-weight: bold'
            else:
                return 'background-color: #4CAF50; color: white; font-weight: bold'
        # Check for Risk Level (string comparison)
        elif isinstance(val, str):
            if 'HIGH RISK' in val:
                return 'background-color: #FF4B4B; color: white; font-weight: bold'
            elif 'MEDIUM RISK' in val:
                return 'background-color: #FFC107; color: black; font-weight: bold'
            elif 'LOW RISK' in val:
                return 'background-color: #4CAF50; color: white; font-weight: bold'
        return ''

    
    styled_df = results_df.style.applymap(highlight_risk, subset=['Risk Level', 'Risk Score'])  # Apply to both columns
    
    st.markdown("### Risk Assessment Results Table")
    st.dataframe(styled_df, use_container_width=True, height=400, hide_index=True)
    
    st.subheader("Risk Summary")
    high_risk = len(results_df[results_df["Risk Score"] >= 70])
    medium_risk = len(results_df[(results_df["Risk Score"] >= 40) & (results_df["Risk Score"] < 70)])
    low_risk = len(results_df[results_df["Risk Score"] < 40])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("High Risk Cases", high_risk)
    with col2:
        st.metric("Medium Risk Cases", medium_risk)
    with col3:
        st.metric("Low Risk Cases", low_risk)
    
    st.subheader("Risk by County")
    fig = heatmap_generator(results_df)
    st.plotly_chart(fig)
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="risk_assessment_results.csv",
        mime="text/csv",
    )
    
    render_navigation_buttons(prev_page=3, next_page=1)  # 'Start Over' sends user to page 1


# -------------------------------
# Main
# -------------------------------

def main():
    set_page_style(st.session_state.page)
    progress_value = (st.session_state.page - 1) / 3
    st.progress(progress_value)
    st.subheader(f"Step {st.session_state.page} of 4")
    
    if st.session_state.page == 1:
        data_fetching_page()
    elif st.session_state.page == 2:
        orange_data_page()
    elif st.session_state.page == 3:
        purple_data_page()
    elif st.session_state.page == 4:
        results_page()

if __name__ == "__main__":
    main()
