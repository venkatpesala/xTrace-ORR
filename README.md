# xTrace-ORR
 Use Case: Xtrace for U.S. Customs and Border Protection (CBP) Title: Explainable AI for Trafficking Risk Assessment and Cross-Border Case Evaluation


 SENTRY-Lite Sponsor Vetting SENTRY-Lite is a streamlined risk assessment tool designed to aid in sponsor vetting for child safety. The tool uses an XGBoost regression model to predict a risk score (SAR) based on multiple inputs—including demographics, historical data, and additional risk indicators. The project includes functions for model training, feature engineering, hyperparameter tuning, and a user-friendly Streamlit interface for risk scoring.

Table of Contents Overview

Project Structure

Installation

Usage

Training the Model

Running the Streamlit App

Feature Engineering and Model Improvements

Contributing

License

Overview The SENTRY-Lite tool provides the following functionalities:

Model Training: Reads a dataset, performs feature engineering (including interaction features such as High_Risk_Indicators), and trains an XGBoost regression model using hyperparameter tuning via GridSearchCV.

Risk Prediction: Uses a preprocessing function to convert user inputs into the expected feature format and predicts the SAR (Sponsor Approval Risk) score. Missing features are automatically added with default values to ensure the input shape matches the training configuration.

User Interface: A Streamlit-based interface allows users to enter sponsor and child details and see real-time risk scoring.

Project Structure

. ├── sentry_lite/ │ ├── init.py │ ├── risk_model.py # Contains training, preprocessing, and prediction functions. │ └── main.py # Streamlit application for risk scoring. ├── models/ # Directory where trained models are saved. │ └── sar_model.pkl # Trained XGBoost model file. ├── data/ # (Optional) Directory to store your training datasets. ├── requirements.txt # List of required Python packages. └── README.md # This file.

Installation Clone the repository:

bash Copy git clone https://github.com/your-username/your-repo-name.git cd your-repo-name Create and activate a virtual environment (recommended):

bash Copy python3 -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate Install the dependencies:

bash Copy pip install -r requirements.txt Example contents of requirements.txt:

nginx Copy pandas scikit-learn xgboost joblib streamlit Usage Training the Model To train the model with your dataset, import and call the train_model function from sentry_lite/risk_model.py. For example:

python Copy import pandas as pd from sentry_lite.risk_model import train_model

Load your dataset (e.g., from an Excel or CSV file)
df = pd.read_excel("data/your_dataset.xlsx")

Train the model and save it to models/sar_model.pkl
model = train_model(df) Make sure your dataset includes all the necessary columns (like UID, SAR, Past_Denials, etc.) as expected by the model script. This training function also performs hyperparameter tuning and feature scaling.

Running the Streamlit App To run the Streamlit application, execute the following command from the root directory:

bash Copy streamlit run sentry_lite/main.py The app provides a user-friendly interface where you can input:

Sponsor Name

Child Age (slider)

Gender (select box)

Country of Origin (select box)

Family Ties Status ("Verified", "Unverified", "Unknown")

Sponsor Criminal History (checkbox)

Past Sponsorships (slider)

Past Denials (slider)

Financial Status ("Low", "Medium", "High")

Prior Trafficking History (checkbox)

Network Affiliation (checkbox)

Known Trafficking Route (checkbox)

Based on your inputs, the app will predict and display the SAR risk score and an associated risk category (e.g., HIGH RISK, MEDIUM RISK, LOW RISK).

Feature Engineering and Model Improvements Interaction Features: The training script creates a High_Risk_Indicators feature based on the presence of both Past_Denials and Criminal_History.

Hyperparameter Tuning: GridSearchCV is used to fine-tune the XGBoost model parameters, optimizing for a lower Mean Squared Error (MSE).

Consistent Feature Handling: The preprocessing functions and fallback mechanisms ensure that the input for predictions always matches the model's expected feature shape.

If you find any performance issues or wish to add more features, the code is modular and easy to extend.

Contributing Contributions are welcome! Please feel free to open issues or pull requests if you have any suggestions or bug fixes. For major changes, please open an issue first to discuss what you would like to change.

About
No description, website, or topics provided.
Resources
 Readme
 Activity
Stars
 0 stars
Watchers
 1 watching
Forks
 0 forks
Releases
No releases published
Create a new release
Packages
No packages published
Publish your first package
Languages
Python
100.0%
Suggested workflows
Based on your tech stack
Publish Python Package logo
Publish Python Package
Publish a Python Package to PyPI on release.
Pylint logo
Pylint
Lint a Python application with pylint.
Python Package using Anaconda logo
Python Package using Anaconda
Create and test a Python package on multiple Python versions using Anaconda for package management.
More workflows
Footer
