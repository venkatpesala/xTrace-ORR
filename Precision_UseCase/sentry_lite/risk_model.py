# # sentry_lite/risk_model.py

# import joblib
# import pandas as pd
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mean_squared_error

# def train_model(df):
#     # Drop unnecessary columns
#     X = df.drop(columns=["UID", "SAR", "HTR", "is_high_risk_sar", "is_high_risk_htr"])

#     # Convert categorical columns to numerical using label encoding
#     label_columns = [
#         'Age', 'Gender', 'Country_of_Origin', 'Family_Ties_Status', 'Financial_Status', 
#         'Criminal_History', 'Known_Trafficking_Route', 'Past_Human_Trafficking_Case',
#         'Multiple_ICE_Investigations', 'Trafficking_Network_Affiliation', 'Illegal_Border_Crossing_Record',
#         'Duplicate_Records', 'Trafficking_Hotspot_Residence', 'Financial_Transactions_Flagged',
#         'Multiple_Unrelated_UACs', 'Background_Check_Status', 'Identity_Document_Verification',
#         'Unusual_Sponsor_UAC_Relationship'
#     ]

#     label_encoder = LabelEncoder()
#     for col in label_columns:
#         X[col] = label_encoder.fit_transform(X[col])

#     y = df["SAR"]

#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Initialize the XGBoost regressor model
#     model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

#     # Train the model
#     model.fit(X_train, y_train)

#     # Save the model to disk using joblib
#     joblib.dump(model, "models/sar_model.pkl")

#     # Evaluate the model using Mean Squared Error
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f"Model Training Complete. MSE: {mse}")

#     return model

# def predict_risk(record, model):
#     # Convert input record to DataFrame
#     record_df = pd.DataFrame([record])

#     # Apply label encoding to categorical features in the record
#     label_columns = [
#         'Age', 'Gender', 'Country_of_Origin', 'Family_Ties_Status', 'Financial_Status', 
#         'Criminal_History', 'Known_Trafficking_Route', 'Past_Human_Trafficking_Case',
#         'Multiple_ICE_Investigations', 'Trafficking_Network_Affiliation', 'Illegal_Border_Crossing_Record',
#         'Duplicate_Records', 'Trafficking_Hotspot_Residence', 'Financial_Transactions_Flagged',
#         'Multiple_Unrelated_UACs', 'Background_Check_Status', 'Identity_Document_Verification',
#         'Unusual_Sponsor_UAC_Relationship'
#     ]
    
#     label_encoder = LabelEncoder()
#     for col in label_columns:
#         if col in record_df.columns:
#             record_df[col] = label_encoder.fit_transform(record_df[col])

#     # Ensure the input record matches the model's expected input format
#     model_columns = model.feature_names_in_
#     for col in model_columns:
#         if col not in record_df.columns:
#             record_df[col] = 0

#     record_df = record_df[model_columns]

#     # Predict the SAR score
#     return model.predict(record_df)[0]



# sentry_lite/risk_model.py

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

def create_interaction_features(df):
    """
    Create interaction features that might capture higher-risk behavior patterns.
    For example, flag if both Past_Denials and Criminal_History are positive.
    """
    df['High_Risk_Indicators'] = (df['Past_Denials'] > 0) & (df['Criminal_History'] > 0)
    return df

def train_model(df):
    # Create new features
    df = create_interaction_features(df)

    # Drop unnecessary columns
    X = df.drop(columns=["UID", "SAR", "HTR", "is_high_risk_sar", "is_high_risk_htr"])

    # Ensure feature names are strings
    X.columns = [str(col) for col in X.columns]

    # Define the columns that are categorical and need label encoding
    label_columns = [
        'Age', 'Gender', 'Country_of_Origin', 'Family_Ties_Status', 'Financial_Status', 
        'Criminal_History', 'Known_Trafficking_Route', 'Past_Human_Trafficking_Case',
        'Multiple_ICE_Investigations', 'Trafficking_Network_Affiliation', 'Illegal_Border_Crossing_Record',
        'Duplicate_Records', 'Trafficking_Hotspot_Residence', 'Financial_Transactions_Flagged',
        'Multiple_Unrelated_UACs', 'Background_Check_Status', 'Identity_Document_Verification',
        'Unusual_Sponsor_UAC_Relationship', 'High_Risk_Indicators'
    ]
    
    label_encoder = LabelEncoder()
    for col in label_columns:
        X[col] = label_encoder.fit_transform(X[col])

    # Scale features (especially important when mixing numeric and encoded features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Target variable
    y = df["SAR"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize the XGBoost regressor model
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model from the grid search
    model = grid_search.best_estimator_

    # Save the model to disk using joblib
    joblib.dump(model, "models/sar_model.pkl")

    # Evaluate the model using Mean Squared Error
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Tuned Model MSE: {mse}")

    return model

def preprocess_user_input(user_input):
    """
    Preprocess user inputs into a format compatible with the trained model.
    Handles categorical conversion and boolean mapping.
    """
    # Map Family Ties Status: 'Verified' -> 1, others -> 0
    family_ties_map = {"Verified": 1, "Unverified": 0, "Unknown": 0}
    user_input["Family_Ties_Status"] = family_ties_map.get(user_input.get("Family_Ties_Status", "Unknown"), 0)
    
    # Gender: 'M' -> 1, 'F' -> 0
    user_input["Gender"] = 1 if user_input.get("Gender", "F") == "M" else 0
    
    # Country: Map countries to numeric codes
    country_map = {"Honduras": 0, "Guatemala": 1, "El Salvador": 2, "Mexico": 3}
    user_input["Country_of_Origin"] = country_map.get(user_input.get("Country_of_Origin", "Guatemala"), -1)
    
    # Financial Status: Map to numeric
    financial_status_map = {"Low": 0, "Medium": 1, "High": 2}
    user_input["Financial_Status"] = financial_status_map.get(user_input.get("Financial_Status", "Low"), 0)

    # Convert boolean inputs to integers (0 or 1)
    user_input["Criminal_History"] = 1 if user_input.get("Criminal_History", False) else 0
    user_input["Prior_Trafficking_History"] = 1 if user_input.get("Prior_Trafficking_History", False) else 0
    user_input["Network_Affiliation"] = 1 if user_input.get("Network_Affiliation", False) else 0
    user_input["Known_Trafficking_Route"] = 1 if user_input.get("Known_Trafficking_Route", False) else 0

    # Ensure numeric features are integers
    user_input["Past_Sponsorships"] = int(user_input.get("Past_Sponsorships", 0))
    user_input["Past_Denials"] = int(user_input.get("Past_Denials", 0))
    
    return user_input

def predict_risk(record, model):
    processed_input = preprocess_user_input(record)

    # Convert the processed input into a DataFrame with string-based columns
    record_df = pd.DataFrame([processed_input])
    record_df.columns = [str(col) for col in record_df.columns]

    # Attempt to retrieve the model's expected feature names.
    try:
        model_columns = model.get_booster().feature_names
        if model_columns is None:
            raise AttributeError("Feature names not available in the model")
    except AttributeError:
        # Fallback: Define the full list of features used during training
        model_columns = [
            'Age', 'Gender', 'Country_of_Origin', 'Family_Ties_Status', 'Prior_Trafficking_History',
            'Past_Sponsorships', 'Past_Denials', 'Financial_Status', 'Criminal_History', 'Known_Trafficking_Route',
            'Past_Human_Trafficking_Case', 'Multiple_ICE_Investigations', 'Trafficking_Network_Affiliation',
            'Illegal_Border_Crossing_Record', 'Duplicate_Records', 'Trafficking_Hotspot_Residence',
            'Financial_Transactions_Flagged', 'Multiple_Unrelated_UACs', 'Background_Check_Status',
            'Identity_Document_Verification', 'Unusual_Sponsor_UAC_Relationship', 'High_Risk_Indicators'
        ]

    # Ensure all the expected columns are in the DataFrame, adding any missing columns with default value 0.
    for col in model_columns:
        if col not in record_df.columns:
            record_df[col] = 0

    # Reorder the columns in the same order as expected by the model.
    record_df = record_df[model_columns]

    # Predict the SAR score using the model.
    prediction = model.predict(record_df)[0]
    return prediction
