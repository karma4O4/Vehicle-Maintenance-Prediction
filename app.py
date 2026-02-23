import streamlit as st
import pandas as pd
import joblib
import os
from src.data_prep import preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Vehicle Maintenance Predictor", layout="wide")

st.title("🚗 Vehicle Maintenance Prediction System")
st.markdown("""
Predict vehicle maintenance needs using Machine Learning. Upload your fleet data or input details manually.
""")

# Dashboard Metrics Placeholder
metric_col1, metric_col2 = st.columns(2)
total_vehicles = st.session_state.get('total_vehicles', 0)
high_risk_vehicles = st.session_state.get('high_risk_vehicles', 0)

metric_col1.metric("Total Vehicles Scanned", total_vehicles)
metric_col2.metric("High Risk Vehicles Found", high_risk_vehicles, delta_color="inverse")

# Sidebar for model info
st.sidebar.header("Vehicle Details (Manual Entry)")

with st.sidebar.form("manual_input_form"):
    v_model = st.selectbox("Vehicle Model", ["Sedan", "SUV", "Truck", "Van", "Coupe"])
    mileage = st.number_input("Mileage", min_value=0, value=50000)
    v_age = st.number_input("Vehicle Age (Years)", min_value=0, value=5)
    m_history = st.selectbox("Maintenance History", ["Good", "Average", "Poor"])
    reported_issues = st.number_input("Reported Issues", min_value=0, value=0)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
    odometer = st.number_input("Odometer Reading", min_value=0, value=mileage + 1000)
    
    submit_button = st.form_submit_button("Predict Maintenance Risk")

st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info("This system uses a Random Forest model trained on vehicle mileage, age, and service history.")

def perform_prediction(df_input):
    if os.path.exists('models/vehicle_model.pkl'):
        model = joblib.load('models/vehicle_model.pkl')
        X = preprocess_data(df_input, is_training=False)
        
        # Ensure only necessary columns are used for prediction
        # Get the feature names the model was trained on
        # If it's a Random Forest or similar from sklearn, it might have feature_names_in_
        try:
            expected_features = model.feature_names_in_
            X_input = X[expected_features]
        except AttributeError:
            if 'Need_Maintenance' in X.columns:
                X_input = X.drop(columns=['Need_Maintenance'])
            else:
                X_input = X
        
        predictions = model.predict(X_input)
        probabilities = model.predict_proba(X_input)[:, 1]
        
        df_result = df_input.copy()
        df_result['Maintenance_Risk'] = predictions
        df_result['Risk_Score'] = probabilities
        return df_result, model, X_input
    else:
        st.error("Model file not found. Please train the model first.")
        return None, None, None

# File upload
uploaded_file = st.file_uploader("Upload Vehicle Data (CSV)", type=["csv"])

def show_maintenance_tips(risk_score):
    st.subheader("🛠️ Maintenance Recommendations")
    if risk_score > 0.8:
        st.error("**Risk Level: CRITICAL**")
        st.markdown("- **Immediate Brake and Oil Check required.**")
        st.markdown("- Inspect tire pressure and tread depth.")
        st.markdown("- Check engine coolant levels.")
    elif risk_score > 0.5:
        st.warning("**Risk Level: MODERATE**")
        st.markdown("- **Schedule a routine service soon.**")
        st.markdown("- Check brake pads for wear.")
        st.markdown("- Rotate tires if not done recently.")
    else:
        st.success("**Risk Level: LOW**")
        st.markdown("- **Vehicle is in good condition.**")
        st.markdown("- Continue regular maintenance schedule.")
        st.markdown("- Monitor for any unusual noises.")

display_df = None

if submit_button:
    manual_df = pd.DataFrame({
        'Vehicle_Model': [v_model],
        'Mileage': [mileage],
        'Maintenance_History': [m_history],
        'Reported_Issues': [reported_issues],
        'Vehicle_Age': [v_age],
        'Fuel_Type': [fuel_type],
        'Odometer_Reading': [odometer]
    })
    
    result_df, model, X_input = perform_prediction(manual_df)
    if result_df is not None:
        # Update session state
        st.session_state['display_df'] = result_df
        st.session_state['model'] = model
        st.session_state['X_input'] = X_input
        st.session_state['total_vehicles'] = 1
        st.session_state['high_risk_vehicles'] = 1 if result_df['Maintenance_Risk'].iloc[0] == 1 else 0
        st.rerun()

elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if st.session_state.get('last_file') != uploaded_file.name:
        result_df, model, X_input = perform_prediction(df)
        if result_df is not None:
            # Update session state
            st.session_state['display_df'] = result_df
            st.session_state['model'] = model
            st.session_state['X_input'] = X_input
            st.session_state['total_vehicles'] = len(df)
            st.session_state['high_risk_vehicles'] = int(result_df['Maintenance_Risk'].sum())
            st.session_state['last_file'] = uploaded_file.name
            st.rerun()
    else:
        st.error("Model file not found. Please train the model first.")

# Retrieve from session state if available
display_df = st.session_state.get('display_df')
model = st.session_state.get('model')
X_input = st.session_state.get('X_input')

if display_df is not None:
    st.subheader("Maintenance Predictions")
    
    # Style result
    def highlight_risk(val):
        color = 'red' if val == 1 else 'green'
        return f'color: {color}'
        
    st.write(display_df[['Vehicle_Model', 'Mileage', 'Vehicle_Age', 'Maintenance_Risk', 'Risk_Score']].head(20))
    
    # Show Maintenance Tips for single prediction
    if len(display_df) == 1:
        show_maintenance_tips(display_df['Risk_Score'].iloc[0])
    else:
        # Show tips for highest risk vehicle in the batch
        high_risk_vehicle = display_df.sort_values('Risk_Score', ascending=False).iloc[0]
        st.info(f"Summary for highest risk vehicle: {high_risk_vehicle['Vehicle_Model']}")
        show_maintenance_tips(high_risk_vehicle['Risk_Score'])

    # Summary Visuals
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Risk Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Maintenance_Risk', data=display_df, ax=ax, palette='viridis')
        # Handle cases with only one risk level present
        labels = []
        if 0 in display_df['Maintenance_Risk'].values: labels.append('Low Risk')
        if 1 in display_df['Maintenance_Risk'].values: labels.append('High Risk')
        # Setting xticks properly
        # ax.set_xticklabels(labels) # This can fail if countplot doesn't have both
        st.pyplot(fig)
        
    with col2:
        st.write("### Factors Importance")
        if model and hasattr(model, 'feature_importances_') and X_input is not None:
            importances = model.feature_importances_
            feat_importance = pd.DataFrame({'Feature': X_input.columns, 'Importance': importances})
            feat_importance = feat_importance.sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=feat_importance, ax=ax, palette='magma')
            st.pyplot(fig)
        else:
            st.info("Feature importance not available (model not loaded or not a tree-based model).")
            
    # Download results
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, "vehicle_predictions.csv", "text/csv")

elif uploaded_file is None and not submit_button:
    st.info("Waiting for CSV file upload or manual entry...")
    
    # Show example format
    st.write("### Expected CSV Format")
    example_df = pd.DataFrame({
        'Vehicle_Model': ['Sedan', 'SUV'],
        'Mileage': [50000, 120000],
        'Maintenance_History': ['Good', 'Poor'],
        'Reported_Issues': [1, 5],
        'Vehicle_Age': [3, 8],
        'Fuel_Type': ['Petrol', 'Diesel'],
        'Odometer_Reading': [51000, 122000]
    })
    st.table(example_df)
