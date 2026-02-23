# Vehicle Maintenance Prediction System

## Project Overview
This project aims to predict vehicle maintenance needs using historical usage, sensor data, and maintenance records. It is designed to help fleet managers optimize servicing schedules and reduce vehicle downtime.

## Milestone 1: ML-Based Vehicle Maintenance Prediction
The goal of this milestone is to build a classification model that predicts whether a vehicle requires immediate maintenance based on its current state and history.

### Team Roles & Responsibilities
This project is divided among 4 members:

1. **Person 1: Data Engineer (Data Preprocessing & Feature Engineering)**
   - Responsible for `src/data_prep.py`.
   - Handles missing data, categorical encoding, and feature scaling.
   
2. **Person 2: ML Scientist (Model Development & Evaluation)**
   - Responsible for `src/train_model.py`.
   - Trains classification models (Random Forest/Logistic Regression) and evaluates metrics.
   
3. **Person 3: UI Developer (Streamlit Application)**
   - Responsible for `app.py`.
   - Creates the user interface for data upload and prediction visualization.
   
4. **Person 4: System Architect (Integration & Documentation)**
   - Responsible for system architecture, integration, and deployment readiness.
   - Maintains `README.md` and system requirements.

## System Architecture (ML Pipeline)
1. **Input**: CSV file containing vehicle mileage, age, history, etc.
2. **Preprocessing**: Data cleaning, encoding categorical variables (`Vehicle_Model`, `Fuel_Type`), and normalization.
3. **Training**: Supervised learning using a Random Forest Classifier.
4. **Output**: Maintenance Risk Prediction (High/Low) and feature importance analysis.
5. **UI**: Streamlit-based dashboard for interaction.

## How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the app: `streamlit run app.py`.
