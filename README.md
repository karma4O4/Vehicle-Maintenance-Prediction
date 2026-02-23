# Vehicle Maintenance Prediction System

## Project Overview
This project aims to predict vehicle maintenance needs using historical usage, sensor data, and maintenance records. It is designed to help fleet managers optimize servicing schedules and reduce vehicle downtime.

## Milestone 1: ML-Based Vehicle Maintenance Prediction
The goal of this milestone is to build a classification model that predicts whether a vehicle requires immediate maintenance based on its current state and history.

### Team Contributions (Group of 4)

This project was a collaborative effort where specific modules were handled by the team, and general system architecture and integration were led by the project lead.

*   **Person 1 (Aman):** Data Engineer - Responsible for data preprocessing, cleaning, and initial feature engineering in `src/data_prep.py`.
*   **Person 2 (Ashish):** ML Scientist - Responsible for model selection, training, and evaluation in `src/train_model.py`.
*   **Person 3 (Abhijeet):** UI Developer - Responsible for the primary Streamlit interface and interactive components in `app.py`.
*   **Harsil (Project Lead):** System Architecture & Integration - Responsible for the overall system design, data validation logic, repository management, final documentation, and end-to-end integration of all modules.

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
