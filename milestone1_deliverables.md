# Milestone 1 Deliverables

## 1. Problem Understanding and Use-Case
Fleet maintenance is often reactive, leading to unexpected breakdowns and high operational costs. By leveraging historical data (mileage, age, service history), we can transition to a predictive maintenance model. This reduces downtime and extends vehicle lifespan.

## 2. Input-Output Specification
- **Inputs**:
    - `Vehicle_Model`: Type/Class of vehicle.
    - `Mileage`: Total distance covered.
    - `Maintenance_History`: Condition based on previous records (Good, Average, Poor).
    - `Reported_Issues`: Number of issues reported by drivers.
    - `Vehicle_Age`: Years since manufacture.
    - `Odometer_Reading`: Current reading.
- **Outputs**:
    - `Need_Maintenance`: Binary prediction (True/False or 0/1).
    - `Maintenance_Probability`: Risk score (0.0 to 1.0).
    - `Contribution Factors`: Key features driving the prediction.

## 3. System Architecture (ML Pipeline)
```text
[ Data Source (CSV) ] --> [ Preprocessing & Feature Engineering ] 
                                      |
                                      V
[ Random Forest Model ] <-- [ Model Training & Evaluation ]
          |
          V
[ Streamlit UI Dashboard ] --> [ Maintenance Prediction & Insights ]
```

## 4. Usage Instructions
1. **Setup Environment**: Ensure Python 3.9+ is installed. Install requirements using `pip install -r requirements.txt`.
2. **Data Generation**: Run `python src/generate_data.py` to create a standard dataset for testing.
3. **Model Training**: Run `python src/train_model.py` to train the Random Forest model and save it to the `models/` directory.
4. **Launch Application**: Run `streamlit run app.py` to open the web dashboard.
5. **Prediction**: Upload a CSV file through the UI. Use the sample in `data/` for the best experience.

## 5. Model Performance Analysis
Based on our current implementation:
- **Model**: Random Forest Classifier
- **Accuracy**: 93%
- **Analysis**: The high accuracy (93%) is attributed to the Random Forest's ability to capture non-linear relationships between vehicle age, mileage, and maintenance history. The feature importance analysis identifies **Mileage** as the primary predictor (approx. 46%), which aligns with real-world fleet dynamics where distance covered is the leading indicator of wear and tear. The model effectively handles categorical data (like `Maintenance_History`) after label encoding, allowing it to differentiate between vehicles with "Good" vs "Poor" service records. Initial overfitting is mitigated by the ensemble nature of the trees, though performance on real-world noisy sensor data (planned for Milestone 2) will require further regularization.
