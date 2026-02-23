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

## 4. Model Performance Analysis (Initial)
Based on the synthetic dataset:
- **Model**: Random Forest Classifier
- **Accuracy**: ~90% (Expected on balanced synthetic data)
- **Key Features**: Mileage, Maintenance_History, and Vehicle_Age are expected to be the most influential factors.
