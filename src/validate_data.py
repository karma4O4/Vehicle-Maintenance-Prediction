import pandas as pd

def validate_csv(df):
    """
    Validates the input dataframe for required columns and data types.
    Returns (is_valid, message)
    """
    required_columns = {
        'Vehicle_Model': 'object',
        'Mileage': 'int64',
        'Maintenance_History': 'object',
        'Reported_Issues': 'int64',
        'Vehicle_Age': 'int64',
        'Fuel_Type': 'object',
        'Odometer_Reading': 'int64'
    }
    
    # Check for missing columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {', '.join(missing_cols)}"
    
    # Check for data types
    # Note: We convert to numeric where possible if types don't exactly match
    for col, expected_type in required_columns.items():
        if expected_type in ['int64', 'float64']:
            try:
                pd.to_numeric(df[col])
            except Exception:
                return False, f"Column '{col}' must be numeric."
        elif expected_type == 'object':
            if not df[col].dtype == 'object' and not df[col].dtype.name == 'category':
                # Convert to string to see if it's usable
                df[col] = df[col].astype(str)

    return True, "Data validation successful!"

if __name__ == "__main__":
    # Test with sample data
    df = pd.read_csv('data/vehicle_maintenance_data.csv')
    valid, msg = validate_csv(df)
    print(msg)
