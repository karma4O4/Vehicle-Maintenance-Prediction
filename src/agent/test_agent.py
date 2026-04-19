# test_agent.py
"""
Test suite for the LLM Reasoning & Reporting Engine.
Covers: normal data, missing mileage (noisy data), and all risk levels.
"""
from src.agent.prompts import generate_report, _classify_risk, _estimate_mileage
from src.agent.workflow import run_maintenance_agent

# ──────────────────────────────────────────────────────────────────────
# Test 1: Full pipeline via workflow (normal data — high risk)
# ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 1: Full Pipeline — High-Risk Truck (via workflow)")
print("=" * 60)

mock_vehicle = {
    'Vehicle_Model': 'Truck',
    'Mileage': 150000,
    'Maintenance_History': 'Poor',
    'Vehicle_Age': 10,
    'Reported_Issues': 7,
    'Fuel_Type': 'Diesel',
    'Odometer_Reading': 152000,
}
risk_score = 0.95
top_factors = ['Maintenance_History', 'Mileage', 'Vehicle_Age']

report = run_maintenance_agent(mock_vehicle, risk_score, top_factors)
print(report)

# ──────────────────────────────────────────────────────────────────────
# Test 2: generate_report() directly — Missing Mileage (noisy data)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 2: Noisy Data — Missing Mileage (estimated from age)")
print("=" * 60)

noisy_vehicle = {
    'Vehicle_Model': 'Van',
    'Mileage': None,             # <-- Missing!
    'Maintenance_History': 'Average',
    'Vehicle_Age': 6,
    'Reported_Issues': 3,
    'Fuel_Type': 'Petrol',
    'Odometer_Reading': 0,
    'Risk_Score': 0.72,
    'Feature_Importance': ['Vehicle_Age', 'Reported_Issues', 'Fuel_Type'],
}

report_noisy = generate_report(ml_data=noisy_vehicle, rag_context="")
print(report_noisy)

# ──────────────────────────────────────────────────────────────────────
# Test 3: generate_report() — Low-Risk Sedan
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3: Low-Risk Sedan")
print("=" * 60)

low_risk_vehicle = {
    'Vehicle_Model': 'Sedan',
    'Mileage': 20000,
    'Maintenance_History': 'Good',
    'Vehicle_Age': 2,
    'Reported_Issues': 0,
    'Fuel_Type': 'Electric',
    'Odometer_Reading': 20500,
    'Risk_Score': 0.12,
    'Feature_Importance': ['Mileage', 'Vehicle_Age'],
}

report_low = generate_report(ml_data=low_risk_vehicle, rag_context="Electric vehicles require less frequent servicing but battery health monitoring is recommended.")
print(report_low)

# ──────────────────────────────────────────────────────────────────────
# Test 4: Unit tests for helper functions
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 4: Unit Tests — Helper Functions")
print("=" * 60)

# Risk classification
assert _classify_risk(0.95) == "CRITICAL", "Failed: 0.95 should be CRITICAL"
assert _classify_risk(0.75) == "HIGH", "Failed: 0.75 should be HIGH"
assert _classify_risk(0.50) == "MODERATE", "Failed: 0.50 should be MODERATE"
assert _classify_risk(0.20) == "LOW", "Failed: 0.20 should be LOW"
print("✅ _classify_risk() — All assertions passed.")

# Mileage estimation
est_truck, was_est = _estimate_mileage("Truck", 5)
assert was_est is True
assert est_truck == 125_000, f"Expected 125000 for Truck*5yr, got {est_truck}"

est_sedan, _ = _estimate_mileage("Sedan", 3)
assert est_sedan == 36_000, f"Expected 36000 for Sedan*3yr, got {est_sedan}"

est_unknown, _ = _estimate_mileage("Hovercraft", 4)
assert est_unknown == 60_000, f"Expected 60000 for unknown*4yr, got {est_unknown}"
print("✅ _estimate_mileage() — All assertions passed.")

# Verify report structure
required_sections = [
    "Vehicle Health Summary",
    "Maintenance Risk",
    "Recommended Actions",
    "Safety Disclaimer",
]
for section in required_sections:
    assert section in report, f"Missing section in report: {section}"
    assert section in report_noisy, f"Missing section in noisy report: {section}"
    assert section in report_low, f"Missing section in low-risk report: {section}"
print("✅ All reports contain required sections.")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✅")
print("=" * 60)
