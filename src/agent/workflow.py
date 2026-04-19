# src/agent/workflow.py
from typing import TypedDict, Annotated
import pandas as pd
from src.agent.rag_engine import get_retriever
from src.agent.prompts import generate_report

class AgentState(TypedDict):
    vehicle_row: dict
    ml_score: float
    importance_factors: list
    rag_context: str
    final_report: str

def run_maintenance_agent(df_row, risk_score, importance_factors):
    """
    Orchestrates the agentic workflow.
    df_row: A single row from the vehicle dataframe (dict)
    risk_score: The probability score from the ML model
    importance_factors: List of top feature importances
    """
    # 1. Initialize RAG and retrieve context
    retriever = get_retriever()
    context = retriever.get_maintenance_context(
        df_row.get('Vehicle_Model'), importance_factors
    )

    # 2. Package ML data for the reasoning engine
    ml_data = {
        **df_row,
        "Risk_Score": risk_score,
        "Feature_Importance": importance_factors,
    }

    # 3. Generate report via the LLM Reasoning Engine
    report = generate_report(ml_data=ml_data, rag_context=context)

    return report
