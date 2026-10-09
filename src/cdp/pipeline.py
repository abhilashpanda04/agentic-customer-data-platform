"""
End-to-End CDP Refresh Pipeline
1. Ingests raw data or generates synthetic multi-channel touchpoints.
2. Runs Identity Resolution Graph to unify customer identities into persistent Golden IDs (UCID).
3. Seeds DuckDB relational store (profiles, identity_map, facts, events, consents).
4. Computes RFM Segmentation, CLTV predictions, and Lead Conversion scores.
5. Populates the `customer_features` table ready for analytics and Agent querying.
"""
import pandas as pd
from datetime import datetime
from src.ingestion.synthetic_generator import generate_cdp_dataset
from src.cdp.identity_resolution import IdentityResolutionEngine
from src.cdp.database import CDPDatabase
from src.analytics.rfm import RFMAnalyzer
from src.analytics.cltv import CLTVPredictor
from src.analytics.lead_scoring import LeadScoringEngine

def run_pipeline(num_customers: int = 1200):
    print("=== Starting End-to-End CDP Pipeline ===")
    
    # 1. Ingestion
    dataset = generate_cdp_dataset(num_customers=num_customers)
    
    # 2. Identity Resolution
    print("\n--- Running Graph Identity Resolution ---")
    resolver = IdentityResolutionEngine(confidence_threshold=0.75)
    resolver.build_identity_graph(dataset["identities"])
    resolved_idm = resolver.resolve_identities()
    print(f"Total resolved mapping edges: {len(resolved_idm)}")
    print(f"Unified Customer Profiles created: {resolved_idm['unified_customer_id'].nunique()}")

    # 3. DuckDB Seeding
    print("\n--- Seeding DuckDB Relational Warehouse ---")
    db = CDPDatabase()
    db.load_unified_data(dataset, resolved_idm)

    # 4. Feature Engineering & Analytics
    print("\n--- Computing RFM Segments ---")
    df_tx = db.query("SELECT * FROM transaction_fact;")
    df_profiles = db.query("SELECT * FROM customer_profile;")
    df_events = db.query("SELECT * FROM event_fact;")

    rfm_analyzer = RFMAnalyzer()
    rfm_df = rfm_analyzer.compute_rfm(df_tx, df_profiles)

    print("\n--- Training Predictive CLTV Model ---")
    cltv_engine = CLTVPredictor(observation_window_days=150)
    cltv_feats = cltv_engine.extract_features(df_tx, df_profiles, df_events)
    cltv_df = cltv_engine.train_and_evaluate(cltv_feats)
    print(f"CLTV Metrics: {cltv_engine.metrics}")

    print("\n--- Training Lead Scoring Propensity Model ---")
    lead_engine = LeadScoringEngine()
    lead_feats = lead_engine.extract_features(df_profiles, df_events, df_tx)
    lead_df = lead_engine.train_and_score(lead_feats)
    print(f"Lead Scoring Metrics: {lead_engine.metrics}")
    # Combine into customer_features table
    print("\n--- Consolidating Customer 360 Feature Store ---")
    features_master = rfm_df.merge(
        cltv_df[["unified_customer_id", "predicted_cltv", "cltv_decile"]],
        on="unified_customer_id", how="left"
    ).merge(
        lead_df[["unified_customer_id", "lead_score"]],
        on="unified_customer_id", how="left"
    )

    # Calculate churn risk proxy (recency_days / 180 clipped)
    features_master["churn_risk"] = (features_master["recency_days"] / 180.0).clip(0.0, 1.0).round(3)
    features_master["computed_at"] = datetime.now()

    db_features = features_master[[
        "unified_customer_id", "recency_days", "frequency", "monetary",
        "gross_margin", "rfm_score", "rfm_segment", "predicted_cltv",
        "cltv_decile", "lead_score", "churn_risk", "computed_at"
    ]]

    db.conn.execute("DELETE FROM customer_features;")
    db.conn.register("tmp_feats", db_features)
    db.conn.execute("INSERT INTO customer_features SELECT * FROM tmp_feats;")
    
    total_feats = db.query("SELECT COUNT(*) AS c FROM customer_features;").iloc[0]["c"]
    print(f"\nSuccessfully stored {total_feats} 360 customer feature records in DuckDB!")
    db.close()
    return True

if __name__ == "__main__":
    run_pipeline()
