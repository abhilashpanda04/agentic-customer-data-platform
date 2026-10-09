"""
RFM Segmentation Engine
Calculates:
- Recency: Days since customer's last purchase.
- Frequency: Number of unique purchase orders.
- Monetary: Total historical revenue and gross margin.
Assigns quintile scores (1 to 5) and generates standardized customer segments:
- Champions (555, 554, 545, 544)
- Loyal Customers (4-5 Frequency, high monetary)
- Potential Loyalists (Recent buyers, medium frequency)
- New Customers (High recency, low frequency)
- At-Risk High Value (High historical monetary & frequency, low recency)
- Hibernating / Churned (Low recency, low frequency)
- One-Time / Low Value
"""
import pandas as pd
import numpy as np
from datetime import datetime

class RFMAnalyzer:
    def __init__(self, reference_date: datetime = None):
        self.reference_date = reference_date

    def compute_rfm(self, df_transactions: pd.DataFrame, df_profiles: pd.DataFrame) -> pd.DataFrame:
        if self.reference_date is None:
            max_date = pd.to_datetime(df_transactions["order_timestamp"]).max()
            self.reference_date = max_date + pd.Timedelta(days=1)
        else:
            self.reference_date = pd.to_datetime(self.reference_date)

        # Aggregate per unified customer
        tx = df_transactions.copy()
        tx["order_timestamp"] = pd.to_datetime(tx["order_timestamp"])

        rfm = tx.groupby("unified_customer_id").agg(
            recency_days=("order_timestamp", lambda dates: (self.reference_date - dates.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("revenue", "sum"),
            gross_margin=("gross_margin", "sum")
        ).reset_index()

        # Merge with all profiles so non-buyers are accounted for
        all_cust = df_profiles[["unified_customer_id"]].drop_duplicates()
        rfm_full = all_cust.merge(rfm, on="unified_customer_id", how="left")

        # Impute non-buyers
        rfm_full["recency_days"] = rfm_full["recency_days"].fillna(999).astype(int)
        rfm_full["frequency"] = rfm_full["frequency"].fillna(0).astype(int)
        rfm_full["monetary"] = rfm_full["monetary"].fillna(0.0).round(2)
        rfm_full["gross_margin"] = rfm_full["gross_margin"].fillna(0.0).round(2)

        # Assign quantile scores (1 to 5)
        # Recency: lower days is better -> rank 5 is most recent
        rfm_full["r_score"] = pd.qcut(rfm_full["recency_days"].rank(method="first", ascending=False), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm_full["f_score"] = pd.qcut(rfm_full["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm_full["m_score"] = pd.qcut(rfm_full["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

        rfm_full["rfm_score"] = (
            rfm_full["r_score"].astype(str) +
            rfm_full["f_score"].astype(str) +
            rfm_full["m_score"].astype(str)
        )

        rfm_full["rfm_segment"] = rfm_full.apply(self._assign_segment, axis=1)
        return rfm_full

    def _assign_segment(self, row) -> str:
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if row["frequency"] == 0:
            return "Prospect (Zero Purchases)"
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif f >= 4 and m >= 3:
            return "Loyal Customers"
        elif r >= 4 and f in [1, 2]:
            return "New / Promising Customers"
        elif r >= 3 and f >= 3:
            return "Potential Loyalists"
        elif r <= 2 and f >= 3 and m >= 3:
            return "At-Risk High Value"
        elif r <= 2 and f >= 2:
            return "Needs Attention"
        elif r <= 2 and f == 1:
            return "Hibernating"
        else:
            return "About to Sleep"

if __name__ == "__main__":
    from src.cdp.database import CDPDatabase
    db = CDPDatabase()
    df_tx = db.query("SELECT * FROM transaction_fact;")
    df_profiles = db.query("SELECT * FROM customer_profile;")
    analyzer = RFMAnalyzer()
    rfm_df = analyzer.compute_rfm(df_tx, df_profiles)
    print("RFM Segmentation Breakdown:")
    print(rfm_df["rfm_segment"].value_counts())
