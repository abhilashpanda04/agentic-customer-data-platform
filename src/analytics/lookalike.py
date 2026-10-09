"""
Lookalike Audience Modeling Engine
Features:
- Seed audience selection (e.g. Champions, Top Decile CLTV, Repeat Subscribers).
- Propensity classifier (Logistic Regression baseline & LightGBM) to find lookalikes among non-seed pool.
- Governance & exclusion rules:
  * Excludes customers without marketing consent.
  * Excludes existing seed customers (suppression).
- Lift, seed feature similarity, and precision ranking.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

class LookalikeEngine:
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            n_estimators=70,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbosity=-1
        )

    def generate_lookalikes(
        self,
        seed_ucids: list,
        df_features: pd.DataFrame,
        df_consent: pd.DataFrame,
        top_k: int = 100
    ) -> pd.DataFrame:
        """
        Trains classifier on seed (label=1) vs unselected pool (label=0),
        then scores and ranks non-seed compliant users.
        """
        seed_set = set(seed_ucids)
        df = df_features.copy()

        # Target: 1 if in seed, 0 otherwise
        df["is_seed"] = df["unified_customer_id"].apply(lambda x: 1 if x in seed_set else 0)

        feature_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in ["is_seed"]]
        X = df[feature_cols].fillna(0)
        y = df["is_seed"]

        if y.sum() < 5:
            raise ValueError("Seed audience is too small (need at least 5 profiles).")

        # Fit propensity model
        self.model.fit(X, y)
        probs = self.model.predict_proba(X)[:, 1]
        df["similarity_score"] = np.round(probs, 4)

        # Merge consent status for compliance
        df = df.merge(df_consent, on="unified_customer_id", how="left")

        # Suppress seed members and filter by consent
        prospects = df[
            (df["is_seed"] == 0) & 
            (df["marketing_opt_in"] == True) & 
            (df["do_not_track"] == False)
        ].copy()

        prospects = prospects.sort_values("similarity_score", ascending=False)
        return prospects.head(top_k)

if __name__ == "__main__":
    from src.cdp.database import CDPDatabase
    db = CDPDatabase()
    df_feat = db.query("SELECT * FROM customer_features;")
    df_cons = db.query("SELECT * FROM consent_preference;")
    if len(df_feat) > 0:
        seed = df_feat[df_feat["rfm_segment"] == "Champions"]["unified_customer_id"].tolist()
        engine = LookalikeEngine()
        lookalikes = engine.generate_lookalikes(seed, df_feat, df_cons, top_k=20)
        print("Top 5 Lookalikes:")
        print(lookalikes[["unified_customer_id", "similarity_score"]].head())
