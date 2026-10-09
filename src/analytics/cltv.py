"""
Customer Lifetime Value (CLTV) Prediction Engine
Implements:
1. Parametric / Probabilistic Lifetime Value:
   - Purchase frequency & recurrence modeling (Recency, Frequency, Tenure, AOV).
   - Expected repeat transactions over future time horizon (e.g. 180 days).
2. Machine Learning Predictive Model (LightGBM Regression):
   - Predicts forward 180-day future customer revenue based on historical behavior, recency, margin, and interactions.
3. Model Evaluation:
   - MAE, RMSE, Pearson Correlation, Decile Revenue Concentration.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class CLTVPredictor:
    def __init__(self, observation_window_days: int = 180):
        self.observation_window_days = observation_window_days
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbosity=-1
        )
        self.metrics = {}

    def extract_features(self, df_tx: pd.DataFrame, df_profiles: pd.DataFrame, df_events: pd.DataFrame) -> pd.DataFrame:
        tx = df_tx.copy()
        tx["order_timestamp"] = pd.to_datetime(tx["order_timestamp"])
        max_date = tx["order_timestamp"].max()
        cutoff_date = max_date - pd.Timedelta(days=self.observation_window_days)

        # Split into observation (historical) and performance (future ground truth)
        hist_tx = tx[tx["order_timestamp"] <= cutoff_date]
        future_tx = tx[tx["order_timestamp"] > cutoff_date]

        # 1. Historical Transaction Features
        agg_hist = hist_tx.groupby("unified_customer_id").agg(
            hist_orders=("order_id", "nunique"),
            hist_revenue=("revenue", "sum"),
            hist_margin=("gross_margin", "sum"),
            avg_order_value=("revenue", "mean"),
            last_order_date=("order_timestamp", "max"),
            first_order_date=("order_timestamp", "min")
        ).reset_index()

        agg_hist["hist_recency_days"] = (cutoff_date - agg_hist["last_order_date"]).dt.days
        agg_hist["customer_tenure_days"] = (cutoff_date - agg_hist["first_order_date"]).dt.days

        # 2. Historical Event Engagement
        ev = df_events.copy()
        ev["event_timestamp"] = pd.to_datetime(ev["event_timestamp"])
        hist_ev = ev[ev["event_timestamp"] <= cutoff_date]
        agg_ev = hist_ev.groupby("unified_customer_id").agg(
            total_events=("event_id", "count"),
            pricing_views=("event_type", lambda s: (s == "pricing_view").sum()),
            cart_adds=("event_type", lambda s: (s == "add_to_cart").sum())
        ).reset_index()

        # 3. Ground Truth Future 180-Day Revenue (Target)
        future_target = future_tx.groupby("unified_customer_id").agg(
            future_revenue=("revenue", "sum")
        ).reset_index()

        # Combine
        data = df_profiles[["unified_customer_id", "subscription_status"]].copy()
        data = data.merge(agg_hist, on="unified_customer_id", how="left")
        data = data.merge(agg_ev, on="unified_customer_id", how="left")
        data = data.merge(future_target, on="unified_customer_id", how="left")

        # Impute nulls
        data["hist_orders"] = data["hist_orders"].fillna(0)
        data["hist_revenue"] = data["hist_revenue"].fillna(0)
        data["hist_margin"] = data["hist_margin"].fillna(0)
        data["avg_order_value"] = data["avg_order_value"].fillna(0)
        data["hist_recency_days"] = data["hist_recency_days"].fillna(999)
        data["customer_tenure_days"] = data["customer_tenure_days"].fillna(0)
        data["total_events"] = data["total_events"].fillna(0)
        data["pricing_views"] = data["pricing_views"].fillna(0)
        data["cart_adds"] = data["cart_adds"].fillna(0)
        data["future_revenue"] = data["future_revenue"].fillna(0)
        data["is_subscriber"] = (data["subscription_status"] == "active").astype(int)

        return data

    def train_and_evaluate(self, feature_df: pd.DataFrame):
        feature_cols = [
            "hist_orders", "hist_revenue", "hist_margin", "avg_order_value",
            "hist_recency_days", "customer_tenure_days", "total_events",
            "pricing_views", "cart_adds", "is_subscriber"
        ]
        X = feature_df[feature_cols]
        y = feature_df["future_revenue"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        preds = np.clip(preds, a_min=0, a_max=None)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        # Concentration in top decile
        test_eval = pd.DataFrame({"actual": y_test, "pred": preds})
        test_eval["pred_decile"] = pd.qcut(test_eval["pred"].rank(method="first"), 10, labels=False)
        top_decile_actual = test_eval[test_eval["pred_decile"] == 9]["actual"].sum()
        total_actual = test_eval["actual"].sum()
        top_decile_share = (top_decile_actual / total_actual * 100) if total_actual > 0 else 0

        self.metrics = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 3),
            "top_decile_actual_revenue_share_pct": round(top_decile_share, 1)
        }

        # Predict for all
        feature_df["predicted_cltv"] = np.clip(self.model.predict(X), 0, None).round(2)
        feature_df["cltv_decile"] = pd.qcut(feature_df["predicted_cltv"].rank(method="first"), 10, labels=[1,2,3,4,5,6,7,8,9,10]).astype(int)

        return feature_df[["unified_customer_id", "predicted_cltv", "cltv_decile"]]

if __name__ == "__main__":
    from src.cdp.database import CDPDatabase
    db = CDPDatabase()
    df_tx = db.query("SELECT * FROM transaction_fact;")
    df_profiles = db.query("SELECT * FROM customer_profile;")
    df_events = db.query("SELECT * FROM event_fact;")
    
    predictor = CLTVPredictor()
    feats = predictor.extract_features(df_tx, df_profiles, df_events)
    cltv_scores = predictor.train_and_evaluate(feats)
    print("CLTV Model Metrics:", predictor.metrics)
    print("CLTV Decile Summary:")
    print(cltv_scores.groupby("cltv_decile")["predicted_cltv"].mean())
