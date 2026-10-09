"""
Lead Scoring & Conversion Propensity Engine

Predicts a genuinely forward-looking outcome: **will this customer purchase in
the next 30 days**, based only on features observable at the cutoff date.

Why this matters
----------------
An earlier version of this module used "has this customer *ever* purchased" as the
target. Because ~97% of customers had purchased, the label was nearly constant and
the resulting PR-AUC (0.986) and lift (1.03x) were artefacts of base rate, not
model skill. This version mirrors the observation/performance split used by the
CLTV module:

    |<---- observation window ---->|<-- 30-day label window -->|
    features computed here          target measured here

Evaluation: ROC-AUC, PR-AUC, Brier score (calibration), decile lift, and
precision/recall at top-k.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

#: Columns the model is allowed to learn from (all pre-cutoff, no leakage).
FEATURE_COLUMNS = [
    "total_sessions",
    "total_pageviews",
    "pricing_views",
    "product_views",
    "cart_adds",
    "form_submits",
    "support_tickets",
    "uses_desktop",
    "uses_mobile",
    "events_last_7d",
    "events_last_30d",
    "recency_of_activity_days",
    "hist_orders",
    "hist_revenue",
    "hist_recency_days",
]


class LeadScoringEngine:
    def __init__(self, label_window_days: int = 30):
        self.label_window_days = label_window_days
        self.model = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbosity=-1,
        )
        self.metrics: Dict[str, Any] = {}
        self.cutoff_date: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------ features ---
    def extract_features(
        self,
        df_profiles: pd.DataFrame,
        df_events: pd.DataFrame,
        df_tx: pd.DataFrame,
        cutoff_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Builds pre-cutoff features and a forward-looking 30-day purchase label.

        Returns a frame with `unified_customer_id`, feature columns, and
        `converted_flag` (1 if the customer purchased within the label window).
        """
        ev = df_events.copy()
        tx = df_tx.copy()
        ev["event_timestamp"] = pd.to_datetime(ev["event_timestamp"])
        tx["order_timestamp"] = pd.to_datetime(tx["order_timestamp"])

        if cutoff_date is None:
            max_date = max(ev["event_timestamp"].max(), tx["order_timestamp"].max())
            cutoff_date = max_date - pd.Timedelta(days=self.label_window_days)
        self.cutoff_date = cutoff_date

        # ---- observation window only (prevents target leakage) ----
        hist_ev = ev[ev["event_timestamp"] <= cutoff_date]
        hist_tx = tx[tx["order_timestamp"] <= cutoff_date]

        agg_events = (
            hist_ev.groupby("unified_customer_id")
            .agg(
                total_sessions=("session_id", "nunique"),
                total_pageviews=("event_id", "count"),
                pricing_views=("event_type", lambda s: (s == "pricing_view").sum()),
                product_views=("event_type", lambda s: (s == "product_view").sum()),
                cart_adds=("event_type", lambda s: (s == "add_to_cart").sum()),
                form_submits=("event_type", lambda s: (s == "lead_form_submitted").sum()),
                support_tickets=("event_type", lambda s: (s == "support_ticket").sum()),
                uses_desktop=("device_type", lambda s: (s == "desktop").sum()),
                uses_mobile=("device_type", lambda s: (s == "mobile").sum()),
                last_event_ts=("event_timestamp", "max"),
            )
            .reset_index()
        )

        # Momentum features: recent engagement relative to overall.
        for days, col in ((7, "events_last_7d"), (30, "events_last_30d")):
            window_start = cutoff_date - pd.Timedelta(days=days)
            recent = (
                hist_ev[hist_ev["event_timestamp"] > window_start]
                .groupby("unified_customer_id")["event_id"]
                .count()
                .rename(col)
            )
            agg_events = agg_events.merge(
                recent, on="unified_customer_id", how="left"
            )

        agg_tx = (
            hist_tx.groupby("unified_customer_id")
            .agg(
                hist_orders=("order_id", "nunique"),
                hist_revenue=("revenue", "sum"),
                hist_last_order=("order_timestamp", "max"),
            )
            .reset_index()
        )

        df = df_profiles[["unified_customer_id"]].copy()
        df = df.merge(agg_events, on="unified_customer_id", how="left")
        df = df.merge(agg_tx, on="unified_customer_id", how="left")

        # ---- forward-looking label: purchase within the next N days? ----
        label_end = cutoff_date + pd.Timedelta(days=self.label_window_days)
        future_buyers = set(
            tx.loc[
                (tx["order_timestamp"] > cutoff_date)
                & (tx["order_timestamp"] <= label_end),
                "unified_customer_id",
            ].unique()
        )
        df["converted_flag"] = (
            df["unified_customer_id"].isin(future_buyers).astype(int)
        )

        # ---- derived features ----
        df["recency_of_activity_days"] = (
            cutoff_date - pd.to_datetime(df["last_event_ts"])
        ).dt.days
        df["hist_recency_days"] = (
            cutoff_date - pd.to_datetime(df["hist_last_order"])
        ).dt.days

        # Customers with no history at all are informative, not missing:
        # zero-fill counts, but keep recency sentinel above the window.
        fill_zero = [c for c in FEATURE_COLUMNS if c not in ("recency_of_activity_days", "hist_recency_days")]
        df[fill_zero] = df[fill_zero].fillna(0)
        df["recency_of_activity_days"] = df["recency_of_activity_days"].fillna(999)
        df["hist_recency_days"] = df["hist_recency_days"].fillna(999)

        return df[["unified_customer_id"] + FEATURE_COLUMNS + ["converted_flag"]]

    # --------------------------------------------------------------- train ---
    def train_and_score(self, df_features: pd.DataFrame) -> pd.DataFrame:
        X = df_features[FEATURE_COLUMNS]
        y = df_features["converted_flag"]

        positives = int(y.sum())
        negatives = int(len(y) - positives)

        if positives < 10 or negatives < 10:
            # Degenerate label: refuse to report misleading metrics.
            self.metrics = {
                "status": "DEGENERATE_LABEL",
                "positives": positives,
                "negatives": negatives,
                "positive_rate": round(float(y.mean()), 4) if len(y) else 0.0,
                "note": (
                    "Label is (near) constant; no reliable propensity model can be "
                    "fitted. Check the observation/label window split."
                ),
            }
            df_features = df_features.copy()
            df_features["lead_score"] = float(positives) / max(len(y), 1)
            df_features["lead_grade"] = "Insufficient-Signal"
            return df_features[["unified_customer_id", "lead_score", "lead_grade"]]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train)

        test_probs = self.model.predict_proba(X_test)[:, 1]
        base_rate = float(y_test.mean())

        roc_auc = roc_auc_score(y_test, test_probs)
        pr_auc = average_precision_score(y_test, test_probs)
        brier = brier_score_loss(y_test, test_probs)

        eval_df = pd.DataFrame({"actual": y_test.to_numpy(), "prob": test_probs})
        eval_df["decile"] = pd.qcut(
            eval_df["prob"].rank(method="first"), 10, labels=range(1, 11)
        )
        top_decile = eval_df[eval_df["decile"] == 10]
        top_decile_cvr = float(top_decile["actual"].mean())
        lift = (top_decile_cvr / base_rate) if base_rate > 0 else 1.0

        # Precision / recall at top 10% (the actionable sales-queue cut).
        k = max(1, int(len(eval_df) * 0.10))
        top_k_idx = np.argsort(-test_probs)[:k]
        y_topk = y_test.to_numpy()[top_k_idx]
        precision_at_k = float(precision_score(y_topk, np.ones_like(y_topk), zero_division=0))
        recall_at_k = float(recall_score(y_test.to_numpy(), np.isin(np.arange(len(y_test)), top_k_idx).astype(int), zero_division=0))

        self.metrics = {
            "status": "OK",
            "label_window_days": self.label_window_days,
            "cutoff_date": str(self.cutoff_date.date()) if self.cutoff_date is not None else None,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "positive_rate": round(base_rate, 4),
            "roc_auc": round(float(roc_auc), 3),
            "pr_auc": round(float(pr_auc), 3),
            "brier_score": round(float(brier), 3),
            "top_decile_lift": round(float(lift), 2),
            "top_decile_cvr": round(top_decile_cvr, 3),
            "precision_at_top_10pct": round(precision_at_k, 3),
            "recall_at_top_10pct": round(recall_at_k, 3),
        }

        df_out = df_features.copy()
        df_out["lead_score"] = np.round(self.model.predict_proba(X)[:, 1], 4)
        df_out["lead_grade"] = pd.cut(
            df_out["lead_score"],
            bins=[-0.001, 0.15, 0.35, 0.6, 1.0],
            labels=["Cold", "Warm", "Hot", "Sales-Ready"],
        ).astype(str)

        return df_out[["unified_customer_id", "lead_score", "lead_grade"]]


if __name__ == "__main__":
    from src.cdp.database import CDPDatabase

    db = CDPDatabase()
    engine = LeadScoringEngine()
    feats = engine.extract_features(
        db.query("SELECT * FROM customer_profile;"),
        db.query("SELECT * FROM event_fact;"),
        db.query("SELECT * FROM transaction_fact;"),
    )
    scored = engine.train_and_score(feats)
    print("Lead Scoring Metrics:", engine.metrics)
    print(scored["lead_grade"].value_counts())
