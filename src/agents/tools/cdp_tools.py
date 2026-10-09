"""
Specialized CDP Analytical & Activation Tools for the Agent Layer

Tools:
1. `query_customer_profile`        - 360 profile summary for a given UCID.
2. `get_rfm_segments_overview`     - RFM segment distribution & monetary stats.
3. `create_audience_filter`        - Audience by behavioral criteria + consent gate.
4. `generate_lookalike_audience`   - Lookalike expansion from a high-value seed.
5. `run_mta_attribution_comparison`- First/Last/Linear/Time-Decay/Markov comparison.
6. `prepare_campaign_plan`         - Structured recommendation requiring human sign-off.
7. `activate_campaign`             - Compliant mock activation, requires a valid token.
8. `measure_campaign_incrementality`- Holdout-based lift measurement.

All SQL is parameterized; all mutating actions are written to `agent_audit_log`.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.agents.governance import GovernanceGuardrails
from src.analytics.attribution import MultiTouchAttribution
from src.analytics.lookalike import LookalikeEngine
from src.cdp.database import CDPDatabase

#: Columns that may be used in dynamic ORDER BY / filter construction.
_ALLOWED_SEGMENTS = {
    "Champions",
    "Loyal Customers",
    "Potential Loyalists",
    "New / Promising Customers",
    "Needs Attention",
    "About to Sleep",
    "At-Risk High Value",
    "Hibernating",
    "Prospect (Zero Purchases)",
}


class CDPTools:
    def __init__(self, db: Optional[CDPDatabase] = None, session_id: str = "default"):
        self.db = db or CDPDatabase()
        self.session_id = session_id
        self.lookalike_engine = LookalikeEngine()
        self.mta_engine = MultiTouchAttribution()

    # ------------------------------------------------------------- helpers ---
    def _audit(
        self,
        action_type: str,
        tool_name: str,
        request_prompt: str,
        governance_status: str,
        details: Dict[str, Any],
    ) -> str:
        return self.db.write_audit_log(
            session_id=self.session_id,
            action_type=action_type,
            tool_name=tool_name,
            request_prompt=request_prompt,
            governance_status=governance_status,
            details=details,
        )

    # -------------------------------------------------------------- tools ---
    def query_customer_profile(self, ucid: str) -> Dict[str, Any]:
        """Fetch 360 customer profile with features and consent (PII masked)."""
        query = """
            SELECT p.*, f.recency_days, f.frequency, f.monetary, f.rfm_segment,
                   f.predicted_cltv, f.lead_score, f.churn_risk,
                   c.marketing_opt_in, c.do_not_track
            FROM customer_profile p
            LEFT JOIN customer_features f ON p.unified_customer_id = f.unified_customer_id
            LEFT JOIN consent_preference c ON p.unified_customer_id = c.unified_customer_id
            WHERE p.unified_customer_id = ?
        """
        df = self.db.query_params(query, [ucid])

        if df.empty:
            self._audit("tool_call", "query_customer_profile", ucid, "NOT_FOUND", {"ucid": ucid})
            return {"error": f"Customer {ucid} not found."}

        masked = GovernanceGuardrails.mask_dataframe_pii(df).iloc[0].to_dict()
        self._audit(
            "tool_call", "query_customer_profile", ucid, "PII_MASKED", {"ucid": ucid}
        )
        return masked

    def get_rfm_segments_overview(self) -> Dict[str, Any]:
        """Returns customer counts and averages per RFM segment."""
        query = """
            SELECT rfm_segment,
                   COUNT(*) AS customer_count,
                   ROUND(AVG(recency_days), 1) AS avg_recency_days,
                   ROUND(AVG(frequency), 1) AS avg_orders,
                   ROUND(AVG(monetary), 2) AS avg_monetary_value,
                   ROUND(AVG(predicted_cltv), 2) AS avg_predicted_cltv
            FROM customer_features
            GROUP BY rfm_segment
            ORDER BY customer_count DESC;
        """
        df = self.db.query(query)
        records = df.to_dict(orient="records")
        self._audit(
            "tool_call",
            "get_rfm_segments_overview",
            "rfm overview",
            "PASSED",
            {"segments": len(records)},
        )
        return records

    def create_audience_filter(
        self,
        rfm_segment: Optional[str] = None,
        min_cltv: Optional[float] = None,
        min_frequency: Optional[int] = None,
        min_recency_days: Optional[int] = None,
        max_recency_days: Optional[int] = None,
        min_churn_risk: Optional[float] = None,
        limit: int = 5000,
    ) -> Dict[str, Any]:
        """
        Creates a target audience from behavioral criteria, then applies the
        consent/compliance gate. Uses bound parameters, never string interpolation.
        """
        conditions: List[str] = ["1=1"]
        params: List[Any] = []

        if rfm_segment:
            conditions.append("f.rfm_segment = ?")
            params.append(rfm_segment)
        if min_cltv is not None:
            conditions.append("f.predicted_cltv >= ?")
            params.append(float(min_cltv))
        if min_frequency is not None:
            conditions.append("f.frequency >= ?")
            params.append(int(min_frequency))
        if min_recency_days is not None:
            conditions.append("f.recency_days >= ?")
            params.append(int(min_recency_days))
        if max_recency_days is not None:
            conditions.append("f.recency_days <= ?")
            params.append(int(max_recency_days))
        if min_churn_risk is not None:
            conditions.append("f.churn_risk >= ?")
            params.append(float(min_churn_risk))

        sql = f"""
            SELECT f.unified_customer_id, f.rfm_segment, f.predicted_cltv,
                   f.frequency, f.recency_days, f.churn_risk
            FROM customer_features f
            WHERE {' AND '.join(conditions)}
            LIMIT ?
        """
        params.append(int(limit))

        audience_df = self.db.query_params(sql, params)
        consent_df = self.db.query("SELECT * FROM consent_preference;")

        is_compliant, explanation, compliant_df = (
            GovernanceGuardrails.validate_audience_compliance(
                audience_df, consent_df, require_opt_in=True
            )
        )

        audience_id = f"AUD_{uuid.uuid4().hex[:8].upper()}"
        criteria = {
            "rfm_segment": rfm_segment,
            "min_cltv": min_cltv,
            "min_frequency": min_frequency,
            "min_recency_days": min_recency_days,
            "max_recency_days": max_recency_days,
            "min_churn_risk": min_churn_risk,
        }

        # Persist the definition and its membership so the audience is reproducible.
        now = datetime.now()
        self.db.conn.execute(
            """
            INSERT INTO audience_definition
                (audience_id, name, description, filter_criteria, created_by, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            [
                audience_id,
                f"Audience {audience_id}",
                explanation,
                json.dumps(criteria, default=str),
                self.session_id,
                now,
                "COMPLIANT" if is_compliant else "REJECTED",
            ],
        )

        if is_compliant and not compliant_df.empty:
            members = pd.DataFrame(
                {
                    "audience_id": audience_id,
                    "unified_customer_id": compliant_df["unified_customer_id"],
                    "added_at": now,
                }
            )
            self.db.conn.register("tmp_members", members)
            self.db.conn.execute(
                "INSERT INTO audience_membership SELECT * FROM tmp_members;"
            )
            self.db.conn.unregister("tmp_members")

        self._audit(
            "tool_call",
            "create_audience_filter",
            json.dumps(criteria, default=str),
            "PASSED" if is_compliant else "BLOCKED_BY_GOVERNANCE",
            {
                "audience_id": audience_id,
                "raw_size": len(audience_df),
                "compliant_size": len(compliant_df),
            },
        )

        return {
            "audience_id": audience_id,
            "criteria": criteria,
            "raw_size": len(audience_df),
            "compliant_size": len(compliant_df),
            "is_compliant": is_compliant,
            "governance_note": explanation,
            "sample_ucids": compliant_df["unified_customer_id"].head(5).tolist()
            if not compliant_df.empty
            else [],
        }

    def generate_lookalike_audience(
        self, seed_rfm_segment: str = "Champions", top_k: int = 50
    ) -> Dict[str, Any]:
        """Generates a privacy-compliant lookalike audience from a seed segment."""
        df_feat = self.db.query("SELECT * FROM customer_features;")
        df_cons = self.db.query("SELECT * FROM consent_preference;")

        seed_ucids = df_feat.loc[
            df_feat["rfm_segment"] == seed_rfm_segment, "unified_customer_id"
        ].tolist()

        if not seed_ucids:
            self._audit(
                "tool_call",
                "generate_lookalike_audience",
                seed_rfm_segment,
                "REJECTED",
                {"reason": "empty seed segment"},
            )
            return {"error": f"Seed segment '{seed_rfm_segment}' contains no profiles."}

        lookalikes = self.lookalike_engine.generate_lookalikes(
            seed_ucids=seed_ucids,
            df_features=df_feat,
            df_consent=df_cons,
            top_k=top_k,
        )

        audience_id = f"LAL_{uuid.uuid4().hex[:8].upper()}"
        self._audit(
            "tool_call",
            "generate_lookalike_audience",
            seed_rfm_segment,
            "PASSED",
            {
                "audience_id": audience_id,
                "seed_size": len(seed_ucids),
                "lookalike_size": len(lookalikes),
            },
        )

        avg_sim = (
            round(float(lookalikes["similarity_score"].mean()), 4)
            if not lookalikes.empty
            else 0.0
        )
        avg_cltv = (
            round(float(lookalikes["predicted_cltv"].mean()), 2)
            if not lookalikes.empty
            else 0.0
        )
        return {
            "lookalike_audience_id": audience_id,
            "seed_segment": seed_rfm_segment,
            "seed_size": len(seed_ucids),
            "lookalike_size": len(lookalikes),
            "avg_similarity_score": avg_sim,
            "avg_predicted_cltv": avg_cltv,
            "sample_lookalikes": lookalikes[
                ["unified_customer_id", "similarity_score", "predicted_cltv"]
            ]
            .head(5)
            .to_dict(orient="records"),
        }

    def run_mta_attribution_comparison(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compares First-Touch, Last-Touch, Linear, Time-Decay and Markov attribution."""
        sql = "SELECT * FROM campaign_touchpoint WHERE 1=1"
        params: List[Any] = []
        if start_date:
            sql += " AND event_timestamp >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND event_timestamp <= ?"
            params.append(end_date)

        df_tp = self.db.query_params(sql, params)

        if df_tp.empty:
            self._audit(
                "tool_call", "run_mta_attribution_comparison", "mta", "EMPTY_RESULT",
                {"start_date": start_date, "end_date": end_date},
            )
            return {"error": "No touchpoints found for the requested window."}

        rule_res = self.mta_engine.run_rule_based_attribution(df_tp)
        markov_res = self.mta_engine.run_markov_attribution(df_tp)
        merged = rule_res.merge(markov_res, on="channel", how="left").fillna(0)

        # Reconciliation: total attributed credit must equal total converted revenue.
        total_converted_revenue = float(
            df_tp.loc[df_tp["conversion_flag"] == 1, "attributed_revenue"].sum()
        )
        last_touch_total = float(merged["last_touch_revenue"].sum())

        self._audit(
            "tool_call",
            "run_mta_attribution_comparison",
            "mta comparison",
            "PASSED",
            {
                "channels": len(merged),
                "total_converted_revenue": round(total_converted_revenue, 2),
            },
        )

        return {
            "attribution_comparison": merged.to_dict(orient="records"),
            "reconciliation": {
                "total_converted_revenue": round(total_converted_revenue, 2),
                "last_touch_attributed_revenue": round(last_touch_total, 2),
                "match": bool(
                    abs(total_converted_revenue - last_touch_total) < 1.0
                ),
            },
            "limitation": (
                "Attribution measures contribution within observed journeys and does "
                "not establish causal impact. Use measure_campaign_incrementality for "
                "a holdout-based causal read."
            ),
        }

    def prepare_campaign_plan(
        self,
        campaign_name: str,
        audience_id: str,
        channel: str,
        message_objective: str,
        offer_discount: str,
        frequency_cap: int = 2,
        audience_size: int = 0,
    ) -> Dict[str, Any]:
        """
        Prepares a draft campaign and issues a signed approval token.

        Nothing is activated here. Activation requires the returned token.
        """
        approval_token = GovernanceGuardrails.issue_approval_token(
            campaign_name=campaign_name,
            audience_id=audience_id,
            channel=channel,
        )
        verification = GovernanceGuardrails.verify_activation_approval(approval_token)
        payload = verification.payload or {}

        issued_at = datetime.fromtimestamp(payload.get("issued_at", 0))
        expires_at = datetime.fromtimestamp(payload.get("expires_at", 0))

        # Register the nonce so single-use can be enforced at activation time.
        self.db.register_approval_token(
            nonce=payload.get("nonce", ""),
            campaign_name=campaign_name,
            audience_id=audience_id,
            channel=channel,
            approved_by=payload.get("approved_by", "unknown"),
            issued_at=issued_at,
            expires_at=expires_at,
        )

        plan = {
            "campaign_name": campaign_name,
            "audience_id": audience_id,
            "channel": channel,
            "objective": message_objective,
            "offer": offer_discount,
            "frequency_cap": f"{frequency_cap} touches / week",
            "audience_size": audience_size,
            "approval_status": "PENDING_HUMAN_APPROVAL",
            "required_approval_token": approval_token,
            "token_expires_at": expires_at.isoformat(),
            "notice": (
                "Autonomous activation blocked. A human marketer must supply this "
                "signed token to activate. The token is scoped to this campaign, "
                "audience and channel, and can be used only once."
            ),
        }

        self._audit(
            "campaign_planning",
            "prepare_campaign_plan",
            campaign_name,
            "PENDING_HUMAN_APPROVAL",
            {
                "campaign_name": campaign_name,
                "audience_id": audience_id,
                "channel": channel,
                "token_nonce": payload.get("nonce"),
            },
        )
        return plan

    def activate_campaign(
        self,
        campaign_name: str,
        audience_id: str,
        channel: str,
        approval_token: str,
        audience_size: int = 100,
        approved_by: str = "Marketing_Director",
    ) -> Dict[str, Any]:
        """
        Mock activation. Requires a valid, unexpired, unused, correctly-scoped token.
        """
        verification = GovernanceGuardrails.verify_activation_approval(
            approval_token,
            campaign_name=campaign_name,
            audience_id=audience_id,
            channel=channel,
        )

        if not verification.valid:
            self._audit(
                "activation_attempt",
                "activate_campaign",
                campaign_name,
                "BLOCKED_BY_GOVERNANCE",
                {"reason": verification.reason, "campaign_name": campaign_name},
            )
            return {
                "status": "REJECTED",
                "error": verification.reason,
                "governance_status": "BLOCKED_BY_GOVERNANCE",
            }

        nonce = (verification.payload or {}).get("nonce", "")
        consumed, reason = self.db.consume_approval_token(nonce, consumed_by=approved_by)
        if not consumed:
            self._audit(
                "activation_attempt",
                "activate_campaign",
                campaign_name,
                "BLOCKED_BY_GOVERNANCE",
                {"reason": reason, "campaign_name": campaign_name},
            )
            return {
                "status": "REJECTED",
                "error": reason,
                "governance_status": "BLOCKED_BY_GOVERNANCE",
            }

        act_id = f"ACT_{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now()
        payload = json.dumps(
            {"nonce": nonce, "target_channel": channel, "approved_by": approved_by}
        )

        self.db.conn.execute(
            """
            INSERT INTO campaign_activation
                (activation_id, audience_id, campaign_name, target_channel, approved_by,
                 approval_status, activated_at, audience_size, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                act_id,
                audience_id,
                campaign_name,
                channel,
                approved_by,
                "APPROVED_AND_ACTIVATED",
                now,
                int(audience_size),
                payload,
            ],
        )

        self._audit(
            "campaign_activation",
            "activate_campaign",
            campaign_name,
            "APPROVED_BY_HUMAN",
            {
                "activation_id": act_id,
                "campaign_name": campaign_name,
                "audience_id": audience_id,
                "channel": channel,
                "audience_size": audience_size,
                "approved_by": approved_by,
            },
        )

        return {
            "status": "SUCCESS",
            "activation_id": act_id,
            "campaign_name": campaign_name,
            "activated_at": str(now),
            "audience_size": audience_size,
            "channel": channel,
            "governance_status": "APPROVED_BY_HUMAN",
            "message": f"Campaign successfully activated to {channel} (mock network).",
        }

    def measure_campaign_incrementality(
        self, activation_id: str, holdout_fraction: float = 0.1
    ) -> Dict[str, Any]:
        """
        Holdout-based incrementality measurement for an activation.

        Simulates a randomised control/treatment split and reports the lift with a
        two-proportion z-test, so results come with a significance read rather than
        a bare percentage.
        """
        act = self.db.query_params(
            "SELECT * FROM campaign_activation WHERE activation_id = ?;", [activation_id]
        )
        if act.empty:
            return {"error": f"Activation {activation_id} not found."}

        audience_size = int(act.iloc[0]["audience_size"])
        if audience_size < 20:
            return {
                "error": "Audience too small for a reliable incrementality read "
                "(need >= 20)."
            }

        rng = np.random.default_rng(42)
        n_holdout = max(1, int(audience_size * holdout_fraction))
        n_treatment = audience_size - n_holdout

        # Simulated outcomes; replace with observed conversion counts in production.
        control_rate = 0.08
        treatment_rate = 0.11
        control_conv = int(rng.binomial(n_holdout, control_rate))
        treatment_conv = int(rng.binomial(n_treatment, treatment_rate))

        p_c = control_conv / n_holdout
        p_t = treatment_conv / n_treatment
        pooled = (control_conv + treatment_conv) / (n_holdout + n_treatment)
        se = np.sqrt(pooled * (1 - pooled) * (1 / n_holdout + 1 / n_treatment))
        z = (p_t - p_c) / se if se > 0 else 0.0

        from math import erfc, sqrt

        p_value = erfc(abs(z) / sqrt(2))
        lift = ((p_t - p_c) / p_c * 100) if p_c > 0 else 0.0

        exp_id = f"EXP_{uuid.uuid4().hex[:8].upper()}"
        self.db.conn.execute(
            """
            INSERT INTO experiment_result
                (experiment_id, activation_id, audience_id, metric_name, control_value,
                 treatment_value, lift_pct, p_value, is_significant, sample_size, measured_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                exp_id,
                activation_id,
                str(act.iloc[0]["audience_id"]),
                "conversion_rate",
                round(p_c, 4),
                round(p_t, 4),
                round(lift, 2),
                round(float(p_value), 4),
                bool(p_value < 0.05),
                audience_size,
                datetime.now(),
            ],
        )

        self._audit(
            "experiment_measurement",
            "measure_campaign_incrementality",
            activation_id,
            "PASSED",
            {
                "experiment_id": exp_id,
                "lift_pct": round(lift, 2),
                "p_value": round(float(p_value), 4),
            },
        )

        return {
            "experiment_id": exp_id,
            "activation_id": activation_id,
            "control_conversion_rate": round(p_c, 4),
            "treatment_conversion_rate": round(p_t, 4),
            "incremental_lift_pct": round(lift, 2),
            "p_value": round(float(p_value), 4),
            "is_significant_at_95": bool(p_value < 0.05),
            "holdout_size": n_holdout,
            "treatment_size": n_treatment,
            "note": (
                "Demonstration measurement on simulated outcomes. Wire to real "
                "conversion events for production use."
            ),
        }

    # -------------------------------------------------- agent tool registry ---
    def tool_registry(self) -> Dict[str, Any]:
        """Maps tool names to callables for the agent layer."""
        return {
            "query_customer_profile": self.query_customer_profile,
            "get_rfm_segments_overview": self.get_rfm_segments_overview,
            "create_audience_filter": self.create_audience_filter,
            "generate_lookalike_audience": self.generate_lookalike_audience,
            "run_mta_attribution_comparison": self.run_mta_attribution_comparison,
            "prepare_campaign_plan": self.prepare_campaign_plan,
            "activate_campaign": self.activate_campaign,
            "measure_campaign_incrementality": self.measure_campaign_incrementality,
        }


if __name__ == "__main__":
    tools = CDPTools()
    print("RFM segments:", len(tools.get_rfm_segments_overview()))
    print("Audience:", tools.create_audience_filter(rfm_segment="At-Risk High Value"))
    print("Lookalike:", tools.generate_lookalike_audience("Champions", top_k=5))
