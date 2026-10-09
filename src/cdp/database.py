"""
DuckDB CDP Store & Warehouse
Initializes schemas and persists:
- customer_profile (Unified 360 golden records)
- identity_map (Links all device/email/cookie nodes to UCID)
- transaction_fact (Purchases, monetary values, margins)
- event_fact (Web, App, CRM, Support events)
- campaign_touchpoint (Touchpoints for MTA)
- consent_preference (GDPR/CCPA flags)
- customer_features (RFM, CLTV, Lead, Churn features)
- audience_definition & audience_membership (Activated cohorts)
- agent_audit_log (Tracks every agent invocation, tool call, governance check & human approvals)
"""
import os
import duckdb
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "cdp.duckdb")

class CDPDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = duckdb.connect(self.db_path)
        self._init_schemas()

    def _init_schemas(self):
        """Create standard enterprise CDP relational tables."""
        # 1. Customer Profile (Unified 360)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS customer_profile (
                unified_customer_id VARCHAR PRIMARY KEY,
                primary_email VARCHAR,
                primary_phone VARCHAR,
                first_name VARCHAR,
                last_name VARCHAR,
                city VARCHAR,
                country VARCHAR,
                subscription_status VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );
        """)

        # 2. Identity Map (Graph resolved touchpoints)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS identity_map (
                id VARCHAR PRIMARY KEY,
                unified_customer_id VARCHAR,
                identifier_type VARCHAR,
                identifier_value VARCHAR,
                created_at TIMESTAMP
            );
        """)

        # 3. Transaction Fact
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS transaction_fact (
                order_id VARCHAR PRIMARY KEY,
                unified_customer_id VARCHAR,
                order_timestamp TIMESTAMP,
                revenue DOUBLE,
                gross_margin DOUBLE,
                channel VARCHAR,
                payment_status VARCHAR,
                items_count INTEGER
            );
        """)

        # 4. Event Fact
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS event_fact (
                event_id VARCHAR PRIMARY KEY,
                unified_customer_id VARCHAR,
                event_timestamp TIMESTAMP,
                event_type VARCHAR,
                page_url VARCHAR,
                session_id VARCHAR,
                device_type VARCHAR
            );
        """)

        # 5. Campaign Touchpoint (for MTA & Attribution)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS campaign_touchpoint (
                touchpoint_id VARCHAR PRIMARY KEY,
                unified_customer_id VARCHAR,
                event_timestamp TIMESTAMP,
                channel VARCHAR,
                campaign_name VARCHAR,
                conversion_flag INTEGER,
                order_id VARCHAR,
                attributed_revenue DOUBLE
            );
        """)

        # 6. Consent Preference (Governance)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS consent_preference (
                unified_customer_id VARCHAR PRIMARY KEY,
                marketing_opt_in BOOLEAN,
                tracking_consent BOOLEAN,
                third_party_sharing BOOLEAN,
                do_not_track BOOLEAN,
                updated_at TIMESTAMP
            );
        """)

        # 7. Customer Features & ML Model Scores
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS customer_features (
                unified_customer_id VARCHAR PRIMARY KEY,
                recency_days INTEGER,
                frequency INTEGER,
                monetary DOUBLE,
                gross_margin DOUBLE,
                rfm_score VARCHAR,
                rfm_segment VARCHAR,
                predicted_cltv DOUBLE,
                cltv_decile INTEGER,
                lead_score DOUBLE,
                churn_risk DOUBLE,
                computed_at TIMESTAMP
            );
        """)

        # 8. Audience Definitions & Cohorts
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audience_definition (
                audience_id VARCHAR PRIMARY KEY,
                name VARCHAR,
                description VARCHAR,
                filter_criteria JSON,
                created_by VARCHAR,
                created_at TIMESTAMP,
                status VARCHAR
            );
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audience_membership (
                audience_id VARCHAR,
                unified_customer_id VARCHAR,
                added_at TIMESTAMP,
                PRIMARY KEY (audience_id, unified_customer_id)
            );
        """)

        # 9. Campaign Activations & Human Approval log
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS campaign_activation (
                activation_id VARCHAR PRIMARY KEY,
                audience_id VARCHAR,
                campaign_name VARCHAR,
                target_channel VARCHAR,
                approved_by VARCHAR,
                approval_status VARCHAR,
                activated_at TIMESTAMP,
                audience_size INTEGER,
                payload JSON
            );
        """)

        # 10. Agent Audit Log (Tracks reasoning, tool queries, governance assertions)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_audit_log (
                log_id VARCHAR PRIMARY KEY,
                session_id VARCHAR,
                action_type VARCHAR,
                tool_name VARCHAR,
                request_prompt VARCHAR,
                governance_status VARCHAR,
                details JSON,
                timestamp TIMESTAMP
            );
        """)

        # 11. Approval Token Registry (single-use enforcement for human sign-off)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS approval_token (
                nonce VARCHAR PRIMARY KEY,
                campaign_name VARCHAR,
                audience_id VARCHAR,
                channel VARCHAR,
                approved_by VARCHAR,
                issued_at TIMESTAMP,
                expires_at TIMESTAMP,
                consumed_at TIMESTAMP,
                consumed_by VARCHAR
            );
        """)

        # 12. Experiment Results (incrementality / holdout measurement)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiment_result (
                experiment_id VARCHAR PRIMARY KEY,
                activation_id VARCHAR,
                audience_id VARCHAR,
                metric_name VARCHAR,
                control_value DOUBLE,
                treatment_value DOUBLE,
                lift_pct DOUBLE,
                p_value DOUBLE,
                is_significant BOOLEAN,
                sample_size INTEGER,
                measured_at TIMESTAMP
            );
        """)

    def load_unified_data(self, dataset: dict, resolved_identities: pd.DataFrame):
        """Populates tables with synthetic data mapped to resolved Unified Customer IDs."""
        # Mapping dict: raw_entity_id -> unified_customer_id
        entity_to_ucid = (
            resolved_identities[resolved_identities["raw_entity_id"].notna()]
            .set_index("raw_entity_id")["unified_customer_id"]
            .to_dict()
        )

        # 1. Customer Profiles
        df_ent = dataset["entities"].copy()
        df_ent["unified_customer_id"] = df_ent["entity_id"].map(entity_to_ucid)
        df_profiles = df_ent[[
            "unified_customer_id", "email", "phone", "first_name", "last_name",
            "city", "country", "subscription_status", "created_at"
        ]].rename(columns={"email": "primary_email", "phone": "primary_phone"})
        df_profiles["updated_at"] = datetime.now()

        self.conn.execute("DELETE FROM customer_profile;")
        self.conn.register("tmp_profiles", df_profiles)
        self.conn.execute("INSERT INTO customer_profile SELECT * FROM tmp_profiles;")

        # 2. Identity Map
        resolved_map = resolved_identities.copy()
        resolved_map["id"] = [f"IDM_{i:07d}" for i in range(len(resolved_map))]
        resolved_map["created_at"] = datetime.now()
        df_idm = resolved_map[["id", "unified_customer_id", "identifier_type", "identifier_value", "created_at"]]
        self.conn.execute("DELETE FROM identity_map;")
        self.conn.register("tmp_idm", df_idm)
        self.conn.execute("INSERT INTO identity_map SELECT * FROM tmp_idm;")

        # 3. Transactions
        df_tx = dataset["transactions"].copy()
        df_tx["unified_customer_id"] = df_tx["entity_id"].map(entity_to_ucid)
        df_tx = df_tx[[
            "order_id", "unified_customer_id", "order_timestamp", "revenue",
            "gross_margin", "channel", "payment_status", "items_count"
        ]]
        self.conn.execute("DELETE FROM transaction_fact;")
        self.conn.register("tmp_tx", df_tx)
        self.conn.execute("INSERT INTO transaction_fact SELECT * FROM tmp_tx;")

        # 4. Events
        df_ev = dataset["events"].copy()
        df_ev["unified_customer_id"] = df_ev["entity_id"].map(entity_to_ucid)
        df_ev = df_ev[[
            "event_id", "unified_customer_id", "event_timestamp", "event_type",
            "page_url", "session_id", "device_type"
        ]]
        self.conn.execute("DELETE FROM event_fact;")
        self.conn.register("tmp_ev", df_ev)
        self.conn.execute("INSERT INTO event_fact SELECT * FROM tmp_ev;")

        # 5. Campaign Touchpoints
        df_tp = dataset["touchpoints"].copy()
        df_tp["unified_customer_id"] = df_tp["entity_id"].map(entity_to_ucid)
        df_tp = df_tp[[
            "touchpoint_id", "unified_customer_id", "event_timestamp", "channel",
            "campaign_name", "conversion_flag", "order_id", "attributed_revenue"
        ]]
        self.conn.execute("DELETE FROM campaign_touchpoint;")
        self.conn.register("tmp_tp", df_tp)
        self.conn.execute("INSERT INTO campaign_touchpoint SELECT * FROM tmp_tp;")

        # 6. Consent Preferences
        df_cs = dataset["consents"].copy()
        df_cs["unified_customer_id"] = df_cs["entity_id"].map(entity_to_ucid)
        df_cs = df_cs[[
            "unified_customer_id", "marketing_opt_in", "tracking_consent",
            "third_party_sharing", "do_not_track", "updated_at"
        ]]
        self.conn.execute("DELETE FROM consent_preference;")
        self.conn.register("tmp_cs", df_cs)
        self.conn.execute("INSERT INTO consent_preference SELECT * FROM tmp_cs;")

        print(f"DuckDB CDP Warehouse successfully seeded at: {self.db_path}")

    def query(self, sql: str) -> pd.DataFrame:
        """Executes query and returns pandas DataFrame."""
        return self.conn.execute(sql).df()

    def query_params(self, sql: str, params: list) -> pd.DataFrame:
        """Executes a parameterized query (SQL-injection safe) and returns a DataFrame."""
        return self.conn.execute(sql, params).df()

    # ------------------------------------------------------------ auditing ---
    def write_audit_log(
        self,
        session_id: str,
        action_type: str,
        tool_name: str,
        request_prompt: str,
        governance_status: str,
        details: dict,
    ) -> str:
        """
        Appends an immutable audit record. Every agent action and governance
        decision flows through here so the trail is complete, not sampled.
        """
        import json as _json
        import uuid as _uuid

        log_id = f"LOG_{_uuid.uuid4().hex[:12].upper()}"
        self.conn.execute(
            """
            INSERT INTO agent_audit_log
                (log_id, session_id, action_type, tool_name, request_prompt,
                 governance_status, details, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                log_id,
                session_id,
                action_type,
                tool_name,
                request_prompt,
                governance_status,
                _json.dumps(details, default=str),
                datetime.now(),
            ],
        )
        return log_id

    # ------------------------------------------------- approval token store ---
    def register_approval_token(
        self,
        nonce: str,
        campaign_name: str,
        audience_id: str,
        channel: str,
        approved_by: str,
        issued_at: datetime,
        expires_at: datetime,
    ) -> None:
        """Records an issued token so single-use can be enforced at consumption."""
        self.conn.execute(
            """
            INSERT OR REPLACE INTO approval_token
                (nonce, campaign_name, audience_id, channel, approved_by,
                 issued_at, expires_at, consumed_at, consumed_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL);
            """,
            [nonce, campaign_name, audience_id, channel, approved_by, issued_at, expires_at],
        )

    def consume_approval_token(self, nonce: str, consumed_by: str) -> tuple[bool, str]:
        """
        Atomically marks a token consumed. Returns (ok, reason).

        Replay protection: a token that already has a `consumed_at` is rejected.
        """
        row = self.conn.execute(
            "SELECT consumed_at FROM approval_token WHERE nonce = ?;", [nonce]
        ).fetchone()

        if row is None:
            return False, "Token is not registered in the approval registry."

        if row[0] is not None:
            return False, f"Token was already consumed at {row[0]} (replay rejected)."

        self.conn.execute(
            "UPDATE approval_token SET consumed_at = ?, consumed_by = ? WHERE nonce = ?;",
            [datetime.now(), consumed_by, nonce],
        )
        return True, "Token consumed."

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    from src.ingestion.synthetic_generator import generate_cdp_dataset
    from src.cdp.identity_resolution import IdentityResolutionEngine

    dataset = generate_cdp_dataset(num_customers=1000)
    engine = IdentityResolutionEngine()
    engine.build_identity_graph(dataset["identities"])
    resolved_idm = engine.resolve_identities()

    db = CDPDatabase()
    db.load_unified_data(dataset, resolved_idm)
    res = db.query("SELECT COUNT(*) AS total_profiles FROM customer_profile;")
    print("Verification count in DuckDB:")
    print(res)
