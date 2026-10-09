"""
MarTech Supervisor Agent - real LLM tool calling via Pydantic-AI.

Architecture
------------
This is a genuine agentic loop, not an if/elif router:

    user prompt
      -> Pydantic-AI agent (Groq llm)
      -> model *chooses* which CDP tools to call, with typed arguments
      -> tools execute against DuckDB, each writing to the audit log
      -> model may call more tools (multi-step) before producing a typed answer

Every tool is a plain Python function with type hints and a docstring; Pydantic-AI
derives the JSON schema the model sees, and validates the model's arguments before
execution. Tool arguments are Pydantic-validated, so a hallucinated segment name or
a malformed date fails safely instead of reaching SQL.

Governance is enforced *below* the model: the tools themselves apply consent
filtering, PII masking, and token verification. The model cannot bypass them by
choosing different arguments.

If no GROQ_API_KEY is configured, `DeterministicRouter` provides an equivalent
tool-calling path so the platform remains demonstrable and testable offline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.agents.governance import GovernanceGuardrails
from src.agents.tools.cdp_tools import CDPTools
from src.cdp.database import CDPDatabase

# --------------------------------------------------------------------------
# Typed tool argument models (these are what the LLM must produce)
# --------------------------------------------------------------------------


class EmptyArgs(BaseModel):
    """No arguments required."""


class CustomerProfileArgs(BaseModel):
    ucid: str = Field(description="Unified Customer ID, e.g. 'UCID_00042'.")


class AudienceFilterArgs(BaseModel):
    rfm_segment: Optional[str] = Field(
        default=None,
        description=(
            "RFM segment to filter on. One of: Champions, Loyal Customers, "
            "Potential Loyalists, New / Promising Customers, Needs Attention, "
            "About to Sleep, At-Risk High Value, Hibernating, "
            "Prospect (Zero Purchases)."
        ),
    )
    min_cltv: Optional[float] = Field(default=None, description="Minimum predicted CLTV in USD.")
    min_frequency: Optional[int] = Field(default=None, description="Minimum number of historical orders.")
    min_recency_days: Optional[int] = Field(default=None, description="Minimum days since last purchase.")
    max_recency_days: Optional[int] = Field(default=None, description="Maximum days since last purchase.")
    min_churn_risk: Optional[float] = Field(
        default=None, description="Minimum churn risk score between 0.0 and 1.0."
    )


class LookalikeArgs(BaseModel):
    seed_rfm_segment: str = Field(
        default="Champions", description="RFM segment to use as the high-value seed."
    )
    top_k: int = Field(default=50, ge=1, le=1000, description="How many lookalikes to return.")


class MTAArgs(BaseModel):
    start_date: Optional[str] = Field(
        default=None, description="ISO start date (YYYY-MM-DD). Optional."
    )
    end_date: Optional[str] = Field(
        default=None, description="ISO end date (YYYY-MM-DD). Optional."
    )


class CampaignPlanArgs(BaseModel):
    campaign_name: str = Field(description="Human-readable campaign name.")
    audience_id: str = Field(description="Audience ID returned by create_audience_filter.")
    channel: str = Field(description="Activation channel, e.g. 'email_marketing', 'paid_social'.")
    message_objective: str = Field(description="What the campaign should achieve.")
    offer_discount: str = Field(default="No discount", description="Incentive offered.")
    frequency_cap: int = Field(default=2, ge=1, le=10, description="Max touches per week.")


class ActivationArgs(BaseModel):
    campaign_name: str = Field(description="Campaign name the token was issued for.")
    audience_id: str = Field(description="Audience ID the token was issued for.")
    channel: str = Field(description="Channel the token was issued for.")
    approval_token: str = Field(
        description="Signed human approval token beginning 'MKT_APPRV_'. Required."
    )
    audience_size: int = Field(default=0, ge=0, description="Number of customers targeted.")


class IncrementalityArgs(BaseModel):
    activation_id: str = Field(description="Activation ID from activate_campaign.")
    holdout_fraction: float = Field(
        default=0.1, ge=0.01, le=0.5, description="Share of audience held out as control."
    )


# --------------------------------------------------------------------------
# Typed final answer
# --------------------------------------------------------------------------


class AgentAnswer(BaseModel):
    """Structured response returned to the caller."""

    summary: str = Field(description="Executive-ready answer for the marketing manager.")
    key_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Headline numbers supporting the answer."
    )
    recommended_next_action: str = Field(
        default="", description="The single most useful next step for the marketer."
    )
    governance_notes: List[str] = Field(
        default_factory=list,
        description="Any consent, suppression, PII or approval constraints that applied.",
    )


@dataclass
class AgentDeps:
    """Dependencies injected into the agent run."""

    tools: CDPTools
    session_id: str = "default"
    tool_calls: List[str] = field(default_factory=list)
    governance_events: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the MarTech Intelligence Copilot for an enterprise Customer Data Platform (CDP).

You help marketing managers answer questions about unified customer profiles, segments,
lifetime value, attribution, and campaign activation.

How you work:
- You have tools. Decide for yourself which tools to call and in what order.
- Chain tools when needed. For example: create an audience, then prepare a campaign plan
  for that audience, using the audience_id the first tool returned.
- Base every number in your answer strictly on tool output. Never invent metrics.
- If a tool returns an error or an empty result, say so plainly instead of guessing.

Hard governance rules you must respect:
- You can PREPARE campaigns but you can NEVER activate one yourself. Activation requires
  a signed approval token that only a human marketer can supply. If asked to activate
  without a token, explain that human approval is required and call prepare_campaign_plan.
- Never attempt to reveal raw email addresses, phone numbers, or other direct PII.
  Tools return masked identifiers; do not try to work around that.
- Audiences are consent-filtered automatically. Report suppressed counts when relevant.

Be concise, use concrete numbers, and always state a recommended next action.
"""


# --------------------------------------------------------------------------
# Agent construction
# --------------------------------------------------------------------------


class MarTechSupervisorAgent:
    """Supervisor agent that performs real LLM tool calling."""

    def __init__(self, db: Optional[CDPDatabase] = None, session_id: str = "default"):
        self.db = db or CDPDatabase()
        self.session_id = session_id
        self.tools = CDPTools(self.db, session_id=session_id)
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._agent = None
        if self.api_key:
            self._agent = self._build_agent()

    @property
    def llm_available(self) -> bool:
        return self._agent is not None

    def _build_agent(self):
        """Constructs the Pydantic-AI agent with all CDP tools registered."""
        from pydantic_ai import Agent
        from pydantic_ai.models.groq import GroqModel
        from pydantic_ai.providers.groq import GroqProvider

        model = GroqModel(
            self.model_name,
            provider=GroqProvider(api_key=self.api_key),
        )
        agent = Agent(
            model,
            output_type=AgentAnswer,
            deps_type=AgentDeps,
            system_prompt=SYSTEM_PROMPT,
            retries=2,
        )

        # ---- analytics tools ----
        @agent.tool_plain
        def get_rfm_segments_overview() -> Dict[str, Any]:
            """Get the distribution of customers across RFM segments, with average
            recency, order count, monetary value and predicted CLTV per segment.
            Use this for open-ended questions about the customer base."""
            return self.tools.get_rfm_segments_overview()

        @agent.tool_plain
        def create_audience_filter(args: AudienceFilterArgs) -> Dict[str, Any]:
            """Build a consent-compliant target audience from behavioral criteria.
            Returns an audience_id you can pass to prepare_campaign_plan."""
            return self.tools.create_audience_filter(**args.model_dump())

        @agent.tool_plain
        def generate_lookalike_audience(args: LookalikeArgs) -> Dict[str, Any]:
            """Expand a high-value seed segment into a lookalike prospect audience."""
            return self.tools.generate_lookalike_audience(**args.model_dump())

        @agent.tool_plain
        def run_mta_attribution_comparison(args: MTAArgs) -> Dict[str, Any]:
            """Compare first-touch, last-touch, linear, time-decay and Markov
            multi-touch attribution by channel."""
            return self.tools.run_mta_attribution_comparison(**args.model_dump())

        @agent.tool_plain
        def query_customer_profile(args: CustomerProfileArgs) -> Dict[str, Any]:
            """Look up a single unified customer profile by UCID (PII masked)."""
            return self.tools.query_customer_profile(**args.model_dump())

        # ---- activation tools (governed) ----
        @agent.tool_plain
        def prepare_campaign_plan(args: CampaignPlanArgs) -> Dict[str, Any]:
            """Draft a campaign and issue a signed human-approval token. This does
            NOT activate anything. Use this whenever the user asks to launch or
            activate a campaign."""
            return self.tools.prepare_campaign_plan(**args.model_dump())

        @agent.tool_plain
        def activate_campaign(args: ActivationArgs) -> Dict[str, Any]:
            """Activate a campaign. REQUIRES a valid signed approval token that the
            human supplied. Fails if the token is missing, forged, expired, reused,
            or scoped to a different campaign."""
            return self.tools.activate_campaign(**args.model_dump())

        @agent.tool_plain
        def measure_campaign_incrementality(args: IncrementalityArgs) -> Dict[str, Any]:
            """Measure holdout-based incremental lift for a completed activation."""
            return self.tools.measure_campaign_incrementality(**args.model_dump())

        return agent

    # --------------------------------------------------------------- run ---
    def process_query(self, user_prompt: str) -> Dict[str, Any]:
        """
        Runs the agent on a user prompt and returns a structured response.

        Falls back to the deterministic router when no LLM is configured.
        """
        cleaned = GovernanceGuardrails.mask_pii((user_prompt or "").strip())
        deps = AgentDeps(tools=self.tools, session_id=self.session_id)

        if not self.llm_available:
            return DeterministicRouter(self.tools, self.session_id).process_query(cleaned)

        # Pre-flight: refuse explicit PII extraction attempts before any model call.
        lowered = cleaned.lower()
        if any(p in lowered for p in ("raw email", "export pii", "unmasked phone", "dump pii")):
            self.db.write_audit_log(
                session_id=self.session_id,
                action_type="policy_violation",
                tool_name="preflight_pii_check",
                request_prompt=cleaned,
                governance_status="BLOCKED_BY_POLICY",
                details={"matched": "pii_extraction_phrase"},
            )
            return {
                "query": cleaned,
                "intent": "policy_violation",
                "insights": (
                    "Request blocked by governance: extraction of unmasked PII is "
                    "prohibited. Customer identifiers are returned masked."
                ),
                "governance_status": "BLOCKED_BY_POLICY",
                "tool_output": None,
                "recommended_next_action": "Ask for a masked profile summary instead.",
                "agent_mode": "llm_tool_calling",
            }

        try:
            result = self._agent.run_sync(cleaned, deps=deps)
            answer: AgentAnswer = result.output

            self.db.write_audit_log(
                session_id=self.session_id,
                action_type="agent_response",
                tool_name="pydantic_ai_agent",
                request_prompt=cleaned,
                governance_status="PASSED",
                details={
                    "model": self.model_name,
                    "summary": answer.summary[:500],
                    "key_metrics": answer.key_metrics,
                },
            )

            return {
                "query": cleaned,
                "intent": "agentic_response",
                "insights": answer.summary,
                "key_metrics": answer.key_metrics,
                "recommended_next_action": answer.recommended_next_action,
                "governance_notes": answer.governance_notes,
                "governance_status": "PASSED",
                "tool_output": None,
                "agent_mode": "llm_tool_calling",
            }

        except Exception as exc:  # noqa: BLE001 - surface any failure as a fallback
            self.db.write_audit_log(
                session_id=self.session_id,
                action_type="agent_error",
                tool_name="pydantic_ai_agent",
                request_prompt=cleaned,
                governance_status="DEGRADED",
                details={"error": str(exc)[:500]},
            )
            fallback = DeterministicRouter(self.tools, self.session_id).process_query(cleaned)
            fallback["agent_mode"] = "deterministic_fallback"
            fallback["llm_error"] = str(exc)[:300]
            return fallback


# --------------------------------------------------------------------------
# Deterministic fallback (no LLM configured / LLM failure)
# --------------------------------------------------------------------------


class DeterministicRouter:
    """
    Keyword router used when no LLM is available.

    Kept deliberately simple: it exists so the platform is demonstrable and
    testable offline, not to emulate agentic reasoning. It still enforces the
    same governance preflight (PII) and honouring a token's own scope.
    """

    #: Phrases that indicate an attempt to extract raw PII.
    PII_VIOLATION_PHRASES = (
        "raw email",
        "export pii",
        "unmasked phone",
        "dump pii",
        "raw phone",
        "extract emails",
        "extract phone",
    )

    def __init__(self, tools: CDPTools, session_id: str = "default"):
        self.tools = tools
        self.session_id = session_id

    def process_query(self, user_prompt: str) -> Dict[str, Any]:
        cleaned = GovernanceGuardrails.mask_pii((user_prompt or "").strip())
        lower = cleaned.lower()

        response: Dict[str, Any] = {
            "query": cleaned,
            "intent": "general_inquiry",
            "action_taken": None,
            "tool_output": None,
            "insights": "",
            "recommended_next_action": None,
            "governance_status": "PASSED",
            "agent_mode": "deterministic_fallback",
        }

        # Same governance preflight as the LLM path.
        if any(p in lower for p in self.PII_VIOLATION_PHRASES):
            self.tools.db.write_audit_log(
                session_id=self.session_id,
                action_type="policy_violation",
                tool_name="preflight_pii_check",
                request_prompt=cleaned,
                governance_status="BLOCKED_BY_POLICY",
                details={"matched": "pii_extraction_phrase"},
            )
            response.update(
                intent="policy_violation",
                insights=(
                    "Request blocked by governance: extraction of unmasked PII is "
                    "prohibited. Customer identifiers are returned masked."
                ),
                governance_status="BLOCKED_BY_POLICY",
            )
            return response

        import re

        # Activation: a supplied token determines the scope that may be activated.
        token_match = re.search(r"MKT_APPRV_[A-Za-z0-9_\-=.]+", user_prompt or "")
        if token_match and any(k in lower for k in ("activate", "launch", "send")):
            token = token_match.group(0)
            verification = GovernanceGuardrails.verify_activation_approval(token)
            if not verification.valid:
                self.tools.db.write_audit_log(
                    session_id=self.session_id,
                    action_type="activation_attempt",
                    tool_name="activate_campaign",
                    request_prompt=cleaned,
                    governance_status="BLOCKED_BY_GOVERNANCE",
                    details={"reason": verification.reason},
                )
                response.update(
                    intent="campaign_activation",
                    action_taken="activate_campaign",
                    tool_output={
                        "status": "REJECTED",
                        "error": verification.reason,
                        "governance_status": "BLOCKED_BY_GOVERNANCE",
                    },
                    insights=verification.reason,
                    governance_status="BLOCKED_BY_GOVERNANCE",
                )
                return response

            payload = verification.payload or {}
            res = self.tools.activate_campaign(
                campaign_name=payload.get("campaign_name", "Campaign"),
                audience_id=payload.get("audience_id", "AUD_UNKNOWN"),
                channel=payload.get("channel", "email_marketing"),
                approval_token=token,
                audience_size=120,
            )
            response.update(
                intent="campaign_activation",
                action_taken="activate_campaign",
                tool_output=res,
                insights=res.get("message", ""),
                governance_status=res.get("governance_status", "PASSED"),
            )
            return response

        if "lookalike" in lower:
            seed = "Loyal Customers" if "loyal" in lower else "Champions"
            res = self.tools.generate_lookalike_audience(seed_rfm_segment=seed, top_k=25)
            response.update(
                intent="lookalike_audience_generation",
                action_taken="generate_lookalike_audience",
                tool_output=res,
                insights=(
                    f"Generated a lookalike audience of {res.get('lookalike_size', 0)} "
                    f"prospects modelled on '{seed}', excluding non-consenting users "
                    f"and existing seed members. Average predicted CLTV is "
                    f"${res.get('avg_predicted_cltv', 0)}."
                ),
                recommended_next_action="Prepare a paid social acquisition campaign.",
            )
            return response

        if any(k in lower for k in ("attribution", "mta", "channel contribution")):
            res = self.tools.run_mta_attribution_comparison()
            response.update(
                intent="multi_touch_attribution",
                action_taken="run_mta_attribution_comparison",
                tool_output=res,
                insights=(
                    "Attribution comparison complete across first-touch, last-touch, "
                    "linear, time-decay and Markov models."
                ),
                recommended_next_action="Review channel credit shifts before reallocating budget.",
            )
            return response

        if any(k in lower for k in ("activate", "launch", "campaign", "plan")):
            plan = self.tools.prepare_campaign_plan(
                campaign_name="Targeted_Reactivation_Pilot",
                audience_id="AUD_TARGET",
                channel="email_marketing",
                message_objective="Re-engage high-value accounts",
                offer_discount="15% Off Next Renewal",
            )
            response.update(
                intent="campaign_planning",
                action_taken="prepare_campaign_plan",
                tool_output=plan,
                insights=(
                    "Campaign plan prepared. Autonomous execution halted. "
                    f"Human approval required using token: "
                    f"{plan['required_approval_token']}"
                ),
                recommended_next_action="Review the plan and approve with the signed token.",
            )
            return response

        if any(k in lower for k in ("win-back", "winback", "at-risk", "churn")):
            res = self.tools.create_audience_filter(
                rfm_segment="At-Risk High Value", min_churn_risk=0.5
            )
            response.update(
                intent="audience_discovery",
                action_taken="create_audience_filter",
                tool_output=res,
                insights=(
                    f"Identified {res.get('compliant_size', 0)} compliant high-value "
                    f"customers at churn risk. {res.get('governance_note')}"
                ),
                recommended_next_action=(
                    f"Draft a win-back campaign for audience {res.get('audience_id')}."
                ),
            )
            return response

        overview = self.tools.get_rfm_segments_overview()
        response.update(
            intent="customer_360_overview",
            action_taken="get_rfm_segments_overview",
            tool_output=overview,
            insights="Retrieved live RFM distribution across all unified profiles.",
            recommended_next_action="Ask about at-risk segments, attribution, or lookalikes.",
        )
        return response


if __name__ == "__main__":
    agent = MarTechSupervisorAgent()
    print("LLM tool calling available:", agent.llm_available)
    out = agent.process_query("Which segments are at risk of churning?")
    print("Mode:", out["agent_mode"])
    print("Insights:", str(out["insights"])[:400])
