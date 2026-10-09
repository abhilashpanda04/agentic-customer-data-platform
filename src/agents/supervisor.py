"""
MarTech Supervisor Agent - multi-agent orchestration with Pydantic-AI.

Architecture
------------
A genuine supervisor/worker pattern:

    user prompt
      -> Supervisor agent (Groq LLM)
      -> decides which specialist(s) to delegate to, and in what order
      -> specialists run their own tools against DuckDB; each writes audit rows
      -> results flow through shared AgentDeps/SessionState, so a value built by
         one specialist (audience_id) is available to the next (campaign planner)
      -> supervisor composes a single typed AgentAnswer

Specialists (see specialists.py): audience analyst, customer analytics,
attribution analyst, lookalike, campaign planner, governance & approval.

The supervisor NEVER activates a campaign autonomously: activation requires a
signed human approval token present in the user's request. Governance lives in
the tools (consent filtering, PII masking, token verification), so no chain of
agent calls can bypass it.

If no GROQ_API_KEY is configured, DeterministicRouter provides a stateful
keyword-driven path so the platform stays demonstrable and testable offline.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_ai import RunContext  # noqa: F401 - used in tool annotations (module scope)

from src.agents.governance import GovernanceGuardrails
from src.agents.state import AgentDeps, SessionState
from src.agents.tools.cdp_tools import CDPTools
from src.cdp.database import CDPDatabase


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


SUPERVISOR_SYSTEM_PROMPT = """You are the MarTech Intelligence Copilot for an enterprise Customer Data Platform (CDP).

You are a SUPERVISOR. You do not perform analytics yourself — you delegate to
specialist agents and compose their results into one answer.

Specialists you can delegate to (use the matching tool):
- Audience Analyst: to build a consent-compliant target audience from criteria.
- Customer Analytics: RFM segment overviews or single-customer 360 profiles.
- Attribution Analyst: multi-touch attribution comparison across channels.
- Lookalike: to expand a high-value seed segment into prospects.
- Campaign Planner: to draft a campaign and issue a signed approval token.
- Governance & Approval: to review an activation request against policy.

How to work:
- Choose the specialist(s) the user's request needs. Several requests need
  multiple specialists — delegate in the right order and chain their outputs
  (e.g. build an audience, then plan a campaign for that audience_id).
- Each delegation tool takes a `task` string: restate what the user needs in
  concrete marketing terms (segment, channel, criteria, objective).
- Base every number in your final answer strictly on what specialists returned.
  If a specialist reports an error or empty result, say so plainly.

Hard governance rules (non-negotiable):
- You may PLAN campaigns, but you may NEVER activate one yourself. Activation
  requires a signed approval token beginning with 'MKT_APPRV_' that a human
  marketer supplied in the conversation. If the user asks to launch/activate
  without such a token, delegate to the Campaign Planner instead and explain
  that human approval is required.
- If the user DOES supply a token (you can see it in the conversation), you may
  call the activation endpoint with it — but only after governance review.
- Never reveal raw emails/phones/PII; tools return masked identifiers.
- Audiences are consent-filtered automatically; report suppressed counts.

Be concise, use concrete numbers, and always recommend a next action.
"""


class MarTechSupervisorAgent:
    """Supervisor agent that delegates to specialist agents."""

    def __init__(self, db: Optional[CDPDatabase] = None, session_id: str = "default"):
        self.db = db or CDPDatabase()
        self.session_id = session_id
        self.state = SessionState(session_id=session_id)
        self.tools = CDPTools(self.db, session_id=session_id)
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._agent = None
        self.specialists: Dict[str, Any] = {}
        if self.api_key:
            self._agent = self._build_agent()

    @property
    def llm_available(self) -> bool:
        return self._agent is not None

    # ---------------------------------------------------------- construction --
    def _build_agent(self):
        """Builds the supervisor agent and the specialist agents it delegates to."""
        from pydantic_ai import Agent, RunContext
        from pydantic_ai.models.groq import GroqModel
        from pydantic_ai.providers.groq import GroqProvider

        from src.agents.specialists import build_specialists

        model = GroqModel(
            self.model_name,
            provider=GroqProvider(api_key=self.api_key),
        )
        # Specialists share the same model; each owns a subset of tools.
        self.specialists = build_specialists(model)

        supervisor = Agent(
            model,
            output_type=AgentAnswer,
            deps_type=AgentDeps,
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            retries=2,
        )

        # ---- delegation tools: one per specialist ----
        @supervisor.tool
        async def delegate_to_audience_analyst(
            ctx: RunContext[AgentDeps], task: str
        ) -> Dict[str, Any]:
            """Delegate audience-building to the Audience Analyst specialist.
            Use for requests like 'build a win-back audience of at-risk high-value
            customers' or 'find people who haven't purchased in 90 days'."""
            result = await self.specialists["audience"].run(task, deps=ctx.deps)
            return result.output.model_dump()

        @supervisor.tool
        async def delegate_to_customer_analytics(
            ctx: RunContext[AgentDeps], task: str
        ) -> Dict[str, Any]:
            """Delegate to the Customer Analytics specialist for RFM segment
            overviews, segment sizes/averages, or single-customer profiles."""
            result = await self.specialists["analytics"].run(task, deps=ctx.deps)
            return result.output.model_dump()

        @supervisor.tool
        async def delegate_to_attribution_analyst(
            ctx: RunContext[AgentDeps], task: str
        ) -> Dict[str, Any]:
            """Delegate to the Attribution Analyst specialist to compare first/last/
            linear/time-decay/Markov attribution across channels."""
            result = await self.specialists["attribution"].run(task, deps=ctx.deps)
            return result.output.model_dump()

        @supervisor.tool
        async def delegate_to_lookalike_agent(
            ctx: RunContext[AgentDeps], task: str
        ) -> Dict[str, Any]:
            """Delegate to the Lookalike specialist to expand a high-value seed
            segment into ranked prospects."""
            result = await self.specialists["lookalike"].run(task, deps=ctx.deps)
            return result.output.model_dump()

        @supervisor.tool
        async def delegate_to_campaign_planner(
            ctx: RunContext[AgentDeps], task: str
        ) -> Dict[str, Any]:
            """Delegate to the Campaign Planner specialist to draft a campaign and
            issue a signed human-approval token. Never activates anything."""
            result = await self.specialists["planner"].run(task, deps=ctx.deps)
            return result.output.model_dump()

        @supervisor.tool
        async def delegate_to_governance_review(
            ctx: RunContext[AgentDeps], task: str
        ) -> Dict[str, Any]:
            """Delegate to the Governance & Approval specialist to review an
            activation request (token validity, expiry, scope, audience) and
            return a policy decision."""
            result = await self.specialists["governance"].run(task, deps=ctx.deps)
            return result.output.model_dump()

        # ---- the one governed endpoint the supervisor may call ----
        @supervisor.tool
        def activate_campaign_with_token(
            ctx: RunContext[AgentDeps], approval_token: str
        ) -> Dict[str, Any]:
            """Activate a campaign using a signed human approval token that the
            marketer supplied in the conversation. The token's embedded scope
            (campaign, audience, channel) determines what is activated. Fails if
            the token is missing, forged, expired, reused, or scope-mismatched."""
            verification = GovernanceGuardrails.verify_activation_approval(approval_token)
            if not verification.valid:
                return {
                    "status": "REJECTED",
                    "error": verification.reason,
                    "governance_status": "BLOCKED_BY_GOVERNANCE",
                }
            payload = verification.payload or {}
            out = self.tools.activate_campaign(
                campaign_name=payload.get("campaign_name", "Campaign"),
                audience_id=payload.get("audience_id", "AUD_UNKNOWN"),
                channel=payload.get("channel", "email_marketing"),
                approval_token=approval_token,
                audience_size=120,
            )
            if out.get("status") == "SUCCESS":
                self.state.activated_campaigns.append(out.get("activation_id", ""))
            return out

        return supervisor

    # ------------------------------------------------------------------ run --
    def process_query(self, user_prompt: str) -> Dict[str, Any]:
        """
        Runs the agentic pipeline on a user prompt.

        LLM path: supervisor delegates to specialists (stateful, multi-step).
        Fallback: DeterministicRouter when no LLM is configured.
        """
        cleaned = GovernanceGuardrails.mask_pii((user_prompt or "").strip())
        deps = AgentDeps(
            tools=self.tools,
            state=self.state,
            session_id=self.session_id,
        )

        if not self.llm_available:
            return DeterministicRouter(self.tools, self.session_id, self.state).process_query(cleaned)

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
                "agent_mode": "llm_multi_agent",
            }

        try:
            result = self._agent.run_sync(cleaned, deps=deps)
            answer: AgentAnswer = result.output

            self.db.write_audit_log(
                session_id=self.session_id,
                action_type="agent_response",
                tool_name="supervisor_agent",
                request_prompt=cleaned,
                governance_status="PASSED",
                details={
                    "model": self.model_name,
                    "delegations": self.state.governance_events[-10:],
                    "summary": answer.summary[:500],
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
                "agent_mode": "llm_multi_agent",
            }

        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            self.db.write_audit_log(
                session_id=self.session_id,
                action_type="agent_error",
                tool_name="supervisor_agent",
                request_prompt=cleaned,
                governance_status="DEGRADED",
                details={"error": str(exc)[:500]},
            )
            fallback = DeterministicRouter(self.tools, self.session_id, self.state).process_query(cleaned)
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
    same governance preflight and honours a token's own scope, and it keeps the
    session state current so stateful chaining works offline too.
    """

    PII_VIOLATION_PHRASES = (
        "raw email",
        "export pii",
        "unmasked phone",
        "dump pii",
        "raw phone",
        "extract emails",
        "extract phone",
    )

    def __init__(
        self,
        tools: CDPTools,
        session_id: str = "default",
        state: Optional[SessionState] = None,
    ):
        self.tools = tools
        self.session_id = session_id
        self.state = state or SessionState(session_id=session_id)

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
            if res.get("status") == "SUCCESS":
                self.state.activated_campaigns.append(res.get("activation_id", ""))
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
            self.state.last_lookalike_id = res.get("lookalike_audience_id")
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
                audience_id=self.state.last_audience_id or "AUD_TARGET",
                channel="email_marketing",
                message_objective="Re-engage high-value accounts",
                offer_discount="15% Off Next Renewal",
            )
            self.state.campaign_drafts.append("Targeted_Reactivation_Pilot")
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
            self.state.last_audience_id = res.get("audience_id")
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
    print("LLM multi-agent available:", agent.llm_available)
    print("Specialists registered:", list(agent.specialists.keys()))
    out = agent.process_query("Which segments are at risk of churning?")
    print("Mode:", out["agent_mode"], "| intent:", out["intent"])
    print("Insights:", str(out["insights"])[:400])