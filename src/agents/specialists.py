"""
Specialist marketing agents for the agentic CDP copilot.

Each specialist is a Pydantic-AI agent with:
  - its own system prompt and purpose,
  - a *subset* of the CDP tools (it owns them; the supervisor does not),
  - a typed output model the supervisor can consume,
  - access to the shared `AgentDeps` (session state) so values chain between agents.

Specialists are deliberately narrow. Reasoning about *which* specialist to use,
in *what order*, is the supervisor's job (see supervisor.py). This is the
classic supervisor/worker pattern used by enterprise agent hubs: each specialist
owns one domain (audience, analytics, attribution, lookalikes, campaign
planning, governance) and operates on the same unified data foundation.

Important Pydantic-AI constraint: because the supervisor *delegates* by calling
`await specialist.run(...)` inside an async tool, these agents must never be run
with `run_sync` from inside a synchronous tool (that deadlocks).
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from src.agents.governance import GovernanceGuardrails
from src.agents.state import AgentDeps

# --------------------------------------------------------------------------
# Typed outputs (what each specialist returns to the supervisor)
# --------------------------------------------------------------------------


class AudienceResult(BaseModel):
    """Result of building a target audience."""

    audience_id: str = Field(description="The generated compliant audience id.")
    raw_size: int = Field(description="Profiles matching criteria before consent filtering.")
    compliant_size: int = Field(description="Profiles that pass consent/suppression rules.")
    is_compliant: bool = Field(description="Whether governance validation passed.")
    governance_note: str = Field(description="Suppression/consent explanation.")
    summary: str = Field(description="Human-readable summary of the audience built.")


class AnalyticsResult(BaseModel):
    """Result of a customer analytics query."""

    segments: List[Dict[str, Any]] = Field(default_factory=list, description="RFM segment stats.")
    profile: Optional[Dict[str, Any]] = Field(default=None, description="Single profile if requested.")
    summary: str = Field(description="Human-readable analytics summary.")


class AttributionResult(BaseModel):
    """Result of a multi-touch attribution comparison."""

    channels: List[Dict[str, Any]] = Field(default_factory=list, description="Per-channel credit.")
    reconciliation: Dict[str, Any] = Field(
        default_factory=dict, description="Attribution reconciliation to real revenue."
    )
    summary: str = Field(description="Human-readable attribution summary.")


class LookalikeResult(BaseModel):
    """Result of generating a lookalike audience."""

    lookalike_audience_id: str = Field(description="The generated lookalike audience id.")
    lookalike_size: int = Field(description="Number of ranked prospects.")
    avg_predicted_cltv: float = Field(description="Average predicted CLTV of the lookalikes.")
    summary: str = Field(description="Human-readable lookalike summary.")


class CampaignPlanResult(BaseModel):
    """Result of campaign planning (never activation)."""

    campaign_name: str
    audience_id: str
    channel: str
    approval_status: Literal["PENDING_HUMAN_APPROVAL"] = "PENDING_HUMAN_APPROVAL"
    approval_token: str = Field(description="Signed token a human must supply to activate.")
    summary: str = Field(description="Human-readable campaign plan summary.")


class GovernanceDecision(BaseModel):
    """Decision by the governance & approval agent."""

    decision: Literal["APPROVED", "REJECTED", "REQUIRES_HUMAN_APPROVAL"]
    reason: str = Field(description="Why this decision was reached.")
    checks: List[str] = Field(default_factory=list, description="Policy checks that ran.")


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def build_specialists(model) -> Dict[str, Agent]:
    """
    Builds and returns all specialist agents bound to `model`.

    Each specialist reads `ctx.deps.tools` for the CDP toolset and
    `ctx.deps.state` for shared session state.
    """
    specialists: Dict[str, Agent] = {}

    # ------------------------------------------------ audience analyst ------
    audience = Agent(
        model,
        output_type=AudienceResult,
        deps_type=AgentDeps,
        system_prompt=(
            "You are the Audience Analyst specialist for an enterprise CDP. "
            "Your only job is to translate a marketer's audience request into "
            "concrete filter criteria and call create_audience once. "
            "Use the returned audience_id when composing your summary. "
            "Report the suppressed (non-consenting) count in your summary."
        ),
    )

    @audience.tool
    def create_audience(
        ctx: RunContext[AgentDeps],
        rfm_segment: Optional[str] = None,
        min_cltv: Optional[float] = None,
        min_frequency: Optional[int] = None,
        min_recency_days: Optional[int] = None,
        max_recency_days: Optional[int] = None,
        min_churn_risk: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build a consent-compliant target audience from behavioral criteria.
        Returns an audience_id that can be reused by other agents."""
        out = ctx.deps.tools.create_audience_filter(
            rfm_segment=rfm_segment,
            min_cltv=min_cltv,
            min_frequency=min_frequency,
            min_recency_days=min_recency_days,
            max_recency_days=max_recency_days,
            min_churn_risk=min_churn_risk,
        )
        ctx.deps.state.last_audience_id = out.get("audience_id")
        return out

    specialists["audience"] = audience

    # -------------------------------------------- customer analytics --------
    analytics = Agent(
        model,
        output_type=AnalyticsResult,
        deps_type=AgentDeps,
        system_prompt=(
            "You are the Customer Analytics specialist. You answer questions about "
            "the customer base: RFM segment distribution, segment sizes and averages, "
            "or a single customer's 360 profile. Call the tools you need, then compose "
            "a concise numeric summary."
        ),
    )

    @analytics.tool
    def get_rfm_segments_overview(ctx: RunContext[AgentDeps]) -> Dict[str, Any]:
        """Get RFM segment distribution with average recency, orders, monetary value and predicted CLTV."""
        return {"segments": ctx.deps.tools.get_rfm_segments_overview()}

    @analytics.tool
    def query_customer_profile(ctx: RunContext[AgentDeps], ucid: str) -> Dict[str, Any]:
        """Look up a single customer's unified 360 profile (PII is masked)."""
        return ctx.deps.tools.query_customer_profile(ucid)

    specialists["analytics"] = analytics

    # ------------------------------------------- attribution analyst --------
    attribution = Agent(
        model,
        output_type=AttributionResult,
        deps_type=AgentDeps,
        system_prompt=(
            "You are the Attribution Analyst specialist. You compare first-touch, "
            "last-touch, linear, time-decay and Markov attribution by channel. "
            "Report which channels drive awareness vs conversion and the "
            "reconciliation check."
        ),
    )

    @attribution.tool
    def run_mta_attribution(
        ctx: RunContext[AgentDeps],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare attribution models across channels for the touchpoint data."""
        return ctx.deps.tools.run_mta_attribution_comparison(
            start_date=start_date, end_date=end_date
        )

    specialists["attribution"] = attribution

    # -------------------------------------------------- lookalike agent ------
    lookalike = Agent(
        model,
        output_type=LookalikeResult,
        deps_type=AgentDeps,
        system_prompt=(
            "You are the Lookalike specialist. You expand a high-value seed segment "
            "into a ranked prospect audience using generate_lookalike_audience, "
            "respecting consent and exclusion rules."
        ),
    )

    @lookalike.tool
    def generate_lookalike_audience(
        ctx: RunContext[AgentDeps],
        seed_rfm_segment: str = "Champions",
        top_k: int = 50,
    ) -> Dict[str, Any]:
        """Generate a lookalike prospect audience from a seed RFM segment."""
        out = ctx.deps.tools.generate_lookalike_audience(
            seed_rfm_segment=seed_rfm_segment, top_k=top_k
        )
        ctx.deps.state.last_lookalike_id = out.get("lookalike_audience_id")
        return out

    specialists["lookalike"] = lookalike

    # ------------------------------------------- campaign planner -----------
    planner = Agent(
        model,
        output_type=CampaignPlanResult,
        deps_type=AgentDeps,
        system_prompt=(
            "You are the Campaign Planner specialist. You prepare a campaign draft: "
            "target audience, channel, message objective, offer, frequency cap and "
            "success metric. You NEVER activate a campaign — you issue a signed "
            "approval token that a human must use to activate. If an audience_id "
            "is available in context, use it."
        ),
    )

    @planner.tool
    def prepare_campaign_plan(
        ctx: RunContext[AgentDeps],
        campaign_name: str,
        channel: str,
        message_objective: str,
        audience_id: Optional[str] = None,
        offer_discount: str = "No discount",
        frequency_cap: int = 2,
    ) -> Dict[str, Any]:
        """Prepare a campaign plan and issue a signed human-approval token.
        Use the audience_id from context if you have one."""
        if not audience_id and ctx.deps.state.last_audience_id:
            audience_id = ctx.deps.state.last_audience_id
        out = ctx.deps.tools.prepare_campaign_plan(
            campaign_name=campaign_name,
            audience_id=audience_id or "AUD_UNKNOWN",
            channel=channel,
            message_objective=message_objective,
            offer_discount=offer_discount,
            frequency_cap=frequency_cap,
        )
        ctx.deps.state.campaign_drafts.append(campaign_name)
        return out

    specialists["planner"] = planner

    # ------------------------------------------ governance & approval -------
    governance = Agent(
        model,
        output_type=GovernanceDecision,
        deps_type=AgentDeps,
        system_prompt=(
            "You are the Governance & Approval specialist. You review campaign "
            "activation requests against policy. If no valid signed human approval "
            "token is supplied, you MUST return REQUIRES_HUMAN_APPROVAL. You never "
            "approve without evidence: call review_activation with the token and "
            "campaign details and base your decision strictly on the result."
        ),
    )

    @governance.tool
    def review_activation(
        ctx: RunContext[AgentDeps],
        campaign_name: str,
        audience_id: str,
        channel: str,
        approval_token: str,
    ) -> Dict[str, Any]:
        """Verify a human approval token (signature, expiry, scope, single-use)
        and confirm the target audience exists."""
        verification = GovernanceGuardrails.verify_activation_approval(
            approval_token,
            campaign_name=campaign_name,
            audience_id=audience_id,
            channel=channel,
        )
        result = {
            "valid": verification.valid,
            "reason": verification.reason,
            "campaign_name": campaign_name,
            "audience_id": audience_id,
            "channel": channel,
        }
        ctx.deps.state.note_governance(f"review:{campaign_name}:{verification.reason}")
        return result

    specialists["governance"] = governance

    return specialists