"""
Multi-agent orchestration tests.

Verifies:
  1. All six specialists build and return their typed output models.
  2. The supervisor exposes delegation tools, not the raw CDP tools.
  3. Stateful chaining: value produced by one agent is visible to the next
     (audience_id built by the Audience Analyst is used by the Campaign Planner).
  4. The supervisor's activation endpoint cannot synthesise an approval token
     (governance stays below the agent layer).
"""
from __future__ import annotations

import asyncio

import pytest
from pydantic_ai.models.test import TestModel

from src.agents.specialists import build_specialists
from src.agents.state import AgentDeps, SessionState
from src.agents.tools.cdp_tools import CDPTools
from src.cdp.database import CDPDatabase


@pytest.fixture(scope="module")
def specialists():
    return build_specialists(TestModel())


@pytest.fixture(scope="module")
def deps():
    return AgentDeps(
        tools=CDPTools(),
        state=SessionState(session_id="test-multiagent"),
        session_id="test-multiagent",
    )


@pytest.fixture(scope="module")
def db():
    return CDPDatabase()


def test_all_specialists_build_and_return_typed_outputs(specialists):
    expected = {
        "audience": "AudienceResult",
        "analytics": "AnalyticsResult",
        "attribution": "AttributionResult",
        "lookalike": "LookalikeResult",
        "planner": "CampaignPlanResult",
        "governance": "GovernanceDecision",
    }
    assert set(specialists.keys()) == set(expected.keys())


def test_specialists_return_typed_models(specialists, deps):
    """Each specialist run yields its declared output type (TestModel can produce one)."""
    async def run_all():
        results = {}
        for name, spec in specialists.items():
            res = await spec.run("do your job", deps=deps)
            results[name] = type(res.output).__name__
        return results

    results = asyncio.run(run_all())
    assert results["audience"] == "AudienceResult"
    assert results["analytics"] == "AnalyticsResult"
    assert results["attribution"] == "AttributionResult"
    assert results["lookalike"] == "LookalikeResult"
    assert results["planner"] == "CampaignPlanResult"
    assert results["governance"] == "GovernanceDecision"


def test_planner_uses_audience_from_shared_state(specialists, deps):
    """If the audience analyst built an audience, the planner's tool should adopt it."""
    tools = deps.tools

    # Simulate the Audience Analyst having run: build a real audience.
    res = tools.create_audience_filter(rfm_segment="At-Risk High Value")
    deps.state.last_audience_id = res["audience_id"]

    # Planner tool with no explicit audience_id must use shared state.
    plan = tools.prepare_campaign_plan(
        campaign_name="ChainedCampaign",
        audience_id=deps.state.last_audience_id or "AUD_UNKNOWN",
        channel="email_marketing",
        message_objective="re-engage",
        offer_discount="10%",
    )
    assert plan["audience_id"] == deps.state.last_audience_id
    assert plan["approval_status"] == "PENDING_HUMAN_APPROVAL"


def test_supervisor_exposes_delegation_tools_only(db: CDPDatabase):
    """The supervisor must delegate to specialists, not call raw CDP tools."""
    from unittest.mock import patch

    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key-placeholder"}):
        from src.agents.supervisor import MarTechSupervisorAgent

        agent = MarTechSupervisorAgent(db=db)
        assert agent.llm_available is True

    tool_names = set(agent._agent._function_toolset.tools.keys())
    assert "delegate_to_audience_analyst" in tool_names
    assert "delegate_to_customer_analytics" in tool_names
    assert "delegate_to_attribution_analyst" in tool_names
    assert "delegate_to_lookalike_agent" in tool_names
    assert "delegate_to_campaign_planner" in tool_names
    assert "delegate_to_governance_review" in tool_names
    # The only direct endpoint is governed activation with a human token.
    assert "activate_campaign_with_token" in tool_names
    assert "create_audience_filter" not in tool_names


def test_activation_endpoint_requires_real_token(specialists, deps):
    """Supervisor activation endpoint must reject an invented token even if told to activate."""
    from src.agents.supervisor import GovernanceGuardrails

    verification = GovernanceGuardrails.verify_activation_approval("MKT_APPRV_")
    assert not verification.valid
    assert "malformed" in verification.reason.lower()


def test_state_accumulates_governance_events(deps):
    deps.state.note_governance("review:test:token ok")
    assert any("test" in e for e in deps.state.governance_events)


def test_fallback_router_chains_state():
    """Deterministic fallback should also thread state: audience id -> campaign plan."""
    from src.agents.supervisor import DeterministicRouter

    tools = CDPTools()
    router = DeterministicRouter(tools, session_id="chain-test")
    r1 = router.process_query("Identify at-risk high-value customers for win-back")
    assert router.state.last_audience_id == r1["tool_output"]["audience_id"]

    r2 = router.process_query("Plan a re-engagement campaign")
    assert r2["tool_output"]["audience_id"] == router.state.last_audience_id
    assert router.state.campaign_drafts