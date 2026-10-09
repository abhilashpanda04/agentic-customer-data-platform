"""
Shared session state for the agentic layer.

The supervisor and every specialist agent receive the *same* `AgentDeps` so that
a value produced by one specialist (e.g. an audience_id) is visible to the next.
This is what makes agentic chaining possible:

    AudienceAnalyst builds AUD_8F2A  ->  state.last_audience_id = "AUD_8F2A"
    CampaignPlanner uses it          ->  plan references AUD_8F2A

State is intentionally a small Pydantic model: the durable source of truth stays
in DuckDB (audit log, audiences, activations); this in-memory object carries the
conversation-level thread between agent turns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids circular import
    from src.agents.tools.cdp_tools import CDPTools


class SessionState(BaseModel):
    """Conversation-level state shared across agents in one session."""

    session_id: str = "default"
    last_audience_id: Optional[str] = Field(
        default=None, description="Most recent compliant audience built this session."
    )
    last_lookalike_id: Optional[str] = Field(
        default=None, description="Most recent lookalike audience built this session."
    )
    campaign_drafts: List[str] = Field(
        default_factory=list, description="Campaign drafts prepared this session."
    )
    activated_campaigns: List[str] = Field(
        default_factory=list, description="Activation IDs completed this session."
    )
    governance_events: List[str] = Field(
        default_factory=list, description="Governance decisions observed this session."
    )

    def note_governance(self, event: str) -> None:
        self.governance_events.append(event)


@dataclass
class AgentDeps:
    """Dependencies injected into every agent run (shared across agents)."""

    tools: "CDPTools" = field(repr=False)
    state: SessionState = field(default_factory=SessionState)
    session_id: str = "default"
    tool_calls: List[str] = field(default_factory=list)
    governance_events: List[str] = field(default_factory=list)