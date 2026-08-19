# ================================================
# SHARED TYPED STATE
# LangGraph requires typed state passed between nodes
# ================================================

from typing import TypedDict, Optional, List
from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    text: str
    score: float
    source: str
    type: str


class AgentState(TypedDict):
    # Input
    question: str

    # Triage
    classification: str  # answerable | requires_clarification
                         # | requires_escalation | out_of_scope

    # Retrieval
    retrieved_chunks: List[dict]

    # Generation
    answer: str
    confidence: float
    requires_human: bool
    reason: str
    sources: List[dict]

    # Verification
    verification_passed: bool
    verification_notes: str

    # Control
    retry_count: int
    final_response: Optional[dict]
    logs: List[str]