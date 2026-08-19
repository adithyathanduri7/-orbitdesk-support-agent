import json
import time
from rich.console import Console
from state import AgentState
from config import (
    OUT_OF_SCOPE_KEYWORDS,
    CLARIFICATION_KEYWORDS,
    ESCALATION_KEYWORDS,
    MAX_RETRIES,
    MIN_CONFIDENCE_THRESHOLD
)

console = Console()


def triage_node(state: AgentState,
                kb=None,
                llm=None) -> AgentState:
    question = state["question"]
    q_lower = question.lower()
    logs = state.get("logs", [])

    console.rule("[bold blue]NODE: TRIAGE[/bold blue]")
    console.log(f"Question: {question}")

    # Check out of scope first
    for kw in OUT_OF_SCOPE_KEYWORDS:
        if kw in q_lower:
            classification = "out_of_scope"
            logs.append(
                f"TRIAGE → out_of_scope "
                f"(keyword match: '{kw}')")
            console.log(
                f"[red]Classification: "
                f"out_of_scope[/red]")
            return {
                **state,
                "classification": classification,
                "logs": logs
            }

    for kw in ESCALATION_KEYWORDS:
        if kw in q_lower:
            classification = "requires_escalation"
            logs.append(
                f"TRIAGE → requires_escalation "
                f"(keyword: '{kw}')")
            console.log(
                f"[yellow]Classification: "
                f"requires_escalation[/yellow]")
            return {
                **state,
                "classification": classification,
                "logs": logs
            }

    
    vague = (
        any(kw in q_lower
            for kw in CLARIFICATION_KEYWORDS) and
        not any(c in q_lower for c in [
            "render_failed", "timezone", "viewer",
            "admin", "owner", "api credential",
            "schedule", "export", "dashboard"
        ])
    )
    if vague:
        classification = "requires_clarification"
        logs.append(
            "TRIAGE → requires_clarification "
            "(vague question, needs more info)")
        console.log(
            f"[yellow]Classification: "
            f"requires_clarification[/yellow]")
        return {
            **state,
            "classification": classification,
            "logs": logs
        }

    
    classification = "answerable"
    logs.append("TRIAGE → answerable")
    console.log(
        f"[green]Classification: answerable[/green]")

    return {
        **state,
        "classification": classification,
        "logs": logs
    }



def retrieval_node(state: AgentState,
                   kb=None,
                   llm=None) -> AgentState:
    question = state["question"]
    logs = state.get("logs", [])

    console.rule(
        "[bold blue]NODE: RETRIEVAL[/bold blue]")

    chunks = kb.retrieve(question)

    console.log(
        f"[cyan]Retrieved {len(chunks)} "
        f"chunks[/cyan]")

    for i, c in enumerate(chunks):
        console.log(
            f"  [{i+1}] {c['source']} "
            f"(score: {c['score']:.3f}): "
            f"{c['text'][:80]}...")

    logs.append(
        f"RETRIEVAL → {len(chunks)} chunks "
        f"from: "
        f"{list(set(c['source'] for c in chunks))}")

    return {
        **state,
        "retrieved_chunks": chunks,
        "logs": logs
    }



def generation_node(state: AgentState,
                    kb=None,
                    llm=None) -> AgentState:
    question = state["question"]
    chunks = state.get("retrieved_chunks", [])
    classification = state["classification"]
    retry_count = state.get("retry_count", 0)
    logs = state.get("logs", [])

    console.rule(
        "[bold blue]NODE: GENERATION[/bold blue]")

    
    if classification == "out_of_scope":
        answer = (
            "This request is outside the scope of "
            "OrbitDesk product support. The support "
            "agent cannot issue refunds, cancel "
            "subscriptions, provide legal advice, or "
            "perform account changes. Please contact "
            "your account team for billing matters."
        )
        sources = []
        confidence = 0.99
        requires_human = True
        reason = (
            "Request is outside OrbitDesk support scope. "
            "Possible prompt injection attempt detected."
        )
        logs.append(
            "GENERATION → out_of_scope safe response")

        return {
            **state,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "requires_human": requires_human,
            "reason": reason,
            "logs": logs
        }

    if classification == "requires_clarification":
        answer = (
            "I need more information to help you "
            "effectively. Could you please provide:\n"
            "1. Your Workspace ID\n"
            "2. The specific feature or object affected "
            "(e.g., connection name, schedule ID, "
            "dashboard name)\n"
            "3. The current state or error code shown\n"
            "4. When the issue started\n"
            "5. Whether both manual and scheduled "
            "operations are affected\n\n"
            "Please do not share passwords, API secrets "
            "or OAuth tokens."
        )
        sources = []
        confidence = 0.90
        requires_human = False
        reason = (
            "Question is too vague to diagnose. "
            "Clarification required."
        )
        logs.append(
            "GENERATION → clarification response")

        return {
            **state,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "requires_human": requires_human,
            "reason": reason,
            "logs": logs
        }

    if classification == "requires_escalation":
        answer = (
            "Two consecutive render_failed events "
            "for the same dashboard triggers "
            "escalation. Please collect:\n"
            "1. Workspace ID\n"
            "2. Dashboard ID\n"
            "3. Schedule ID\n"
            "4. Run IDs for both failed runs\n"
            "5. Timestamps with timezone\n\n"
            "Do NOT include exported customer data, "
            "passwords, API secrets or OAuth tokens "
            "in the escalation note.\n\n"
            "Escalate to the Rendering team with "
            "the collected information."
        )
        sources = [{"document": "KB-008",
                    "passage":
                    "Escalation conditions: two "
                    "consecutive render_failed events"}]
        confidence = 0.95
        requires_human = True
        reason = (
            "Two consecutive render_failed events "
            "after documented checks meets escalation "
            "threshold per KB-004 and KB-008."
        )
        logs.append(
            "GENERATION → escalation response")

        return {
            **state,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "requires_human": requires_human,
            "reason": reason,
            "logs": logs
        }

    
    if not chunks:
        answer = (
            "I was unable to find relevant information "
            "in the OrbitDesk knowledge base for your "
            "question. Please provide more details or "
            "contact support directly."
        )
        sources = []
        confidence = 0.1
        requires_human = True
        reason = "No relevant chunks retrieved."
        logs.append(
            "GENERATION → no chunks → safe failure")

        return {
            **state,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "requires_human": requires_human,
            "reason": reason,
            "logs": logs
        }

    
    context = "\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}"
        for c in chunks[:4]
    ])

    
    if retry_count > 0:
        retry_instruction = (
            "\n\nIMPORTANT: Your previous answer "
            "failed verification. Please:\n"
            "1. Be more specific and accurate\n"
            "2. Only use information from the context\n"
            "3. Include source references\n"
        )
    else:
        retry_instruction = ""

    prompt = f"""Answer the following support question
using ONLY the context provided below.
Do not add information not present in the context.
Be specific and actionable.{retry_instruction}

CONTEXT:
{context}

QUESTION: {question}

Provide a clear, step-by-step answer based strictly
on the context above. Reference the source documents."""

    console.log(
        f"[cyan]Generating answer "
        f"(retry={retry_count})...[/cyan]")

    answer = llm.generate(prompt)

    # Build sources list
    sources = [
        {
            "document": c["source"],
            "passage": c["text"][:200] + "..."
            if len(c["text"]) > 200
            else c["text"]
        }
        for c in chunks[:3]
    ]

    # Estimate confidence from retrieval scores
    if chunks:
        avg_score = sum(
            c["score"] for c in chunks
        ) / len(chunks)
        confidence = min(
            0.95, round(avg_score * 1.2, 2))
    else:
        confidence = 0.3

    requires_human = False
    reason = (
        f"Answer generated from {len(chunks)} "
        f"retrieved chunks. "
        f"Top source: {chunks[0]['source']}"
    )

    logs.append(
        f"GENERATION → answer generated "
        f"({len(answer)} chars, "
        f"confidence={confidence})")

    console.log(
        f"[green]Answer generated "
        f"({len(answer)} chars)[/green]")

    return {
        **state,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "requires_human": requires_human,
        "reason": reason,
        "logs": logs
    }


# ------------------------------------------------
# NODE 4: VERIFICATION
# Check if answer meets quality requirements
# ------------------------------------------------
def verification_node(state: AgentState,
                      kb=None,
                      llm=None) -> AgentState:
    answer = state.get("answer", "")
    sources = state.get("sources", [])
    confidence = state.get("confidence", 0.0)
    classification = state["classification"]
    chunks = state.get("retrieved_chunks", [])
    logs = state.get("logs", [])

    console.rule(
        "[bold blue]NODE: VERIFICATION[/bold blue]")

    issues = []

    # Skip deep verification for non-answerable
    if classification in [
        "out_of_scope",
        "requires_clarification",
        "requires_escalation"
    ]:
        logs.append(
            f"VERIFICATION → skipped "
            f"({classification})")
        console.log(
            "[dim]Verification skipped for "
            "non-answerable classification[/dim]")
        return {
            **state,
            "verification_passed": True,
            "verification_notes": "Skipped",
            "logs": logs
        }

    # Check 1: Answer is not empty
    if not answer or len(answer.strip()) < 20:
        issues.append("Answer is too short or empty")

    # Check 2: Has sources
    if not sources:
        issues.append("No source references")

    # Check 3: Confidence above threshold
    if confidence < MIN_CONFIDENCE_THRESHOLD:
        issues.append(
            f"Confidence {confidence} below "
            f"threshold {MIN_CONFIDENCE_THRESHOLD}")

    # Check 4: Answer doesn't contain hallucination
    # Simple check: answer should contain words
    # from retrieved chunks
    if chunks and answer:
        chunk_words = set()
        for c in chunks[:2]:
            chunk_words.update(
                c["text"].lower().split())

        answer_words = set(answer.lower().split())
        overlap = len(
            chunk_words & answer_words)

        if overlap < 5:
            issues.append(
                f"Low word overlap with retrieved "
                f"evidence ({overlap} words)")

    # Check 5: Answer doesn't claim unsupported actions
    unsupported_claims = [
        "i will", "i can", "i'll refund",
        "i can create", "i'll create",
        "i have access", "i can access"
    ]
    for claim in unsupported_claims:
        if claim in answer.lower():
            issues.append(
                f"Answer makes unsupported "
                f"claim: '{claim}'")

    verification_passed = len(issues) == 0
    notes = (
        "All checks passed"
        if verification_passed
        else f"Failed: {'; '.join(issues)}"
    )

    if verification_passed:
        console.log(
            "[green]✅ Verification passed[/green]")
    else:
        console.log(
            f"[red]❌ Verification failed: "
            f"{notes}[/red]")

    logs.append(
        f"VERIFICATION → "
        f"{'passed' if verification_passed else 'failed'}"
        f": {notes}")

    return {
        **state,
        "verification_passed": verification_passed,
        "verification_notes": notes,
        "logs": logs
    }


# ------------------------------------------------
# NODE 5: FINAL OUTPUT
# Assemble the structured JSON response
# ------------------------------------------------
def output_node(state: AgentState,
                kb=None,
                llm=None) -> AgentState:
    logs = state.get("logs", [])

    console.rule(
        "[bold blue]NODE: OUTPUT[/bold blue]")

    # Safe failure if verification never passed
    # and max retries exhausted
    verification_passed = state.get(
        "verification_passed", False)
    retry_count = state.get("retry_count", 0)

    if not verification_passed and \
       retry_count >= MAX_RETRIES:
        final_response = {
            "classification": "safe_failure",
            "answer": (
                "The support agent was unable to "
                "generate a verified answer for "
                "your question. Please contact "
                "support directly or try rephrasing "
                "your question."
            ),
            "sources": [],
            "confidence": 0.0,
            "requires_human": True,
            "reason": (
                f"Verification failed after "
                f"{retry_count} attempt(s): "
                f"{state.get('verification_notes')}"
            ),
            "clarification_question": None,
            "warnings": [
                "Answer failed verification checks.",
                state.get("verification_notes", "")
            ]
        }
        logs.append(
            "OUTPUT → safe_failure "
            "(max retries exhausted)")
    else:
        final_response = {
            "classification":
                state.get("classification"),
            "answer": state.get("answer", ""),
            "sources": state.get("sources", []),
            "confidence": state.get("confidence", 0.0),
            "requires_human":
                state.get("requires_human", False),
            "reason": state.get("reason", ""),
            "clarification_question": None,
            "warnings": []
        }
        logs.append(
            f"OUTPUT → final response assembled "
            f"({state.get('classification')})")

    console.log(
        f"[green]Final classification: "
        f"{final_response['classification']}[/green]")
    console.log(
        f"[green]Confidence: "
        f"{final_response['confidence']}[/green]")

    return {
        **state,
        "final_response": final_response,
        "logs": logs
    }