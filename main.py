
import json
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from knowledge_base import KnowledgeBase
from llm_engine import LocalLLM
from graph import build_graph
from state import AgentState
from config import LLM_MODEL, EMBEDDING_MODEL

console = Console()


def run_question(graph, question: str) -> dict:
    """Run a single question through the graph"""
    console.print(Panel(
        f"[bold]{question}[/bold]",
        title="Question",
        border_style="blue"
    ))

    initial_state: AgentState = {
        "question": question,
        "classification": "",
        "retrieved_chunks": [],
        "answer": "",
        "confidence": 0.0,
        "requires_human": False,
        "reason": "",
        "sources": [],
        "verification_passed": False,
        "verification_notes": "",
        "retry_count": 0,
        "final_response": None,
        "logs": []
    }

    t0 = time.time()
    result = graph.invoke(initial_state)
    elapsed = round(time.time() - t0, 2)

    final = result.get("final_response", {})

    console.print("\n[bold cyan]Execution Log:[/bold cyan]")
    for log in result.get("logs", []):
        console.print(f"  [dim]→ {log}[/dim]")

    console.print("\n[bold green]Response:[/bold green]")
    console.print(json.dumps(final, indent=2))

    console.print(
        f"\n[dim]Total time: {elapsed}s[/dim]"
    )

    return final


def run_all_samples(graph):
    """Run all sample questions"""

    with open(r"..\AI Engineer Internship - Assignment material\sample_questions.json", "r") as f:
        data = json.load(f)

    questions = data["questions"]
    results = []

    console.print(Panel(
        "[bold]Running All Sample Questions[/bold]",
        border_style="green"
    ))

    for q in questions:
        console.rule(f"[bold]{q['question_id']}[/bold]")

        result = run_question(
            graph,
            q["question"]
        )

        results.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "response": result
        })

        console.print()

    table = Table(title="Results Summary")
    table.add_column("ID")
    table.add_column("Classification")
    table.add_column("Confidence")
    table.add_column("Human?")

    for r in results:
        resp = r["response"]

        table.add_row(
            r["question_id"],
            str(resp.get("classification", "")),
            str(resp.get("confidence", "")),
            "Yes" if resp.get("requires_human") else "No"
        )

    console.print(table)

    with open("sample_outputs.json", "w") as f:
        json.dump(results, f, indent=2)

    console.print(
        "[green]✅ Saved sample_outputs.json[/green]"
    )


def interactive_mode(graph):
    """Interactive CLI"""

    console.print(Panel(
        "[bold]OrbitDesk Support Agent[/bold]\n"
        "Type your question.\n"
        "Type 'quit' to exit.",
        border_style="blue"
    ))

    while True:

        question = console.input(
            "\n[bold blue]Question:[/bold blue] "
        ).strip()

        if question.lower() in (
            "quit",
            "exit",
            "q"
        ):
            break

        if not question:
            continue

        run_question(graph, question)


if __name__ == "__main__":

    import sys

    console.print(Panel(
        "[bold cyan]OrbitDesk Support Agent[/bold cyan]\n"
        "Tantrabodh AI Assignment\n\n"
        f"Embedding: {EMBEDDING_MODEL}\n"
        f"LLM: {LLM_MODEL}",
        border_style="cyan"
    ))

    console.log("[cyan]Initializing...[/cyan]")

    kb = KnowledgeBase()
    llm = LocalLLM()

    graph = build_graph(kb, llm)

    console.log("[green]✅ Agent Ready[/green]")

    if len(sys.argv) > 1:

        if sys.argv[1] == "--samples":
            run_all_samples(graph)

        elif sys.argv[1] == "--interactive":
            interactive_mode(graph)

        else:
            question = " ".join(sys.argv[1:])
            run_question(graph, question)

    else:
        run_all_samples(graph)
        interactive_mode(graph)
