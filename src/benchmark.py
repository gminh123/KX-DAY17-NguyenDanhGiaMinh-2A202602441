from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover
    tabulate = None


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read the benchmark JSON from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"Expected a list of conversations in {path}")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    """Score the recall on a three-level scale: 0 / 0.5 / 1."""

    if not expected:
        return 0.0

    answer_lower = (answer or "").lower()
    hits = 0
    for item in expected:
        target = str(item).lower().strip()
        if target and target in answer_lower:
            hits += 1

    if hits == 0:
        return 0.0
    if hits == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """A lightweight quality score usable when no judge model is available."""

    coverage = recall_points(answer, expected)
    text = (answer or "").strip()
    if not text:
        return 0.0

    lower = text.lower()
    structure_bonus = 0.2 if any(marker in lower for marker in ["bullet", "- ", "1.", "2.", "3.", "điểm", "ngắn"] ) else 0.0
    specificity_bonus = 0.15 if len(text) > 20 else 0.0
    completeness_bonus = 0.15 if coverage > 0 else 0.0
    return min(1.0, coverage * 0.7 + structure_bonus + specificity_bonus + completeness_bonus)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Run the full benchmark over a dataset and report aggregate metrics."""

    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_total = 0.0
    quality_total = 0.0
    memory_growth = 0
    compactions = 0

    for conversation in conversations:
        user_id = str(conversation.get("user_id", "bench-user"))
        conv_id = str(conversation.get("id", "bench-conv"))
        turns = conversation.get("turns", [])
        recall_questions = conversation.get("recall_questions", [])

        train_thread = f"{conv_id}-train"
        for turn in turns:
            agent.reply(user_id, train_thread, str(turn))

        total_agent_tokens += int(agent.token_usage(train_thread))
        total_prompt_tokens += int(agent.prompt_token_usage(train_thread))

        for idx, question in enumerate(recall_questions):
            fresh_thread = f"{conv_id}-recall-{idx}"
            response = agent.reply(user_id, fresh_thread, str(question.get("question", "")))
            answer = str(response.get("response", response.get("answer", "")))
            expected = list(question.get("expected_contains", []))
            recall_total += recall_points(answer, expected)
            quality_total += heuristic_quality(answer, expected)

        if hasattr(agent, "memory_file_size"):
            memory_growth += int(agent.memory_file_size(user_id))
        if hasattr(agent, "compaction_count"):
            compactions += int(agent.compaction_count(train_thread))

    num_conversations = max(1, len(conversations))
    num_questions = max(1, sum(len(c.get("recall_questions", [])) for c in conversations))

    row = BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=int(total_agent_tokens // num_conversations),
        prompt_tokens_processed=int(total_prompt_tokens // num_conversations),
        recall_score=round(recall_total / num_questions, 3),
        response_quality=round(quality_total / num_questions, 3),
        memory_growth_bytes=int(memory_growth),
        compactions=int(compactions),
    )
    return row


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Return a formatted table for display in the terminal."""

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table = [[
        row.agent_name,
        row.agent_tokens_only,
        row.prompt_tokens_processed,
        row.recall_score,
        row.response_quality,
        row.memory_growth_bytes,
        row.compactions,
    ] for row in rows]
    if tabulate is not None:
        return tabulate(table, headers=headers, tablefmt="github")

    lines = [" | ".join(headers)]
    lines.append(" | ".join(["---"] * len(headers)))
    for row in table:
        lines.append(" | ".join(str(value) for value in row))
    return "\n".join(lines)


def main() -> None:
    """Run both benchmark suites and print their comparison tables."""

    root = Path(__file__).resolve().parent.parent
    config = load_config(root)

    standard_conversations = load_conversations(root / "data" / "conversations.json")
    long_conversations = load_conversations(root / "data" / "advanced_long_context.json")

    baseline_standard = BaselineAgent(config=config, force_offline=True)
    advanced_standard = AdvancedAgent(config=config, force_offline=True)

    standard_rows = [
        run_agent_benchmark("Baseline", baseline_standard, standard_conversations, config),
        run_agent_benchmark("Advanced", advanced_standard, standard_conversations, config),
    ]

    baseline_long = BaselineAgent(config=config, force_offline=True)
    advanced_long = AdvancedAgent(config=config, force_offline=True)

    long_rows = [
        run_agent_benchmark("Baseline", baseline_long, long_conversations, config),
        run_agent_benchmark("Advanced", advanced_long, long_conversations, config),
    ]

    print("Standard Benchmark")
    print(format_rows(standard_rows))
    print()
    print("Long-Context Stress Benchmark")
    print(format_rows(long_rows))


if __name__ == "__main__":
    main()
