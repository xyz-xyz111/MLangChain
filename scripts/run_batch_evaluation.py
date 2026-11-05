#!/usr/bin/env python3
"""Run a reproducible multi-category evaluation of Workflow V2."""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.workflow_v2 import run_workflow_v2


def load_tasks(path: Path, limit: int | None = None) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        tasks = [json.loads(line) for line in handle if line.strip()]
    return tasks[:limit] if limit else tasks


def score_history(state: dict) -> list[float]:
    return [float(item["score"]) for item in state.get("iteration_history", []) if "score" in item]


def record_from_state(task: dict, state: dict, duration: float) -> dict:
    scores = score_history(state)
    final_score = float(state.get("best_score_so_far") or state.get("quality_score") or 0.0)
    return {
        "id": task["id"],
        "type": task["type"],
        "task": task["task"],
        "completed": bool(state.get("is_complete")) and not state.get("error_msg"),
        "error": state.get("error_msg"),
        "initial_score": scores[0] if scores else None,
        "final_score": final_score,
        "score_gain": round(final_score - scores[0], 4) if scores else None,
        "iterations": int(state.get("iteration_count", 0)),
        "duration_seconds": round(duration, 3),
        "reached_threshold": final_score >= 8.5,
        "history": state.get("iteration_history", []),
    }


def aggregate(records: list[dict]) -> dict:
    total = len(records)
    completed = [row for row in records if row["completed"]]
    scored = [row for row in completed if row["initial_score"] is not None]

    def metrics(rows: list[dict]) -> dict:
        scored_rows = [row for row in rows if row["completed"] and row["initial_score"] is not None]
        return {
            "count": len(rows),
            "completion_rate": round(sum(row["completed"] for row in rows) / len(rows), 4) if rows else 0.0,
            "threshold_rate": round(sum(row["reached_threshold"] for row in rows) / len(rows), 4) if rows else 0.0,
            "avg_initial_score": round(statistics.mean(row["initial_score"] for row in scored_rows), 4)
            if scored_rows else None,
            "avg_final_score": round(statistics.mean(row["final_score"] for row in scored_rows), 4)
            if scored_rows else None,
            "avg_score_gain": round(statistics.mean(row["score_gain"] for row in scored_rows), 4)
            if scored_rows else None,
            "avg_iterations": round(statistics.mean(row["iterations"] for row in scored_rows), 4)
            if scored_rows else None,
            "avg_duration_seconds": round(statistics.mean(row["duration_seconds"] for row in rows), 4) if rows else None,
        }

    grouped = defaultdict(list)
    for row in records:
        grouped[row["type"]].append(row)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_tasks": total,
        "completed_tasks": len(completed),
        "scored_tasks": len(scored),
        "failed_tasks": total - len(completed),
        "failure_reasons": [row["error"] for row in records if row.get("error")],
        "overall": metrics(records),
        "by_type": {name: metrics(rows) for name, rows in sorted(grouped.items())},
    }


def markdown(summary: dict) -> str:
    overall = summary["overall"]
    lines = [
        "# MLangChain batch evaluation",
        "",
        f"- Tasks: {summary['total_tasks']}",
        f"- Completion rate: {overall['completion_rate']:.1%}",
        f"- Threshold rate (score >= 8.5): {overall['threshold_rate']:.1%}",
        f"- Average score: {overall['avg_initial_score']} -> {overall['avg_final_score']}",
        f"- Average gain: {overall['avg_score_gain']}",
        f"- Average iterations: {overall['avg_iterations']}",
        f"- Average duration: {overall['avg_duration_seconds']} s",
        "",
        "| Type | N | Completion | Threshold | Initial | Final | Gain | Iterations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary["by_type"].items():
        lines.append(
            f"| {name} | {item['count']} | {item['completion_rate']:.1%} | {item['threshold_rate']:.1%} | "
            f"{item['avg_initial_score']} | {item['avg_final_score']} | {item['avg_score_gain']} | {item['avg_iterations']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path("evaluation/tasks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--n-versions", type=int, default=2, choices=(2, 3))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "batch_records.jsonl"
    tasks = load_tasks(args.tasks, args.limit)
    records = []
    with records_path.open("w", encoding="utf-8") as output:
        for index, task in enumerate(tasks, 1):
            print(f"[{index}/{len(tasks)}] {task['id']}: {task['task']}")
            started = time.perf_counter()
            try:
                state = run_workflow_v2(task["task"], use_best_of_n=True, n_versions=args.n_versions, verbose=False)
                row = record_from_state(task, state, time.perf_counter() - started)
            except Exception as exc:
                row = {
                    "id": task["id"], "type": task["type"], "task": task["task"], "completed": False,
                    "error": f"{type(exc).__name__}: {exc}", "initial_score": None, "final_score": 0.0,
                    "score_gain": None, "iterations": 0, "duration_seconds": round(time.perf_counter() - started, 3),
                    "reached_threshold": False, "history": [],
                }
            records.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()

    summary = aggregate(records)
    (args.output_dir / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "batch_summary.md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))


if __name__ == "__main__":
    main()
