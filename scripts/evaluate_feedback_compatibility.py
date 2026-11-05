#!/usr/bin/env python3
"""Offline ablation for Critic feedback schema compatibility."""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from pydantic import BaseModel, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.state import CritiqueResult


class StrictCritiqueResult(BaseModel):
    """Schema used before structured feedback normalization was added."""

    score: float
    feedback: str
    issues: List[str]


def load_payloads(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(payloads: list[dict]) -> dict:
    rows = []
    for payload in payloads:
        values = {"score": 8.0, "feedback": payload.get("feedback"), "issues": payload.get("issues")}
        strict_ok = True
        normalized_ok = True
        try:
            StrictCritiqueResult(**values)
        except ValidationError:
            strict_ok = False
        try:
            CritiqueResult(**values)
        except ValidationError:
            normalized_ok = False
        rows.append({"id": payload["id"], "strict_ok": strict_ok, "normalized_ok": normalized_ok})

    total = len(rows)
    strict_success = sum(row["strict_ok"] for row in rows)
    normalized_success = sum(row["normalized_ok"] for row in rows)
    return {
        "total_payloads": total,
        "strict_success": strict_success,
        "strict_failure_rate": round((total - strict_success) / total, 4) if total else 0.0,
        "normalized_success": normalized_success,
        "normalized_failure_rate": round((total - normalized_success) / total, 4) if total else 0.0,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payloads", type=Path, default=Path("evaluation/feedback_payloads.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/feedback_compatibility.json"))
    args = parser.parse_args()

    summary = evaluate(load_payloads(args.payloads))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
