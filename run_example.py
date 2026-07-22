"""Run one de-identified case through the full SAGE-Path workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from sage_path import SUPPORTED_TASKS, create_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        default="examples/demo_case.json",
        help="JSON file containing report, task, and optional fields.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the locally prepared knowledge-base files.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path. Raw reports and guideline text are never saved.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    case_path = Path(args.case)
    case = json.loads(case_path.read_text(encoding="utf-8"))

    task = case.get("task", "rare_tumor")
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}")
    report = case.get("report", "")
    fields = case.get("fields", {})

    pipeline = create_pipeline(args.data_dir)
    result = pipeline.run(report, task=task, fields=fields)
    public_result = result.to_public_dict()

    print(result.answer)
    print("\nReflection decisions:")
    for record in result.reflection_history:
        print(f"- round {record.iteration}: {record.evaluation}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(public_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved privacy-minimized output to: {output_path}")


if __name__ == "__main__":
    main()
