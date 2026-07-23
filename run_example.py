#!/usr/bin/env python3
"""Run the diagnosis pipeline on one JSON object or a list of objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag_agent import create_pipeline


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return data
    raise ValueError("Input must be a JSON object or a list of JSON objects.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/results.json"))
    parser.add_argument("--disable-redaction", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = create_pipeline(
        args.data_dir,
        redact_reports=not args.disable_redaction,
    )

    results: list[dict[str, Any]] = []
    for position, case in enumerate(load_cases(args.cases), start=1):
        case_id = str(case.get("case_id") or f"case-{position:04d}")
        report = str(case.get("report") or "")
        print(f"Processing {case_id}...")
        try:
            result = pipeline.run(report).to_dict()
            result["case_id"] = case_id
            results.append(result)
        except Exception as exc:
            results.append(
                {
                    "case_id": case_id,
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} result(s) to {args.output}")


if __name__ == "__main__":
    main()
