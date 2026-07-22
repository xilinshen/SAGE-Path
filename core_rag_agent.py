#!/usr/bin/env python3
"""Privacy-conscious pathology RAG agent.

This public version keeps the original hybrid retrieval and self-review workflow,
while removing secrets, personal paths, notebook outputs, and raw case logging.

Important: reports are sent to a third-party API after best-effort redaction.
Review the redacted text and your organization's data policy before real use.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import requests


CHAT_URL = "https://api.siliconflow.cn/v1/chat/completions"
EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"


def _load_local_env(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs without adding a runtime dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class APIError(RuntimeError):
    """Raised when the remote model API cannot return a valid response."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    chat_model: str = "deepseek-ai/DeepSeek-R1"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    connect_timeout: int = 15
    read_timeout: int = 120
    max_retries: int = 5
    max_rounds: int = 5

    @classmethod
    def from_env(cls) -> "Settings":
        _load_local_env()
        api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "Missing SILICONFLOW_API_KEY. Copy .env.example to .env and set your key."
            )
        return cls(
            api_key=api_key,
            chat_model=os.getenv("CHAT_MODEL", cls.chat_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            max_rounds=int(os.getenv("MAX_ROUNDS", cls.max_rounds)),
        )


class PrivacyFilter:
    """Best-effort removal of common direct identifiers.

    This is not a formal anonymization guarantee. Always inspect source data and
    the redacted result before sending clinical text to an external service.
    """

    FIELD_PATTERNS = (
        r"(?i)(濮撳悕|鎮ｈ€呭鍚峾name|patient\s*name)\s*[:锛歖\s*[^,锛�;锛沑n]+",
        r"(?i)(韬唤璇佸彿?|璇佷欢鍙穦id\s*(?:number|no\.?))\s*[:锛歖\s*[A-Za-z0-9-]+",
        r"(?i)(浣忛櫌鍙穦闂ㄨ瘖鍙穦鐥呯悊鍙穦妫€鏌ュ彿|medical\s*record\s*(?:number|no\.?))\s*[:锛歖\s*[A-Za-z0-9-]+",
        r"(?i)(鐢佃瘽|鎵嬫満鍙穦鑱旂郴鏂瑰紡|phone|mobile)\s*[:锛歖\s*[+()0-9 \t-]{7,}",
        r"(?i)(鍦板潃|浣忓潃|address)\s*[:锛歖\s*[^,锛�;锛沑n]+",
        r"(?i)(鍑虹敓鏃ユ湡|鐢熸棩|date\s*of\s*birth|dob)\s*[:锛歖\s*[^,锛�;锛沑n]+",
    )
    EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    PHONE_PATTERN = r"(?<!\d)(?:\+?\d[\d\s()-]{8,}\d)(?!\d)"
    CHINESE_ID_PATTERN = r"(?<!\d)\d{17}[\dXx](?!\d)"

    @classmethod
    def redact(cls, text: str) -> str:
        redacted = text
        for pattern in cls.FIELD_PATTERNS:
            redacted = re.sub(pattern, "[REDACTED_FIELD]", redacted)
        redacted = re.sub(cls.EMAIL_PATTERN, "[REDACTED_EMAIL]", redacted)
        redacted = re.sub(cls.CHINESE_ID_PATTERN, "[REDACTED_ID]", redacted)
        redacted = re.sub(cls.PHONE_PATTERN, "[REDACTED_PHONE]", redacted)
        return redacted.strip()


class SiliconFlowClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        }

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries):
            try:
                response = self.session.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=(
                        self.settings.connect_timeout,
                        self.settings.read_timeout,
                    ),
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise APIError(f"Temporary API error: HTTP {response.status_code}")
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError, APIError) as exc:
                last_error = exc
                if attempt + 1 == self.settings.max_retries:
                    break
                delay = min(20.0, 2**attempt) * (0.7 + 0.6 * random.random())
                time.sleep(delay)
        raise APIError("Remote API request failed after retries") from last_error

    def chat(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        payload = {
            "model": self.settings.chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "n": 1,
        }
        data = self._post_json(CHAT_URL, payload)
        choices = data.get("choices") or []
        if not choices:
            raise APIError("Chat API returned no choices")
        content = (choices[0].get("message") or {}).get("content")
        if not content:
            raise APIError("Chat API returned empty content")
        return str(content).strip()

    def embed(self, text: str) -> np.ndarray:
        payload = {
            "input": text,
            "encoding_format": "float",
            "model": self.settings.embedding_model,
        }
        data = self._post_json(EMBEDDING_URL, payload)
        items = data.get("data") or []
        if not items or "embedding" not in items[0]:
            raise APIError("Embedding API returned an invalid payload")
        return np.asarray(items[0]["embedding"], dtype=np.float32).reshape(1, -1)


@dataclass
class RAGResources:
    chunks: list[dict[str, Any]]
    bm25: Any
    faiss_index: Any
    faiss_meta: list[dict[str, Any]]

    @classmethod
    def load(
        cls,
        chunks_path: Path,
        bm25_path: Path,
        faiss_index_path: Path,
        faiss_meta_path: Path,
    ) -> "RAGResources":
        with chunks_path.open("r", encoding="utf-8") as handle:
            chunks = json.load(handle)
        with bm25_path.open("rb") as handle:
            bm25_obj = pickle.load(handle)
        bm25 = bm25_obj[0] if isinstance(bm25_obj, tuple) else bm25_obj
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss-cpu is required to load the vector index") from exc
        faiss_index = faiss.read_index(str(faiss_index_path))
        with faiss_meta_path.open("r", encoding="utf-8") as handle:
            faiss_meta = json.load(handle)
        return cls(chunks=chunks, bm25=bm25, faiss_index=faiss_index, faiss_meta=faiss_meta)


class PathologyRAGAgent:
    def __init__(
        self,
        client: SiliconFlowClient,
        resources: RAGResources,
        *,
        redact_reports: bool = True,
        include_retrieved_text: bool = False,
    ) -> None:
        self.client = client
        self.resources = resources
        self.redact_reports = redact_reports
        self.include_retrieved_text = include_retrieved_text

    @staticmethod
    def _parse_numbers(text: str, upper_bound: int) -> list[int]:
        indices: list[int] = []
        for value in re.findall(r"\d+", text):
            index = int(value) - 1
            if 0 <= index < upper_bound and index not in indices:
                indices.append(index)
        return indices

    @staticmethod
    def _merge_and_deduplicate(
        first: Iterable[dict[str, Any]], second: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for item in [*first, *second]:
            key = str(item.get("chunk_id") or item.get("id") or item.get("text", "")[:80])
            if key and key not in seen:
                seen.add(key)
                merged.append(item)
        return merged

    def _bm25_retrieve(self, query: str, k: int = 30) -> list[dict[str, Any]]:
        scores = self.resources.bm25.get_scores(query.lower().split())
        top_ids = np.argsort(scores)[::-1][:k]
        return [self.resources.chunks[int(i)] for i in top_ids]

    def _faiss_retrieve(self, query: str, k: int = 30) -> list[dict[str, Any]]:
        vector = self.client.embed(query)
        _, indices = self.resources.faiss_index.search(vector, k)
        data = self.resources.faiss_meta or self.resources.chunks
        return [data[int(i)] for i in indices[0] if 0 <= int(i) < len(data)]

    def _rerank(
        self, query: str, candidates: Sequence[dict[str, Any]], top_k: int = 3
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        passages = "\n".join(
            f"[{i + 1}] {item.get('text', '')}" for i, item in enumerate(candidates)
        )
        relevance_prompt = f"""Select passages relevant to the pathology query.

Query:
{query}

Passages:
{passages}

Return only relevant passage numbers separated by commas."""
        relevant_raw = self.client.chat(relevance_prompt, max_tokens=200)
        relevant_ids = self._parse_numbers(relevant_raw, len(candidates))
        relevant = [candidates[i] for i in relevant_ids]
        if not relevant:
            return []

        relevant_text = "\n".join(
            f"[{i + 1}] {item.get('text', '')}" for i, item in enumerate(relevant)
        )
        rerank_prompt = f"""Rank the passages from most to least relevant.

Query:
{query}

Passages:
{relevant_text}

Return only passage numbers separated by commas."""
        order_raw = self.client.chat(rerank_prompt, max_tokens=200)
        order = self._parse_numbers(order_raw, len(relevant))
        return [relevant[i] for i in order[:top_k]]

    @staticmethod
    def _format_chunks(chunks: Sequence[dict[str, Any]]) -> str:
        formatted: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            title = chunk.get("title", "Untitled section")
            raw_path = chunk.get("path", [])
            path = " > ".join(raw_path) if isinstance(raw_path, list) else str(raw_path)
            text = str(chunk.get("text", "")).strip()
            formatted.append(f"[{i}]\nSection: {title}\nPath: {path}\n{text}")
        return "\n\n".join(formatted)

    def retrieve_guidelines(self, report: str) -> str | None:
        translation_prompt = f"""Translate this Chinese pathology report into literal English.
Do not add interpretation or extra medical terms. Preserve its structure.
Translate (+) as positive and (-) as negative.

Report:
{report}"""
        translated = self.client.chat(translation_prompt, max_tokens=1024)
        query = f"""Pathology report:
{translated}

Task: identify the most likely tumor type using morphology, site, and immunohistochemistry."""
        candidates = self._merge_and_deduplicate(
            self._bm25_retrieve(query), self._faiss_retrieve(query)
        )
        selected = self._rerank(query, candidates, top_k=3)
        return self._format_chunks(selected) if selected else None

    @staticmethod
    def _parse_diagnosis(output: str) -> tuple[str, str]:
        clean = output.replace("*", "")
        diagnosis_match = re.search(
            r"Diagnosis:\s*(.+?)(?=\n\s*Reasoning:|$)", clean, re.IGNORECASE | re.DOTALL
        )
        reasoning_match = re.search(
            r"Reasoning:\s*(.+)", clean, re.IGNORECASE | re.DOTALL
        )
        if not diagnosis_match:
            raise ValueError("Model output does not contain a Diagnosis field")
        diagnosis = diagnosis_match.group(1).strip()
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        return diagnosis, reasoning

    def _generate(
        self,
        report: str,
        guideline_content: str | None,
        previous_answer: str | None = None,
        feedback: str | None = None,
    ) -> tuple[str, str]:
        prompt = f"""You are assisting with a research differential-diagnosis task.
Use only the provided pathology report and guideline excerpts.

Pathology report:
{report}

Relevant guideline excerpts:
{guideline_content or 'No relevant passage was retrieved.'}

Identify the most likely tumor type. Give the disease name in Chinese and a concise 1-2 sentence explanation.

Format exactly:
Diagnosis: [Chinese disease name]
Reasoning: [concise explanation]"""
        if previous_answer and feedback:
            prompt += f"""

Previous answer:
{previous_answer}

Reviewer feedback:
{feedback}

Revise only when the evidence supports a change."""
        return self._parse_diagnosis(self.client.chat(prompt, temperature=0.3, max_tokens=1024))

    def _review(self, report: str, diagnosis: str, reasoning: str, round_index: int) -> tuple[str, str]:
        prompt = f"""Review the proposed pathology diagnosis using only the supplied case.
Consider site, morphology, immunohistochemistry, and treatment history.
Do not invent findings.

Pathology report:
{report}

Proposed diagnosis:
{diagnosis}

Proposed reasoning:
{reasoning}

Return exactly:
Evaluation: [PASS or NEEDS_IMPROVEMENT]
Reasoning: [1-2 sentence explanation]"""
        temperature = max(0.2, 0.3 - round_index * 0.05)
        output = self.client.chat(prompt, temperature=temperature, max_tokens=512)
        evaluation_match = re.search(
            r"Evaluation:\s*(PASS|NEEDS_IMPROVEMENT)", output, re.IGNORECASE
        )
        reason_match = re.search(r"Reasoning:\s*(.+)", output, re.IGNORECASE | re.DOTALL)
        evaluation = evaluation_match.group(1).upper() if evaluation_match else "NEEDS_IMPROVEMENT"
        reason = reason_match.group(1).strip() if reason_match else "The review output was incomplete."
        return evaluation, reason

    def diagnose(self, case_id: str, report: str) -> dict[str, Any]:
        safe_report = PrivacyFilter.redact(report) if self.redact_reports else report.strip()
        if not safe_report:
            raise ValueError("The report is empty after redaction")

        guideline_content = self.retrieve_guidelines(safe_report)
        previous_answer: str | None = None
        feedback: str | None = None
        answer_history: list[str] = []
        diagnosis = ""
        reasoning = ""
        evaluation = "NEEDS_IMPROVEMENT"
        for round_index in range(self.client.settings.max_rounds):
            diagnosis, reasoning = self._generate(
                safe_report,
                guideline_content,
                previous_answer=previous_answer,
                feedback=feedback,
            )
            evaluation, feedback = self._review(
                safe_report, diagnosis, reasoning, round_index
            )
            answer_history.append(diagnosis)
            if evaluation == "PASS":
                break
            previous_answer = diagnosis
            if len(answer_history) >= 3 and len(set(answer_history[-3:])) == 1:
                break

        return {
            "case_id": case_id,
            "diagnosis": diagnosis,
            "reasoning": reasoning,
            "review": evaluation,
            "rounds": len(answer_history),
            "guideline_context_used": bool(guideline_content),
            **({"retrieved_guidelines": guideline_content} if self.include_retrieved_text else {}),
        }


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Cases file must contain a JSON object or a list of objects")
    return data


def save_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="JSON cases file")
    parser.add_argument("--chunks", type=Path, required=True, help="Guideline chunks JSON")
    parser.add_argument("--bm25", type=Path, required=True, help="BM25 pickle")
    parser.add_argument("--faiss-index", type=Path, required=True, help="FAISS index")
    parser.add_argument("--faiss-meta", type=Path, required=True, help="FAISS metadata JSON")
    parser.add_argument("--output", type=Path, default=Path("outputs/results.json"))
    parser.add_argument(
        "--include-retrieved-text",
        action="store_true",
        help="Store retrieved guideline text in results. Disabled by default.",
    )
    parser.add_argument(
        "--disable-redaction",
        action="store_true",
        help="Send raw report text. Not recommended for sensitive data.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings.from_env()
    client = SiliconFlowClient(settings)
    resources = RAGResources.load(
        args.chunks, args.bm25, args.faiss_index, args.faiss_meta
    )
    agent = PathologyRAGAgent(
        client,
        resources,
        redact_reports=not args.disable_redaction,
        include_retrieved_text=args.include_retrieved_text,
    )

    results: list[dict[str, Any]] = []
    for position, case in enumerate(load_cases(args.cases), start=1):
        case_id = str(case.get("case_id") or f"case-{position:04d}")
        report = str(case.get("report") or "")
        print(f"Processing {case_id}...")
        try:
            results.append(agent.diagnose(case_id, report))
        except Exception as exc:  # Keep batch processing without exposing the report.
            results.append(
                {
                    "case_id": case_id,
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            )

    save_results(args.output, results)
    print(f"Saved {len(results)} result(s) to {args.output}")


if __name__ == "__main__":
    main()



