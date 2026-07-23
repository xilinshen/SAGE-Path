#!/usr/bin/env python3
"""A compact retrieval-and-reflection pipeline for pathology diagnosis.

Workflow:

    hybrid retrieval -> relevance filtering and reranking -> generation
    -> reflection -> optional re-retrieval and revision
"""

from __future__ import annotations

import json
import os
import pickle
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import requests

from prompts import (
    DIAGNOSIS_PROMPT,
    REFLECTION_PROMPT,
    RELEVANCE_PROMPT,
    RERANK_PROMPT,
    RETRIEVAL_INSTRUCTION,
    REVISION_CONTEXT,
    TRANSLATION_PROMPT,
)


class ConfigurationError(RuntimeError):
    """Raised when required local configuration is missing or invalid."""


class ModelAPIError(RuntimeError):
    """Raised when a model API request fails after retries."""


def load_local_env(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs without an additional dependency."""
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


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    api_key: str
    api_base_url: str
    chat_model: str
    embedding_model: str
    timeout_seconds: int = 240
    temperature: float = 0.1
    top_p: float = 0.95
    max_retries: int = 5
    max_reflection_loops: int = 5

    @classmethod
    def from_environment(cls) -> "Settings":
        load_local_env()
        values = {
            "API_KEY": os.getenv("API_KEY", "").strip(),
            "API_BASE_URL": os.getenv("API_BASE_URL", "").strip(),
            "CHAT_MODEL": os.getenv("CHAT_MODEL", "").strip(),
            "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "").strip(),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ConfigurationError(
                "Missing required environment variables: " + ", ".join(missing)
            )

        return cls(
            api_key=values["API_KEY"],
            api_base_url=values["API_BASE_URL"].rstrip("/"),
            chat_model=values["CHAT_MODEL"],
            embedding_model=values["EMBEDDING_MODEL"],
            timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "240")),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            top_p=float(os.getenv("TOP_P", "0.95")),
            max_retries=int(os.getenv("MAX_RETRIES", "5")),
            max_reflection_loops=int(os.getenv("MAX_REFLECTION_LOOPS", "5")),
        )


class PrivacyFilter:
    """Best-effort removal of common direct identifiers."""

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


class OpenAICompatibleClient:
    """Small client for compatible chat and embedding endpoints."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = requests.Session()

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        url = f"{self.settings.api_base_url}/{endpoint.lstrip('/')}"

        for attempt in range(self.settings.max_retries):
            try:
                response = self.session.post(
                    url,
                    headers=self.headers,
                    json=dict(payload),
                    timeout=self.settings.timeout_seconds,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"Transient API error {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("API response is not a JSON object")
                return data
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == self.settings.max_retries - 1:
                    break
                delay = min(20.0, 1.5**attempt) * random.uniform(0.8, 1.2)
                time.sleep(delay)

        raise ModelAPIError(f"API request failed after retries: {last_error}")

    def chat(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float | None = None,
    ) -> str:
        payload = {
            "model": self.settings.chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": (
                self.settings.temperature if temperature is None else temperature
            ),
            "top_p": self.settings.top_p,
            "frequency_penalty": 0,
            "n": 1,
        }
        data = self._post("chat/completions", payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelAPIError("Unexpected chat response structure") from exc
        if not isinstance(content, str) or not content.strip():
            raise ModelAPIError("Chat endpoint returned empty content")
        return content.strip()

    def embed(self, text: str) -> np.ndarray:
        payload = {
            "model": self.settings.embedding_model,
            "input": text,
            "encoding_format": "float",
        }
        data = self._post("embeddings", payload)
        try:
            vector = np.asarray(data["data"][0]["embedding"], dtype=np.float32)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelAPIError("Unexpected embedding response structure") from exc
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ModelAPIError("Embedding endpoint returned a zero vector")
        return vector / norm


@dataclass
class KnowledgeBase:
    """Locally stored BM25 and FAISS resources."""

    bm25: Any
    sparse_chunks: list[dict[str, Any]]
    faiss_index: Any
    dense_chunks: list[dict[str, Any]]

    @classmethod
    def load(cls, data_dir: str | Path = "data") -> "KnowledgeBase":
        root = Path(data_dir)
        required = {
            "corpus": root / "11_who_chunk.json",
            "bm25": root / "bm25.pkl",
            "faiss": root / "faiss_index.index",
            "metadata": root / "faiss_index_meta.json",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise ConfigurationError(
                "Missing local knowledge-base files:\n- " + "\n- ".join(missing)
            )

        with required["corpus"].open("r", encoding="utf-8") as handle:
            sparse_chunks = json.load(handle)
        with required["metadata"].open("r", encoding="utf-8") as handle:
            dense_chunks = json.load(handle)
        with required["bm25"].open("rb") as handle:
            bm25_payload = pickle.load(handle)
        bm25 = bm25_payload[0] if isinstance(bm25_payload, tuple) else bm25_payload

        try:
            import faiss
        except ImportError as exc:
            raise ConfigurationError(
                "faiss-cpu is required to load the vector index."
            ) from exc
        index = faiss.read_index(str(required["faiss"]))

        if not isinstance(sparse_chunks, list) or not isinstance(dense_chunks, list):
            raise ConfigurationError("Corpus and FAISS metadata must be JSON lists.")
        if len(dense_chunks) != index.ntotal:
            raise ConfigurationError(
                "FAISS metadata length does not match the number of indexed vectors."
            )
        for name, chunks in (
            ("11_who_chunk.json", sparse_chunks),
            ("faiss_index_meta.json", dense_chunks),
        ):
            if any(not isinstance(item, dict) or "text" not in item for item in chunks):
                raise ConfigurationError(
                    f"Every entry in {name} must be an object with a text field."
                )

        return cls(
            bm25=bm25,
            sparse_chunks=sparse_chunks,
            faiss_index=index,
            dense_chunks=dense_chunks,
        )


@dataclass(frozen=True)
class RetrievedPassage:
    chunk: dict[str, Any]
    rank: int


class HybridRetriever:
    """BM25 and FAISS retrieval followed by LLM filtering and reranking."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        client: OpenAICompatibleClient,
        *,
        candidate_k: int = 30,
        final_k: int = 5,
    ) -> None:
        self.kb = knowledge_base
        self.client = client
        self.candidate_k = candidate_k
        self.final_k = final_k

    @staticmethod
    def _dedupe_key(chunk: Mapping[str, Any]) -> str:
        for key in ("chunk_id", "id"):
            value = chunk.get(key)
            if value is not None:
                return f"{key}:{value}"
        return str(chunk.get("text", ""))[:160]

    @staticmethod
    def _parse_indices(raw: str, upper_bound: int) -> list[int]:
        values: list[int] = []
        for token in re.findall(r"\d+", raw):
            index = int(token) - 1
            if 0 <= index < upper_bound and index not in values:
                values.append(index)
        return values

    def _sparse_search(self, query: str) -> list[dict[str, Any]]:
        query_tokens = query.lower().split()
        scores = np.asarray(self.kb.bm25.get_scores(query_tokens))
        ids = np.argsort(scores)[::-1][: self.candidate_k]
        return [self.kb.sparse_chunks[int(index)] for index in ids]

    def _dense_search(self, query: str) -> list[dict[str, Any]]:
        vector = self.client.embed(query).reshape(1, -1).astype(np.float32)
        _, indices = self.kb.faiss_index.search(vector, self.candidate_k)
        valid = [
            int(index)
            for index in indices[0]
            if 0 <= int(index) < len(self.kb.dense_chunks)
        ]
        return [self.kb.dense_chunks[index] for index in valid]

    def _merge(self, *groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for group in groups:
            for chunk in group:
                key = self._dedupe_key(chunk)
                if key not in seen:
                    seen.add(key)
                    merged.append(chunk)
        return merged

    @staticmethod
    def _numbered_passages(passages: Sequence[Mapping[str, Any]]) -> str:
        return "\n\n".join(
            f"[{index}] {passage.get('text', '').strip()}"
            for index, passage in enumerate(passages, start=1)
        )

    def _filter_relevant(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prompt = RELEVANCE_PROMPT.format(
            query=query,
            passages=self._numbered_passages(candidates),
        )
        raw = self.client.chat(prompt, max_tokens=200, temperature=0.1)
        indices = self._parse_indices(raw, len(candidates))
        return [candidates[index] for index in indices]

    def _rerank(
        self,
        query: str,
        passages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prompt = RERANK_PROMPT.format(
            query=query,
            passages=self._numbered_passages(passages),
        )
        raw = self.client.chat(prompt, max_tokens=200, temperature=0.1)
        indices = self._parse_indices(raw, len(passages))
        return [passages[index] for index in indices]

    def retrieve(self, query: str) -> list[RetrievedPassage]:
        candidates = self._merge(
            self._sparse_search(query),
            self._dense_search(query),
        )
        if not candidates:
            return []

        relevant = self._filter_relevant(query, candidates) or candidates
        ranked = self._rerank(query, relevant) or relevant
        return [
            RetrievedPassage(chunk=chunk, rank=rank)
            for rank, chunk in enumerate(ranked[: self.final_k], start=1)
        ]


@dataclass
class ReflectionRecord:
    iteration: int
    evaluation: str
    feedback: str


@dataclass
class SAGEPathResult:
    answer: str
    parsed: dict[str, str | None]
    reflection_history: list[ReflectionRecord] = field(default_factory=list)
    retrieved_passage_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable result without report or passage text."""
        return {
            "answer": self.answer,
            "parsed": self.parsed,
            "reflection_history": [
                {
                    "iteration": record.iteration,
                    "evaluation": record.evaluation,
                    "feedback": record.feedback,
                }
                for record in self.reflection_history
            ],
            "retrieved_passage_ids": self.retrieved_passage_ids,
        }


class SAGEPath:
    """Pathology diagnosis pipeline with retrieval and iterative reflection."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        retriever: HybridRetriever,
        *,
        max_reflection_loops: int = 5,
        redact_reports: bool = True,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.max_reflection_loops = max_reflection_loops
        self.redact_reports = redact_reports

    @staticmethod
    def _format_chunk(chunk: Mapping[str, Any], rank: int) -> str:
        title = str(chunk.get("title", "Untitled Section"))
        raw_path = chunk.get("path", [])
        path = " > ".join(str(item) for item in raw_path) if isinstance(raw_path, list) else str(raw_path)
        text = str(chunk.get("text", "")).strip()
        return f"[{rank}]\n[Section: {title}]\n[Path: {path}]\n{text}"

    @classmethod
    def _format_guidelines(cls, passages: Sequence[RetrievedPassage]) -> str:
        return "\n\n".join(
            cls._format_chunk(passage.chunk, passage.rank) for passage in passages
        )

    @staticmethod
    def _extract_field(text: str, field_name: str) -> str | None:
        pattern = rf"(?im)^\s*{re.escape(field_name)}\s*:\s*(.+?)(?=\n\s*[A-Za-z][A-Za-z ]*\s*:|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    @classmethod
    def _parse_answer(cls, raw: str) -> dict[str, str | None]:
        return {
            "diagnosis": cls._extract_field(raw, "Diagnosis"),
            "reasoning": cls._extract_field(raw, "Reasoning"),
        }

    @classmethod
    def _parse_reflection(cls, raw: str) -> tuple[str, str]:
        evaluation = cls._extract_field(raw, "Evaluation") or "NEEDS_IMPROVEMENT"
        feedback = cls._extract_field(raw, "Reasoning") or raw.strip()
        normalized = evaluation.upper().replace(" ", "_")
        if "PASS" in normalized and "NEEDS" not in normalized:
            normalized = "PASS"
        else:
            normalized = "NEEDS_IMPROVEMENT"
        return normalized, feedback

    def _translate_for_retrieval(self, report: str) -> str:
        return self.client.chat(
            TRANSLATION_PROMPT.format(report=report),
            max_tokens=2048,
            temperature=0.1,
        )

    @staticmethod
    def _build_retrieval_query(
        translated_report: str,
        previous_answer: str | None = None,
        feedback: str | None = None,
    ) -> str:
        parts = [
            f"Pathology report:\n{translated_report}",
            f"Task: {RETRIEVAL_INSTRUCTION}",
        ]
        if previous_answer:
            parts.append(f"Previous answer:\n{previous_answer}")
        if feedback:
            parts.append(f"Reflection feedback:\n{feedback}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_inference_prompt(
        report: str,
        guideline_content: str,
        previous_answer: str | None = None,
        feedback: str | None = None,
    ) -> str:
        prompt = DIAGNOSIS_PROMPT.format(
            report=report,
            guideline=guideline_content,
        )
        if previous_answer and feedback:
            prompt += REVISION_CONTEXT.format(
                previous_answer=previous_answer,
                feedback=feedback,
            )
        return prompt

    @staticmethod
    def _build_reflection_prompt(
        report: str,
        parsed: Mapping[str, str | None],
        raw_answer: str,
    ) -> str:
        return REFLECTION_PROMPT.format(
            report=report,
            diagnosis=parsed.get("diagnosis") or raw_answer,
            reasoning=parsed.get("reasoning") or "",
        )

    def run(self, report: str) -> SAGEPathResult:
        """Run diagnostic inference with retrieval and iterative reflection."""
        safe_report = PrivacyFilter.redact(report) if self.redact_reports else report.strip()
        if not safe_report:
            raise ValueError("Report text is empty.")

        translated_report = self._translate_for_retrieval(safe_report)
        previous_answer: str | None = None
        feedback: str | None = None
        history: list[ReflectionRecord] = []
        latest_passages: list[RetrievedPassage] = []
        raw_answer = ""
        parsed: dict[str, str | None] = {}

        for iteration in range(1, self.max_reflection_loops + 1):
            query = self._build_retrieval_query(
                translated_report,
                previous_answer=previous_answer,
                feedback=feedback,
            )
            latest_passages = self.retriever.retrieve(query)
            if not latest_passages:
                raise RuntimeError("No knowledge-base passages were retrieved.")

            inference_prompt = self._build_inference_prompt(
                safe_report,
                self._format_guidelines(latest_passages),
                previous_answer=previous_answer,
                feedback=feedback,
            )
            raw_answer = self.client.chat(
                inference_prompt,
                max_tokens=2048,
                temperature=0.1,
            )
            parsed = self._parse_answer(raw_answer)

            reflection_prompt = self._build_reflection_prompt(
                safe_report,
                parsed,
                raw_answer,
            )
            raw_reflection = self.client.chat(
                reflection_prompt,
                max_tokens=768,
                temperature=max(0.1, 0.3 - 0.05 * (iteration - 1)),
            )
            evaluation, feedback = self._parse_reflection(raw_reflection)
            history.append(
                ReflectionRecord(
                    iteration=iteration,
                    evaluation=evaluation,
                    feedback=feedback,
                )
            )
            if evaluation == "PASS":
                break
            previous_answer = raw_answer

        passage_ids = [
            str(
                passage.chunk.get("chunk_id")
                or passage.chunk.get("id")
                or f"rank-{passage.rank}"
            )
            for passage in latest_passages
        ]
        return SAGEPathResult(
            answer=raw_answer,
            parsed=parsed,
            reflection_history=history,
            retrieved_passage_ids=passage_ids,
        )


def create_pipeline(
    data_dir: str | Path = "data",
    *,
    redact_reports: bool = True,
) -> SAGEPath:
    """Create a pipeline from environment settings and local index files."""
    settings = Settings.from_environment()
    client = OpenAICompatibleClient(settings)
    knowledge_base = KnowledgeBase.load(data_dir)
    retriever = HybridRetriever(knowledge_base, client)
    return SAGEPath(
        client,
        retriever,
        max_reflection_loops=settings.max_reflection_loops,
        redact_reports=redact_reports,
    )
