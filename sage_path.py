"""Minimal public implementation of the SAGE-Path inference workflow.

This module implements only the full workflow used in the paper:

    hybrid retrieval -> LLM relevance filtering and reranking -> generation
    -> reflection -> optional re-retrieval and revision

It intentionally excludes benchmarks, ablations, evaluation metrics, plotting,
and any copyrighted or patient-level data.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import faiss
import numpy as np
import requests

from prompts import PROMPTS, SUPPORTED_TASKS, TASKS


class ConfigurationError(RuntimeError):
    """Raised when required local configuration is missing or invalid."""


class ModelAPIError(RuntimeError):
    """Raised when a model API request fails after retries."""


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    api_key: str
    api_base_url: str = "https://api.siliconflow.cn/v1"
    chat_model: str = "deepseek-ai/DeepSeek-R1"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    timeout_seconds: int = 240
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 50

    @classmethod
    def from_environment(cls) -> "Settings":
        key = (
            os.getenv("SILICONFLOW_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("SILICONFLOW_API_KEY_CHAT")
        )
        if not key:
            raise ConfigurationError(
                "Set SILICONFLOW_API_KEY (or OPENAI_API_KEY) before running inference."
            )
        return cls(
            api_key=key,
            api_base_url=os.getenv(
                "API_BASE_URL", "https://api.siliconflow.cn/v1"
            ).rstrip("/"),
            chat_model=os.getenv("CHAT_MODEL", "deepseek-ai/DeepSeek-R1"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"
            ),
            timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "240")),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
        )


class OpenAICompatibleClient:
    """Small client for OpenAI-compatible chat and embedding endpoints."""

    def __init__(self, settings: Settings, max_retries: int = 5) -> None:
        self.settings = settings
        self.max_retries = max_retries
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

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    url,
                    headers=self.headers,
                    json=dict(payload),
                    timeout=self.settings.timeout_seconds,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"Transient API error {response.status_code}", response=response
                    )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("API response is not a JSON object")
                return data
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
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
            "top_k": self.settings.top_k,
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
    """Locally held BM25 and FAISS resources.

    The files used in the study are not distributed with this repository.
    See ``data/README.md`` for the expected local layout and copyright notice.
    """

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
        index = faiss.read_index(str(required["faiss"]))

        if not isinstance(sparse_chunks, list) or not isinstance(dense_chunks, list):
            raise ConfigurationError("Corpus and FAISS metadata must be JSON lists.")
        if len(dense_chunks) != index.ntotal:
            raise ConfigurationError(
                "FAISS metadata length does not match the number of indexed vectors."
            )
        for collection_name, chunks in (
            ("11_who_chunk.json", sparse_chunks),
            ("faiss_index_meta.json", dense_chunks),
        ):
            if any(not isinstance(item, dict) or "text" not in item for item in chunks):
                raise ConfigurationError(
                    f"Every entry in {collection_name} must be an object with a text field."
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
    """BM25 + FAISS retrieval followed by LLM filtering and reranking."""

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
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        prompt = f"""You are given a user query and a list of candidate passages. Determine which passages are topically relevant to the query.

Query:
{query}

Passages:
{self._numbered_passages(candidates)}

Only output the numbers of the relevant passages, separated by commas. Do not output any other text."""
        raw = self.client.chat(prompt, max_tokens=200, temperature=0.1)
        indices = self._parse_indices(raw, len(candidates))
        return [candidates[index] for index in indices]

    def _rerank(
        self, query: str, passages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        prompt = f"""You are given a user query and a list of relevant passages. Rank them from most to least relevant to the query.

Query:
{query}

Passages:
{self._numbered_passages(passages)}

Only output the passage numbers in ranked order, separated by commas. Do not output any other text."""
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

        relevant = self._filter_relevant(query, candidates)
        if not relevant:
            relevant = candidates
        ranked = self._rerank(query, relevant)
        if not ranked:
            ranked = relevant

        return [
            RetrievedPassage(chunk=chunk, rank=rank)
            for rank, chunk in enumerate(ranked[: self.final_k], start=1)
        ]


@dataclass
class ReflectionRecord:
    iteration: int
    evaluation: str
    feedback: str
    module_with_error: str | None = None


@dataclass
class SAGEPathResult:
    task: str
    answer: str
    parsed: dict[str, str | None]
    reflection_history: list[ReflectionRecord] = field(default_factory=list)
    retrieved_passage_ids: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        """Return an output record without report text or retrieved guideline text."""
        return {
            "task": self.task,
            "answer": self.answer,
            "parsed": self.parsed,
            "reflection_history": [
                {
                    "iteration": record.iteration,
                    "evaluation": record.evaluation,
                    "module_with_error": record.module_with_error,
                }
                for record in self.reflection_history
            ],
            "retrieved_passage_ids": self.retrieved_passage_ids,
        }


class SAGEPath:
    """Full SAGE-Path workflow with retrieval and reflection always enabled."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        retriever: HybridRetriever,
        *,
        max_reflection_loops: int = 5,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.max_reflection_loops = max_reflection_loops

    @staticmethod
    def _format_chunk(chunk: Mapping[str, Any], rank: int) -> str:
        title = str(chunk.get("title", "Untitled Section"))
        raw_path = chunk.get("path", [])
        if isinstance(raw_path, list):
            path = " > ".join(str(item) for item in raw_path)
        else:
            path = str(raw_path)
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
    def _parse_answer(cls, task: str, raw: str) -> dict[str, str | None]:
        if task == "complex_pathology":
            return {
                "diagnosis": cls._extract_field(raw, "Diagnosis"),
                "reasoning": cls._extract_field(raw, "Reasoning"),
                "recommendation": (
                    cls._extract_field(raw, "Follow up recommendation")
                    or cls._extract_field(raw, "Recommendation")
                ),
                "rationale": cls._extract_field(raw, "Rationale"),
            }
        return {
            "diagnosis": cls._extract_field(raw, "Diagnosis"),
            "reasoning": cls._extract_field(raw, "Reasoning"),
        }

    @classmethod
    def _parse_reflection(cls, raw: str) -> tuple[str, str, str | None]:
        evaluation = cls._extract_field(raw, "Evaluation") or "NEEDS_IMPROVEMENT"
        feedback = cls._extract_field(raw, "Reasoning") or raw.strip()
        module = cls._extract_field(raw, "ModuleWithError")
        normalized = evaluation.upper().replace(" ", "_")
        if "PASS" in normalized and "NEEDS" not in normalized:
            normalized = "PASS"
        else:
            normalized = "NEEDS_IMPROVEMENT"
        return normalized, feedback, module

    def _translate_for_retrieval(self, report: str) -> str:
        prompt = PROMPTS["translation"].replace("{case['report']}", report)
        return self.client.chat(prompt, max_tokens=2048, temperature=0.1)

    @staticmethod
    def _validate_task(task: str, fields: Mapping[str, Any]) -> None:
        if task not in SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task '{task}'. Choose one of: {', '.join(SUPPORTED_TASKS)}"
            )
        missing = [name for name in TASKS[task]["required_fields"] if not fields.get(name)]
        if missing:
            raise ValueError(f"Missing required task fields: {', '.join(missing)}")

    def _build_retrieval_query(
        self,
        task: str,
        translated_report: str,
        fields: Mapping[str, Any],
        previous_answer: str | None = None,
        feedback: str | None = None,
    ) -> str:
        location = fields.get("location")
        parts = [f"Pathology Report:\n{translated_report}"]
        if location:
            parts.append(f"Metastatic Site: {location}")
        parts.append(f"Task: {TASKS[task]['retrieval_instruction']}")
        if previous_answer:
            parts.append(f"Previous answer:\n{previous_answer}")
        if feedback:
            parts.append(f"Reflection feedback:\n{feedback}")
        return "\n\n".join(parts)

    def _build_inference_prompt(
        self,
        task: str,
        report: str,
        guideline_content: str,
        fields: Mapping[str, Any],
        previous_answer: str | None = None,
        feedback: str | None = None,
    ) -> str:
        template = TASKS[task]["inference_prompt"]
        values = {
            "report": report,
            "who_guideline_content": guideline_content,
            "guideline": guideline_content,
            "location": fields.get("location", ""),
        }
        prompt = template.format(**values)
        if previous_answer and feedback:
            prompt += (
                "\n\nRevision context:\n"
                f"Previous answer:\n{previous_answer}\n\n"
                f"Reflection feedback:\n{feedback}\n\n"
                "Revise the answer only where necessary. Keep the required output format, "
                "use only the case information, and avoid unsupported assumptions."
            )
        return prompt

    def _build_reflection_prompt(
        self,
        task: str,
        report: str,
        parsed: Mapping[str, str | None],
        fields: Mapping[str, Any],
        raw_answer: str,
    ) -> str:
        template = TASKS[task]["reflection_prompt"]
        values = {
            "case_text": report,
            "report": report,
            "metastasis_location": fields.get("location", ""),
            "model_diagnosis": parsed.get("diagnosis") or raw_answer,
            "reasoning_text": parsed.get("reasoning") or "",
            "diagnosis": parsed.get("diagnosis") or raw_answer,
            "reasoning": parsed.get("reasoning") or "",
            "recommendation": parsed.get("recommendation") or "None",
            "rationale": parsed.get("rationale") or "",
        }
        return template.format(**values)

    def run(
        self,
        report: str,
        *,
        task: str = "rare_tumor",
        fields: Mapping[str, Any] | None = None,
    ) -> SAGEPathResult:
        """Run the full retrieval-and-reflection pipeline.

        The caller must provide a de-identified report. The report and retrieved
        guideline text are not included in the returned public result object.
        """
        fields = dict(fields or {})
        self._validate_task(task, fields)
        if not report.strip():
            raise ValueError("Report text is empty.")

        translated_report = self._translate_for_retrieval(report)
        previous_answer: str | None = None
        feedback: str | None = None
        history: list[ReflectionRecord] = []
        latest_passages: list[RetrievedPassage] = []
        raw_answer = ""
        parsed: dict[str, str | None] = {}

        for iteration in range(1, self.max_reflection_loops + 1):
            query = self._build_retrieval_query(
                task,
                translated_report,
                fields,
                previous_answer=previous_answer,
                feedback=feedback,
            )
            latest_passages = self.retriever.retrieve(query)
            if not latest_passages:
                raise RuntimeError("No guideline passages were retrieved.")

            inference_prompt = self._build_inference_prompt(
                task,
                report,
                self._format_guidelines(latest_passages),
                fields,
                previous_answer=previous_answer,
                feedback=feedback,
            )
            raw_answer = self.client.chat(
                inference_prompt, max_tokens=2048, temperature=0.1
            )
            parsed = self._parse_answer(task, raw_answer)

            reflection_prompt = self._build_reflection_prompt(
                task, report, parsed, fields, raw_answer
            )
            raw_reflection = self.client.chat(
                reflection_prompt,
                max_tokens=768,
                temperature=max(0.1, 0.3 - 0.05 * (iteration - 1)),
            )
            evaluation, feedback, module = self._parse_reflection(raw_reflection)
            history.append(
                ReflectionRecord(
                    iteration=iteration,
                    evaluation=evaluation,
                    feedback=feedback,
                    module_with_error=module,
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
            task=task,
            answer=raw_answer,
            parsed=parsed,
            reflection_history=history,
            retrieved_passage_ids=passage_ids,
        )


def create_pipeline(
    data_dir: str | Path = "data", *, max_reflection_loops: int = 5
) -> SAGEPath:
    """Create a pipeline from environment settings and local index files."""
    settings = Settings.from_environment()
    client = OpenAICompatibleClient(settings)
    knowledge_base = KnowledgeBase.load(data_dir)
    retriever = HybridRetriever(knowledge_base, client)
    return SAGEPath(
        client,
        retriever,
        max_reflection_loops=max_reflection_loops,
    )
