# SAGE-Path

Minimal reference implementation of **SAGE-Path** (Self-reflective Agentic Guideline-grounded Engine for text-based pathology diagnostic assistance), accompanying the manuscript *“Guideline-grounded agentic AI enables reliable report-level diagnostic reasoning in oncologic pathology.”*

This repository contains only the full inference workflow used for SAGE-Path:

```text
De-identified pathology report
        ↓
Literal English translation for retrieval
        ↓
BM25 + FAISS hybrid retrieval
        ↓
LLM relevance filtering and reranking
        ↓
Task-specific answer generation
        ↓
Reflection without retrieved passages
        ↓
Re-retrieval and revision when needed
```

Retrieval and reflection are always enabled. Benchmarking, ablation settings, evaluation metrics, statistical analyses, reader-study code, and figure-generation scripts are intentionally excluded.

## Included task prompts

Task-specific inference and reflection prompts are transcribed from Supplementary Table 3:

- rare-tumour differential diagnosis;
- metastatic primary-site inference; and
- complex pathology diagnosis with follow-up testing recommendations.

The implementation is shared as a concise methodological reference, not as a complete reproduction package for every numerical result in the manuscript.

## Repository layout

```text
SAGE-Path/
├── sage_path.py              # retrieval, reranking, generation, and reflection
├── prompts.py                # task prompts from Supplementary Table 3
├── run_example.py            # one-case command-line example
├── examples/
│   ├── demo_case.json        # synthetic placeholder case
│   └── chunk_schema.json     # expected corpus-item structure
├── data/
│   └── README.md             # local index layout and copyright notice
├── .env.example
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## API configuration

Copy the example environment file and insert a newly issued API key:

```bash
cp .env.example .env
```

```env
SILICONFLOW_API_KEY=replace_with_your_own_key
API_BASE_URL=https://api.siliconflow.cn/v1
CHAT_MODEL=deepseek-ai/DeepSeek-R1
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

Never commit `.env` or an API key. Keys that were previously embedded in research scripts should be revoked before the repository is made public.

## Knowledge-base availability

The WHO-derived knowledge base used in the study was constructed from 11 volumes of the fifth edition of the *WHO Classification of Tumours*. The processed study corpus comprised 1,811 tumour entities and 48,396 retrievable text units.

The following study resources are **not included**:

```text
data/11_who_chunk.json
data/bm25.pkl
data/faiss_index.index
data/faiss_index_meta.json
```

They were derived from copyrighted WHO publications and may contain, reproduce, encode, or provide structured access to licensed content. They are therefore not redistributed through this repository and are not available from the authors for onward redistribution.

Users wishing to run retrieval must obtain lawful access to appropriate source material and prepare compatible local resources in accordance with the applicable licence terms. The source code license does not grant rights to the *WHO Classification of Tumours* or any other third-party material. See [`data/README.md`](data/README.md) for the expected layout and schema.

## Running one case

Edit `examples/demo_case.json` and replace the synthetic placeholder with a **fully de-identified** report. Do not place direct identifiers or other protected health information in the repository.

Rare-tumour diagnosis:

```json
{
  "task": "rare_tumor",
  "report": "<fully de-identified pathology report>",
  "fields": {}
}
```

Metastatic primary-site inference additionally requires the metastatic site:

```json
{
  "task": "metastatic_primary_site",
  "report": "<fully de-identified pathology report>",
  "fields": {
    "location": "liver"
  }
}
```

Complex pathology diagnosis and follow-up recommendations:

```json
{
  "task": "complex_pathology",
  "report": "<fully de-identified pathology report>",
  "fields": {}
}
```

Run:

```bash
python run_example.py --case examples/demo_case.json --data-dir data
```

Optionally save a privacy-minimized output that excludes the input report and retrieved guideline text:

```bash
python run_example.py \
  --case examples/demo_case.json \
  --data-dir data \
  --output outputs/demo_result.json
```

## Privacy and security

- Only de-identified reports should be submitted to an external model API.
- The repository contains no patient data, model outputs from study cases, credentials, or personal filesystem paths.
- Public result serialization excludes the report, retrieved guideline passages, and detailed reflection feedback by default.
- The synthetic example is not derived from a real patient.
- Review the data-processing agreement and privacy terms of the selected API provider before processing clinical text.

## Reproducibility scope

This repository reproduces the software logic of the SAGE-Path inference workflow. Exact study results also depend on restricted clinical data, the frozen WHO-derived corpus and indexes, model-service versions, prompts, and inference-time behavior. Consequently, this repository should not be interpreted as a standalone package for reproducing every value reported in the manuscript.

## Research-use disclaimer

This software is provided for research and methodological review only. It is not a medical device and must not be used for clinical diagnosis, treatment selection, or patient management without independent review by qualified professionals and all required regulatory, ethical, and institutional approvals.

## License

The source code is released under the Apache License 2.0. The license applies only to original source code in this repository and does not cover WHO publications, clinical data, model APIs, or other third-party material.
