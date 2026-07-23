# SAGE-Path

SAGE-Path is a compact pathology diagnostic assistant that combines hybrid retrieval, evidence reranking, answer generation, and iterative self-review.

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

## Prompt customization

`prompts.py` contains example prompts for one pathology diagnosis workflow. Edit these templates to match the intended diagnostic task, output format and language.

## Repository layout

```text
SAGE-Path/
├── sage_path.py              # retrieval, reranking, generation, and reflection
├── prompts.py                # task prompt examples
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
The knowledge base was derived from 11 volumes of the fifth edition of the *WHO Classification of Tumours*. The following files are not included because they contain or encode copyrighted WHO content:

```text
data/11_who_chunk.json
data/bm25.pkl
data/faiss_index.index
data/faiss_index_meta.json
```

Users must obtain lawful access to the source materials and prepare compatible local resources. The repository license applies only to the source code and does not cover WHO or other third-party content. See data/README.md for the expected file layout.

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
- The synthetic example is not derived from a real patient.


## License

The source code is released under the Apache License 2.0. The license applies only to original source code in this repository and does not cover WHO publications, clinical data, model APIs, or other third-party material.
