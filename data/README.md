# Local knowledge base

Place the local retrieval resources in this directory:

```text
11_who_chunk.json
bm25.pkl
faiss_index.index
faiss_index_meta.json
```

`11_who_chunk.json` and `faiss_index_meta.json` must be JSON lists whose entries contain a `text` field. Optional fields such as `id`, `chunk_id`, `title`, and `path` are used when available.

These files are excluded from Git. Use only source materials that you are permitted to process and store.
