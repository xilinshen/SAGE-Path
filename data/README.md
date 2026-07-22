# Local knowledge-base files

Place the locally prepared knowledge-base resources in this directory:

```text
data/
├── 11_who_chunk.json
├── bm25.pkl
├── faiss_index.index
└── faiss_index_meta.json
```

These files are intentionally excluded from the public repository. The resources used in the study were derived from copyrighted third-party publications and may contain, reproduce, encode, or provide structured access to licensed content.

Users are responsible for:

1. obtaining lawful access to the source documents;
2. complying with the applicable licence and copyright terms;
3. preparing the corpus and indexes locally; and
4. ensuring that restricted content is not redistributed.

## Expected formats

`11_who_chunk.json` and `faiss_index_meta.json` must be JSON lists. Every list item must contain a `text` field. The optional `chunk_id`, `title`, and `path` fields are used for deduplication and readable context formatting. See `examples/chunk_schema.json`.

`bm25.pkl` may contain either the BM25 object itself or a tuple whose first element is the BM25 object, matching the format used by the original research scripts. Because Python pickle files can execute code during loading, use only an index file that you created yourself or obtained from a fully trusted source.

`faiss_index.index` must use the same row order as `faiss_index_meta.json`. The public implementation L2-normalizes query embeddings before FAISS search, consistent with cosine-similarity retrieval using normalized vectors.

Do not commit any locally generated corpus or index files to a public repository.
