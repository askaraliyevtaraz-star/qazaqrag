## Vector Database

Qdrant stores document-chunk embeddings together with
retrieval metadata.

Each point contains:

vector
+ chunk text
+ source
+ language
+ chunk offsets

Qdrant provides:
- persistent vector storage
- cosine similarity search
- metadata filtering
- payload indexing
- HNSW-based ANN search

## Hybrid Retrieval

The retrieval pipeline combines:

1. multilingual dense retrieval using E5 + Qdrant;
2. lexical BM25 retrieval;
3. Reciprocal Rank Fusion (RRF);
4. multilingual cross-encoder reranking.

This architecture combines semantic matching with exact
keyword/identifier retrieval.

Retrieval is optimized for candidate recall, while the
reranker improves final precision.