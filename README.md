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

## Retrieval Evaluation

The retrieval stack is evaluated against a manually labeled
multilingual golden query set.

The benchmark includes:

- semantic queries;
- cross-lingual queries;
- exact identifiers;
- hard-negative documents;
- unanswerable queries.

Metrics:

- Recall@1 / Recall@3 / Recall@5
- MRR@5
- NDCG@5
- median local query latency

Compared systems:

1. BM25
2. Multilingual E5 + Qdrant
3. Dense + BM25 + Reciprocal Rank Fusion
4. Hybrid retrieval + multilingual cross-encoder reranking

                         QUERY
                           │
                           ▼
                        FastAPI
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
         Multilingual E5            BM25
                │                     │
                ▼                     │
             Qdrant                   │
                │                     │
                └─────────┬───────────┘
                          ▼
                         RRF
                          │
                          ▼
                 BGE Reranker
                          │
                          ▼
                     Top-K Context
                          │
                          ▼
                     OpenAI LLM
                          │
                          ▼
                 Answer + Citations