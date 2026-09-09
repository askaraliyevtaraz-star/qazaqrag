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