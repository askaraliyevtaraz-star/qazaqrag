import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from rag.chunking import Chunk, load_and_chunk_directory
from rag.embeddings import EmbeddingModel
from rag.evaluation import (
    GoldenQuery,
    RankedChunk,
    load_golden_queries,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag.hybrid import reciprocal_rank_fusion
from rag.lexical import BM25Retriever
from rag.reranking import MultilingualReranker
from rag.vector_store import QdrantVectorStore

DATA_DIR = Path("data/demo")
GOLDEN_PATH = Path("evaluation/golden_queries.jsonl")
REPORT_DIR = Path("reports")


class RetrievalBenchmark:
    def __init__(
        self,
        chunks: list[Chunk],
        enable_reranker: bool,
    ) -> None:
        self.chunks = chunks

        print("Loading embedding model...")
        self.embedding_model = EmbeddingModel()

        self.vector_store = QdrantVectorStore()

        self.bm25 = BM25Retriever(chunks)

        self.reranker = None

        if enable_reranker:
            print("Loading reranker...")
            self.reranker = MultilingualReranker()

    def dense(
        self,
        query: str,
        top_k: int,
    ) -> list[RankedChunk]:
        embedding = self.embedding_model.embed_query(query)

        results = self.vector_store.search(
            query_embedding=embedding,
            top_k=top_k,
        )

        return [
            RankedChunk(
                chunk=result.chunk,
                score=result.score,
            )
            for result in results
        ]

    def lexical(
        self,
        query: str,
        top_k: int,
    ) -> list[RankedChunk]:
        results = self.bm25.search(
            query=query,
            top_k=top_k,
        )

        return [
            RankedChunk(
                chunk=result.chunk,
                score=result.score,
            )
            for result in results
        ]

    def hybrid(
        self,
        query: str,
        top_k: int,
        candidate_k: int = 10,
    ) -> list[RankedChunk]:
        embedding = self.embedding_model.embed_query(query)

        dense_results = self.vector_store.search(
            query_embedding=embedding,
            top_k=candidate_k,
        )

        lexical_results = self.bm25.search(
            query=query,
            top_k=candidate_k,
        )

        fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=lexical_results,
            top_k=top_k,
        )

        return [
            RankedChunk(
                chunk=result.chunk,
                score=result.rrf_score,
            )
            for result in fused
        ]

    def reranked(
        self,
        query: str,
        top_k: int,
        candidate_k: int = 10,
        rerank_k: int = 8,
    ) -> list[RankedChunk]:
        if self.reranker is None:
            raise RuntimeError("Reranker is disabled")

        embedding = self.embedding_model.embed_query(query)

        dense_results = self.vector_store.search(
            query_embedding=embedding,
            top_k=candidate_k,
        )

        lexical_results = self.bm25.search(
            query=query,
            top_k=candidate_k,
        )

        fused = reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=lexical_results,
            top_k=rerank_k,
        )

        reranked = self.reranker.rerank(
            query=query,
            candidates=fused,
            top_k=top_k,
        )

        return [
            RankedChunk(
                chunk=result.candidate.chunk,
                score=result.rerank_score,
            )
            for result in reranked
        ]


def evaluate_query(
    method_name: str,
    benchmark: RetrievalBenchmark,
    query: GoldenQuery,
    corpus: list[Chunk],
) -> dict:
    start = time.perf_counter()

    if method_name == "dense":
        results = benchmark.dense(
            query.query,
            top_k=5,
        )

    elif method_name == "bm25":
        results = benchmark.lexical(
            query.query,
            top_k=5,
        )

    elif method_name == "hybrid":
        results = benchmark.hybrid(
            query.query,
            top_k=5,
            candidate_k=10,
        )

    elif method_name == "reranked":
        results = benchmark.reranked(
            query.query,
            top_k=5,
            candidate_k=10,
            rerank_k=8,
        )

    else:
        raise ValueError(f"Unknown method: {method_name}")

    latency_ms = (time.perf_counter() - start) * 1000

    row = {
        "method": method_name,
        "query_id": query.query_id,
        "query": query.query,
        "query_type": query.query_type,
        "query_language": query.query_language,
        "answerable": query.answerable,
        "latency_ms": latency_ms,
        "top1_source": results[0].chunk.source if results else "",
        "top1_score": results[0].score if results else None,
    }

    if query.answerable:
        row.update(
            {
                "recall@1": recall_at_k(
                    results,
                    corpus,
                    query,
                    1,
                ),
                "recall@3": recall_at_k(
                    results,
                    corpus,
                    query,
                    3,
                ),
                "recall@5": recall_at_k(
                    results,
                    corpus,
                    query,
                    5,
                ),
                "mrr@5": reciprocal_rank(
                    results,
                    query,
                    5,
                ),
                "ndcg@5": ndcg_at_k(
                    results,
                    corpus,
                    query,
                    5,
                ),
            }
        )

    else:
        row.update(
            {
                "recall@1": None,
                "recall@3": None,
                "recall@5": None,
                "mrr@5": None,
                "ndcg@5": None,
            }
        )

    return row


def mean_metric(
    rows: list[dict],
    key: str,
) -> float:
    values = [float(row[key]) for row in rows if row[key] is not None]

    if not values:
        return 0.0

    return statistics.mean(values)


def build_summary(
    rows: list[dict],
) -> dict:
    summary = {}

    methods = sorted({row["method"] for row in rows})

    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]

        answerable_rows = [row for row in method_rows if row["answerable"]]

        summary[method] = {
            "recall@1": mean_metric(
                answerable_rows,
                "recall@1",
            ),
            "recall@3": mean_metric(
                answerable_rows,
                "recall@3",
            ),
            "recall@5": mean_metric(
                answerable_rows,
                "recall@5",
            ),
            "mrr@5": mean_metric(
                answerable_rows,
                "mrr@5",
            ),
            "ndcg@5": mean_metric(
                answerable_rows,
                "ndcg@5",
            ),
            "median_latency_ms": statistics.median([row["latency_ms"] for row in method_rows]),
        }

    return summary


def build_type_breakdown(
    rows: list[dict],
) -> dict:
    output = {}

    answerable_types = sorted({row["query_type"] for row in rows if row["answerable"]})

    methods = sorted({row["method"] for row in rows})

    for query_type in answerable_types:
        output[query_type] = {}

        for method in methods:
            subset = [
                row
                for row in rows
                if (
                    row["query_type"] == query_type
                    and row["method"] == method
                    and row["answerable"]
                )
            ]

            output[query_type][method] = {
                "recall@1": mean_metric(
                    subset,
                    "recall@1",
                ),
                "recall@3": mean_metric(
                    subset,
                    "recall@3",
                ),
                "recall@5": mean_metric(
                    subset,
                    "recall@5",
                ),
                "mrr@5": mean_metric(
                    subset,
                    "mrr@5",
                ),
                "ndcg@5": mean_metric(
                    subset,
                    "ndcg@5",
                ),
                "median_latency_ms": (
                    statistics.median([row["latency_ms"] for row in subset]) if subset else 0.0
                ),
            }

    return output


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="Evaluate dense, BM25 and hybrid retrieval only.",
    )

    args = parser.parse_args()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks = load_and_chunk_directory(
        DATA_DIR,
        chunk_size=60,
        overlap=15,
    )

    queries = load_golden_queries(GOLDEN_PATH)

    print(f"Loaded {len(chunks)} chunks")

    print(f"Loaded {len(queries)} golden queries")

    benchmark = RetrievalBenchmark(
        chunks=chunks,
        enable_reranker=(not args.skip_reranker),
    )

    methods = [
        "dense",
        "bm25",
        "hybrid",
    ]

    if not args.skip_reranker:
        methods.append("reranked")

    print()
    print("Warming up models...")

    benchmark.dense(
        "warm up query",
        top_k=3,
    )

    if benchmark.reranker is not None:
        benchmark.reranked(
            "warm up query",
            top_k=3,
            candidate_k=5,
            rerank_k=5,
        )

    rows = []

    for method in methods:
        print()
        print("=" * 80)
        print("METHOD:", method)

        for query in queries:
            row = evaluate_query(
                method_name=method,
                benchmark=benchmark,
                query=query,
                corpus=chunks,
            )

            rows.append(row)

            print(
                query.query_id,
                f"[{query.query_type}]",
                "→",
                row["top1_source"],
                f"({row['latency_ms']:.1f} ms)",
            )

    summary = build_summary(rows)

    type_breakdown = build_type_breakdown(rows)

    csv_path = REPORT_DIR / "retrieval_per_query.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    summary_path = REPORT_DIR / "retrieval_summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    type_path = REPORT_DIR / "retrieval_by_type.json"

    type_path.write_text(
        json.dumps(
            type_breakdown,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("SUMMARY")
    print()

    header = f"{'method':<12}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}{'NDCG':>8}{'lat(ms)':>12}"

    print(header)

    for method, metrics in summary.items():
        print(
            f"{method:<12}"
            f"{metrics['recall@1']:>8.3f}"
            f"{metrics['recall@3']:>8.3f}"
            f"{metrics['recall@5']:>8.3f}"
            f"{metrics['mrr@5']:>8.3f}"
            f"{metrics['ndcg@5']:>8.3f}"
            f"{metrics['median_latency_ms']:>12.1f}"
        )

    print()
    print("=" * 80)
    print("BREAKDOWN BY QUERY TYPE")
    print()

    for query_type, methods_data in type_breakdown.items():
        print(query_type.upper())

        for method, metrics in methods_data.items():
            print(
                f"  {method:<12}"
                f" R@3={metrics['recall@3']:.3f}"
                f" MRR={metrics['mrr@5']:.3f}"
                f" NDCG={metrics['ndcg@5']:.3f}"
            )

        print()

    print("Saved:", csv_path)
    print("Saved:", summary_path)
    print("Saved:", type_path)


if __name__ == "__main__":
    main()
