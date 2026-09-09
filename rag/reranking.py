from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from rag.hybrid import (
    HybridSearchResult,
)

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass
class RerankedResult:
    candidate: HybridSearchResult
    rerank_score: float


class MultilingualReranker:
    def __init__(
        self,
        model_name: str = (DEFAULT_RERANKER_MODEL),
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(
            "Reranker device:",
            self.device,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.to(self.device)

        self.model.eval()

    def rerank(
        self,
        query: str,
        candidates: list[HybridSearchResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        if not candidates:
            return []

        pairs = [
            [
                query,
                candidate.chunk.text,
            ]
            for candidate in candidates
        ]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits.view(-1).float().cpu()

        scores = logits.tolist()

        results = [
            RerankedResult(
                candidate=candidate,
                rerank_score=float(score),
            )
            for candidate, score in zip(
                candidates,
                scores,
                strict=True,
            )
        ]

        results.sort(
            key=lambda result: result.rerank_score,
            reverse=True,
        )

        return results[:top_k]
