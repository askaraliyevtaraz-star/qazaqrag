import operator
from typing import Annotated, Literal, TypedDict

RouteName = Literal[
    "rag",
    "identifier",
    "sources",
]


class AgentSource(TypedDict):
    citation_id: str
    source: str
    chunk_id: str
    score: float
    text: str


class AgentState(TypedDict, total=False):
    question: str

    effective_query: str

    route: RouteName

    retrieval_attempts: int
    max_retrieval_attempts: int

    top_score: float | None

    context: str

    sources: list[AgentSource]

    answer: str

    trace: Annotated[
        list[str],
        operator.add,
    ]
