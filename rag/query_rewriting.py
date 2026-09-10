from openai import OpenAI


class QueryRewriter:
    def __init__(
        self,
        provider: str,
        api_key: str | None,
        model: str,
    ) -> None:
        self.provider = provider
        self.model = model

        self.client = None

        if provider == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required for OpenAI query rewriting.")

            self.client = OpenAI(api_key=api_key)

    def rewrite(
        self,
        original_question: str,
        current_query: str,
    ) -> str:
        if self.provider == "stub":
            return original_question.strip()

        instructions = """
Rewrite the user question into a concise
standalone search query for document retrieval.

Do not answer the question.

Preserve:
- exact identifiers;
- numbers;
- names;
- important technical terms.

Return only the rewritten search query.
""".strip()

        prompt = f"""
ORIGINAL QUESTION:
{original_question}

CURRENT SEARCH QUERY:
{current_query}
""".strip()

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
        )

        rewritten = response.output_text.strip()

        if not rewritten:
            return current_query

        return rewritten
