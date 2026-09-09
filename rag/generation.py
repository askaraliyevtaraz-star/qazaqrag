from typing import Protocol

from openai import OpenAI


class LLMGenerator(Protocol):
    @property
    def model_name(self) -> str: ...

    def generate(
        self,
        question: str,
        context: str,
    ) -> str: ...


class StubGenerator:
    @property
    def model_name(self) -> str:
        return "stub"

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        del question
        del context

        return "LLM generation is disabled. Retrieved sources are returned with the response."


class OpenAIGenerator:
    def __init__(
        self,
        api_key: str,
        model: str,
    ) -> None:
        self.client = OpenAI(api_key=api_key)

        self._model_name = model

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        instructions = """
You are a multilingual knowledge-base assistant.

Use only the supplied context.

Do not use outside knowledge.

If the context does not contain enough
information to answer the question,
say that the available documents do not
contain the answer.

Always cite supporting context using
the identifiers [S1], [S2], etc.

Answer in the language used by the user.
""".strip()

        prompt = f"""
QUESTION:
{question}

CONTEXT:
{context}
""".strip()

        response = self.client.responses.create(
            model=self._model_name,
            instructions=instructions,
            input=prompt,
        )

        return response.output_text.strip()
