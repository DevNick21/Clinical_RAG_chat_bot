"""LangChain chat model that targets the OpenAI/Azure Responses API.

Captures reasoning summaries (`reasoning.summary` items in the response output)
and exposes them via `last_reasoning_summaries` so callers can persist them
for clinical audit. Conforms to `BaseChatModel` so existing LangChain chains
like `create_stuff_documents_chain` keep working unchanged.

Note: OpenAI does not expose raw chain-of-thought tokens for the gpt-5 /
o-series. The Responses API instead returns model-generated summaries of
each reasoning step (requested via `reasoning.summary = "auto"`). These are
the most reliable trace we can get for audit purposes — more trustworthy
than asking the model to "explain its reasoning" in-prompt, which it can
fabricate.
"""
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ResponsesAPIChatModel(BaseChatModel):
    """Chat model backed by `/v1/responses`."""

    client: Any
    model: str
    reasoning_effort: str = "low"
    max_output_tokens: int = 2048

    # Populated after each `_generate` call so callers can read the latest
    # reasoning trace for audit logging. Not thread-safe across concurrent
    # invocations on the same instance — fine for a single-user chat flow.
    last_reasoning_summaries: List[str] = Field(default_factory=list)
    last_response_id: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "azure-responses-api"

    @staticmethod
    def _messages_to_input(messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to Responses API input items."""
        out = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "user"
            out.append({"role": role, "content": str(msg.content)})
        return out

    @staticmethod
    def _extract_reasoning_summaries(response: Any) -> List[str]:
        """Pull text from reasoning items in the Responses API output."""
        summaries: List[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "reasoning":
                continue
            for s in getattr(item, "summary", []) or []:
                text = getattr(s, "text", None)
                if text:
                    summaries.append(text)
        return summaries

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self.client.responses.create(
            model=self.model,
            input=self._messages_to_input(messages),
            reasoning={"effort": self.reasoning_effort, "summary": "auto"},
            max_output_tokens=self.max_output_tokens,
        )

        # Stash for audit retrieval by the caller
        self.last_reasoning_summaries = self._extract_reasoning_summaries(response)
        self.last_response_id = getattr(response, "id", None)

        content = getattr(response, "output_text", "") or ""
        ai_msg = AIMessage(
            content=content,
            additional_kwargs={
                "reasoning_summaries": self.last_reasoning_summaries,
                "response_id": self.last_response_id,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])
