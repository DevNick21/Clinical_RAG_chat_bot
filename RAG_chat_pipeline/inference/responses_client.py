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
from typing import Any, Iterator, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
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

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """True token-by-token streaming via the Responses API.

        Yields ChatGenerationChunk for each `response.output_text.delta`
        event so callers see content arriving as the model produces it,
        instead of one big chunk after `_generate` returns. Reasoning
        summary deltas are accumulated internally and stashed on
        `last_reasoning_summaries` once the stream completes, so the
        audit layer keeps working unchanged.
        """
        # Reset per-call state. Important when the same model instance is
        # used across many requests in a long-running Flask process.
        reasoning_buffer: List[str] = []
        current_reasoning = ""
        response_id: Optional[str] = None

        stream = self.client.responses.create(
            model=self.model,
            input=self._messages_to_input(messages),
            reasoning={"effort": self.reasoning_effort, "summary": "auto"},
            max_output_tokens=self.max_output_tokens,
            stream=True,
        )

        for event in stream:
            event_type = getattr(event, "type", "")

            # Output text deltas — yield to the caller as they arrive.
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=delta)
                    )
                    if run_manager is not None:
                        run_manager.on_llm_new_token(delta, chunk=chunk)
                    yield chunk

            # Reasoning summary deltas — accumulate; flush to buffer on done.
            elif event_type == "response.reasoning_summary_text.delta":
                current_reasoning += getattr(event, "delta", "") or ""

            elif event_type == "response.reasoning_summary_text.done":
                if current_reasoning:
                    reasoning_buffer.append(current_reasoning)
                    current_reasoning = ""

            # Final event — capture the response id for audit cross-ref.
            elif event_type == "response.completed":
                resp = getattr(event, "response", None)
                if resp is not None:
                    response_id = getattr(resp, "id", None)
                    # Belt-and-braces: if the streamed reasoning items
                    # weren't captured via deltas (older SDK behaviour),
                    # extract them from the final response payload.
                    if not reasoning_buffer:
                        reasoning_buffer = self._extract_reasoning_summaries(resp)

        # Flush any unterminated reasoning (defensive — shouldn't happen
        # with a well-formed stream).
        if current_reasoning:
            reasoning_buffer.append(current_reasoning)

        # Make the trace available to the caller (audit layer in
        # clinical_rag.chat_stream() reads these after the stream
        # completes).
        self.last_reasoning_summaries = reasoning_buffer
        self.last_response_id = response_id
