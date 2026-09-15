from openai import APIError, APIStatusError, AsyncOpenAI, AuthenticationError, RateLimitError

from app.core.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_REASONING_EFFORT,
)


class LLMClient:
    def __init__(self):
        client_kwargs = {
            "api_key": LLM_API_KEY,
        }

        if LLM_BASE_URL:
            client_kwargs["base_url"] = LLM_BASE_URL

        self.client = AsyncOpenAI(**client_kwargs)
        self.model = LLM_MODEL

    @staticmethod
    def _safe_provider_error(error: APIError) -> RuntimeError:
        """Do not expose provider payloads, which can include key fragments."""
        if isinstance(error, AuthenticationError):
            return RuntimeError("LLM authentication failed. Check LLM_API_KEY and LLM_BASE_URL.")
        if isinstance(error, RateLimitError):
            return RuntimeError("LLM request was rate limited. Check provider quota and retry later.")
        if isinstance(error, APIStatusError):
            return RuntimeError(f"LLM provider request failed with HTTP {error.status_code}.")
        return RuntimeError("LLM provider request failed. Check provider connectivity and configuration.")
        
    async def chat(self, messages: list[dict]) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        except APIError as error:
            raise self._safe_provider_error(error) from error
        return response.choices[0].message.content or ""


    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ):
        request_args = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        if LLM_REASONING_EFFORT:
            request_args["reasoning_effort"] = LLM_REASONING_EFFORT

        try:
            response = await self.client.chat.completions.create(**request_args)
        except APIError as error:
            raise self._safe_provider_error(error) from error
        return response
    
llm_client = LLMClient()

