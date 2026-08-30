from openai import AsyncOpenAI

from app.core.config import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL


class LLMClient:
    def __init__(self):
        client_kwargs = {
            "api_key": LLM_API_KEY,
        }

        if LLM_BASE_URL:
            client_kwargs["base_url"] = LLM_BASE_URL

        self.client = AsyncOpenAI(**client_kwargs)
        self.model = LLM_MODEL
        
    async def chat(self, messages: list[dict]) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
       

        return response.choices[0].message.content or ""


    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ):
        response = await self.client.chat.completions.create(
    model=self.model,
    messages=messages,
    tools=tools,
    tool_choice="auto",
    reasoning_effort="none",
)
    
        return response
    
llm_client = LLMClient()

