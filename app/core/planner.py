import json
from typing import List

from pydantic import BaseModel, ValidationError

from app.core.llm import llm_client


class ResearchPlan(BaseModel):
    goal: str
    tasks: List[str]


async def create_research_plan(user_query: str) -> ResearchPlan:
    """
    Create a research plan using an LLM.
    """

    messages = [
        {
            "role": "system",
            "content": """
You are a research planning agent.

Given a user's research question, create a concise and actionable research plan.

Requirements:
1. Clearly define the research goal.
2. Break the research into 3-5 concrete tasks.
3. Each task must be specific and actionable.
4. Tasks should focus on information that needs to be searched,
   compared, verified, or analyzed.
5. Avoid redundant tasks.
6. Do not answer the research question itself.
7. Return ONLY valid JSON.
8. Do NOT use Markdown code fences.

Return JSON with exactly this structure:

{
    "goal": "research goal",
    "tasks": [
        "task 1",
        "task 2",
        "task 3"
    ]
}
""",
        },
        {
            "role": "user",
            "content": user_query,
        },
    ]

    response = await llm_client.chat(messages)

    try:
        data = json.loads(response)
        return ResearchPlan(**data)

    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(
            f"Invalid research plan returned by LLM: {response}"
        ) from e


if __name__ == "__main__":
    import asyncio

    query = "请搜索 2026 年 AI Agent 最新发展"

    plan = asyncio.run(create_research_plan(query))

    print("========== Research Plan ==========")
    print(f"Goal: {plan.goal}")
    print()

    for i, task in enumerate(plan.tasks, start=1):
        print(f"{i}. {task}")