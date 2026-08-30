# Enterprise Research Agent

An evidence-grounded enterprise research agent that plans a research task, collects web evidence, executes research subtasks, and produces a cited report.

## Phase 1 capabilities

- `POST /research`: execute a complete research run
- `GET /runs/{run_id}`: retrieve the plan, task states, evidence, and report
- Evidence deduplication and traceable evidence IDs
- Tool timeout and retry handling

## Run locally

1. Copy `.env.example` to `.env` and configure `LLM_API_KEY` and `TAVILY_API_KEY`.
2. Install dependencies: `pip install -e ".[dev]"`
3. Start the API: `uvicorn app.main:app --reload`
4. Open `http://127.0.0.1:8000/docs`.

Example request:

```json
POST /research
{"query": "分析某企业的产品、竞争格局和主要风险"}
```

The run store is intentionally in memory in Phase 1. PostgreSQL and Redis will replace it in the persistence phase.
