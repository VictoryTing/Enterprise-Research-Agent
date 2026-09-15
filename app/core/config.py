import os

from dotenv import load_dotenv


load_dotenv()


LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.6")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
# Leave empty for OpenAI-compatible providers that do not support this
# OpenAI-specific parameter, such as DeepSeek.
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT")
# Set false when web-search credits are unavailable. The agent will then use
# only its language-model knowledge and clearly disclose that limitation.
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
# The adapter keeps the Agent independent of a specific web-search vendor.
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "tavily").strip().lower()
SEARCH_TIMEOUT_SECONDS = int(os.getenv("SEARCH_TIMEOUT_SECONDS", "10"))
# Balanced-mode budgets. They cap repeated tool calls and prevent a simple
# prompt from producing dozens of near-duplicate search results.
MAX_RESEARCH_TASKS = int(os.getenv("MAX_RESEARCH_TASKS", "3"))
MAX_LLM_ITERATIONS = int(os.getenv("MAX_LLM_ITERATIONS", "3"))
MAX_SEARCHES_PER_TASK = int(os.getenv("MAX_SEARCHES_PER_TASK", "2"))
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "3"))
RAG_MAX_EVIDENCE = int(os.getenv("RAG_MAX_EVIDENCE", "3"))
RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "3600"))
# ---------------------------------------------------------
# Reranker configuration
# ---------------------------------------------------------
RERANKER_ENABLED = (
    os.getenv("RERANKER_ENABLED", "false")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "BAAI/bge-reranker-v2-m3",
)

RERANK_CANDIDATE_K = int(
    os.getenv("RERANK_CANDIDATE_K", "10")
)