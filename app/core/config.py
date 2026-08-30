import os

from dotenv import load_dotenv


load_dotenv()


LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.6")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")