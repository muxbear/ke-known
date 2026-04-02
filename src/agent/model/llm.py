from langchain_openai import ChatOpenAI

from agent.config.config import DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY

# DeepSeek 大语言模型
deepseek_llm = ChatOpenAI(base_url=DEEPSEEK_BASE_URL, api_key=DEEPSEEK_API_KEY, model='deepseek-chat')
