from langchain_openai import ChatOpenAI

from agent.config.config import DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY, DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY

# DeepSeek 大语言模型
deepseek_lm = ChatOpenAI(base_url=DEEPSEEK_BASE_URL, api_key=DEEPSEEK_API_KEY, model='deepseek-chat')

# Dashscope 大语言模型
dashscope_lm = ChatOpenAI(base_url=DASHSCOPE_BASE_URL, api_key=DASHSCOPE_API_KEY, model='qwen-plus')