from langchain.agents import create_agent

from agent.model.llm import deepseek_llm

graph = create_agent(
    model=deepseek_llm,
    tools=[]
)