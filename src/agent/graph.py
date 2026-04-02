from langchain.agents import create_agent

from agent.model.llm import deepseek_lm
from agent.tool.rag_retriever_tool import RAGRetrieverTool

rag_tool = RAGRetrieverTool()

graph = create_agent(
    model=deepseek_lm,
    tools=[rag_tool],
    system_prompt="你是一个个人助手智能体，尽量调用工具来回答我的问题"
)

if __name__ == '__main__':
    response = graph.invoke(
        {
            "messages": [
                {"role": "user", "content": "什么是Java语言？"}
            ],
            "db_name": "ke_known_db",
            "collection_name": "ke_known_collection",
        }
    )
    print(response)