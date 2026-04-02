from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph

from agent.model.llm import deepseek_lm


class RAGState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def llm_node(state: RAGState):
    return {"messages": [deepseek_lm.invoke(state["messages"])]}

graph_builder = StateGraph(RAGState)

graph_builder.add_node(llm_node)

graph_builder.add_edge(START, "llm_node")
graph_builder.add_edge("llm_node", END)

corrective_graph = graph_builder.compile()

if __name__ == "__main__":
    response = corrective_graph.invoke({
        "messages": [
            {"role": "user", "content": "请介绍一下Python语言"}
        ]
    })
    print(response)
