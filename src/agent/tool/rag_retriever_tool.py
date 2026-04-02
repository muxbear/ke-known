from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agent.dao.ke_known_dao.ke_known_dao import KeKnownDao

class RAGRetrieverToolArgs(BaseModel):
    db_name: str = Field(description="知识库名称")
    collection_name: str = Field(description="知识库中集合的名称")
    question: str = Field(description="用户的问题")

class RAGRetrieverTool(BaseTool):
    name: str = "RAGRetrieverTool"
    description: str = "RAG 检索工具，用于从知识库中检索相关的知识"

    args_schema: type[BaseModel] = RAGRetrieverToolArgs

    def _run(self, db_name: str, collection_name: str, question: str) -> str:
        ke_known_dao = KeKnownDao(db_name=db_name)
        result = ke_known_dao.hybrid(collection_name=collection_name, question=question)
        return f"检索结果：{result}"


