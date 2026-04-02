from langchain_core.documents import Document

class RagIndexService(object):
    """RAG 创建索引接口"""

    def rag_index(self, path: str):
        """RAG过程中，给文档建立索引过程"""
        pass

    def load_doc(self, path: str) -> list[Document]:
        """加载文档"""
        pass

    def split_doc(self, docs: list[Document]) -> list[Document]:
        """切分文档"""
        pass

    def embed_doc(self, docs: list[Document]) -> list[list[float]]:
        """向量化文档"""
        pass

    def store_doc(self, docs: list[Document]) -> list[str]:
        """存储文档"""
        pass