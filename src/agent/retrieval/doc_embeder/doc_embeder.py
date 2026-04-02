from langchain_core.documents import Document


class DocEmbeder(object):
    """文本嵌入器"""

    def embed_docs(self, docs: list[Document]):
        """嵌入文本接口"""
        pass