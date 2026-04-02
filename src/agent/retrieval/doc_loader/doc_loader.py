from langchain_core.documents import Document

class DocLoader:
    """加载文档类接口"""

    def load_doc(self, path) -> list[Document]:
        pass

