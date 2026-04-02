from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from agent.retrieval.doc_loader.doc_loader import DocLoader


class WebDocLoaderImpl(DocLoader):

    def load_doc(self, path: str = "https://news.qq.com/rain/a/20260112A06LOJ00") -> list[Document]:
        """从 web 加载文档"""
        loader = WebBaseLoader(path)
        loader.requests_kwargs = {'verify': False}
        return loader.load()