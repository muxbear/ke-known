from langchain_core.documents import Document

from agent.retrieval.doc_loader.doc_loader import DocLoader


class LocalDocLoaderImpl(DocLoader):

    def load(self, path) -> list[Document]:
        pass