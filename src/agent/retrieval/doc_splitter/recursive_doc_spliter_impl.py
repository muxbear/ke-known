from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.retrieval.doc_splitter.doc_splitter import DocSplitter


class RecursiveDocSplitterImpl(DocSplitter):
    """递归文档切分器"""

    def split_doc(self, docs: list[Document]) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(docs)