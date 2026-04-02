from langchain_core.documents import Document

from agent.model.vm import dashscope_em
from agent.retrieval.doc_embeder.doc_embeder import DocEmbeder
from agent.retrieval.doc_embeder.text_embedding_doc_embedder_impl import TextEmbeddingDocEmbedder
from agent.retrieval.doc_loader.doc_loader import DocLoader
from agent.retrieval.doc_loader.local_doc_loader_impl import LocalDocLoaderImpl
from agent.retrieval.doc_splitter.doc_splitter import DocSplitter
from agent.retrieval.doc_splitter.recursive_doc_spliter_impl import RecursiveDocSplitterImpl
from agent.retrieval.doc_storer.chroma_doc_storer_impl import ChromaDocStorerImpl
from agent.retrieval.doc_storer.doc_storer import DocStorer
from agent.service.rag_index_service.rag_index_service import RagIndexService


class RagIndexServiceImpl(RagIndexService):
    """RAG 创建索引接口"""

    doc_loader: DocLoader = LocalDocLoaderImpl()

    doc_splitter: DocSplitter = RecursiveDocSplitterImpl()

    doc_embeder: DocEmbeder = TextEmbeddingDocEmbedder()

    doc_storer: DocStorer = ChromaDocStorerImpl(dashscope_em)

    def rag_index(self, path: str) -> list[str]:
        """RAG过程中，给文档建立索引过程"""

        # 加载文档
        docs = self.load_doc(path)

        # 文档切片
        chunks = self.split_doc(docs)

        # 向量化
        # embeddings = self.embed_doc(chunks)

        # 向量化存储
        return self.store_doc(chunks)

    def load_doc(self, path: str) -> list[Document]:
        """
        加载文档
        :param path: 资源加载路径
        :return: 文档列表
        """
        return self.doc_loader.load_doc(path)

    def split_doc(self, docs: list[Document]) -> list[Document]:
        """切分文档"""
        return self.doc_splitter.split_doc(docs)

    def embed_doc(self, docs: list[Document]) -> list[list[float]]:
        """向量化文档"""
        return self.doc_embeder.embed_docs(docs)

    def store_doc(self, docs: list[Document]) -> list[str]:
        """存储文档"""
        return self.doc_storer.doc_store(docs)

    def set_doc_loader(self, doc_loader: DocLoader):
        """设置 doc_loader"""
        self.doc_loader = doc_loader

    def set_doc_splitter(self, doc_splitter: DocSplitter):
        """设置 doc_splitter"""
        self.doc_splitter = doc_splitter

    def set_doc_embeder(self, doc_embeder: DocEmbeder):
        """这是 doc_embeder"""
        self.doc_embeder = doc_embeder

    def set_doc_storer(self, doc_storer: DocStorer):
        """设置 doc_storer"""
        self.doc_storer = doc_storer