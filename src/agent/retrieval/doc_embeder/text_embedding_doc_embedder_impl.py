from langchain_core.documents import Document

from agent.model.vm import dashscope_em
from agent.retrieval.doc_embeder.doc_embeder import DocEmbeder


class TextEmbeddingDocEmbedder(DocEmbeder):

    def embed_docs(self, docs: list[Document]):
        if len(docs) == 0:
            return

        texts = [doc.page_content for doc in docs]
        embeddings = dashscope_em.embed_documents(texts)
        return embeddings

