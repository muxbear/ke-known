from agent.component.doc_retriever.chroma_doc_retriever_impl import ChromaDocRetriever
from agent.component.doc_retriever.doc_retriever import DocRetriever
from agent.model.vm import dashscope_em
from agent.service.rag_retrieve_service.rag_retrieve_service import RagRetrieveService


class RagRetrieveServiceImpl(RagRetrieveService):

    doc_retriever: DocRetriever = ChromaDocRetriever(dashscope_em)

    def rag_retrieve(self, question: str, session_id: str) -> str:
        # return self.doc_retriever.retrieve_doc(question)
        return self.doc_retriever.retrieve_doc_with_history(question, session_id)

    def set_doc_retriever(self, doc_retriever: DocRetriever) -> None:
        self.doc_retriever = doc_retriever