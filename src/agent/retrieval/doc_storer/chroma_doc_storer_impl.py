import os

from langchain_chroma import Chroma
from langchain_core.documents import Document

from agent.retrieval.doc_storer.doc_storer import DocStorer


class ChromaDocStorerImpl(DocStorer):

    persist_directory = "./chroma_langchain_db"

    collection_name = "ke_known_db"  # TODO

    chroma: Chroma = None

    def __init__(self, embeddings):
        static_root = os.getenv("STATIC_ROOT")
        vdb_path = os.getenv("VDB_PATH")

        self.persist_directory = f'{static_root}{vdb_path}{self.collection_name}'
        print(f"Chroma 数据存储目录：{self.persist_directory}")

        self.embeddings = embeddings
        self.__get_vector_store()

    def doc_store(self, docs: list[Document]) -> list[str]:
        return self.chroma.add_documents(docs)

    def delete_store(self):
        self.chroma.delete_collection()

    def __get_vector_store(self):
        self.chroma = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,  # Where to save data locally, remove if not necessary
            create_collection_if_not_exists=True
        )
