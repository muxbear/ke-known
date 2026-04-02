import os

from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction

from agent.retrieval.doc_storer.doc_storer import DocStorer


class MilvusDocStorerImpl(DocStorer):
    """ Milvus 的 document storer 的实现（目前报错 TODO） """

    def __init__(self):
        self.MILVUS_URI = os.environ.get("MILVUS_URI")
        self.MILVUS_USER = os.environ.get("MILVUS_USER")
        self.MILVUS_PASSWORD = os.environ.get("MILVUS_PASSWORD")

        self.storer = Milvus(
            embedding_function=None,
            collection_name="ke_known_collection",
            builtin_function=BM25BuiltInFunction(
                output_field_names="sparse",
            ),
            vector_field=['sparse'],
            consistency_level="Strong",
            auto_id=True,
            connection_args={
                "uri": self.MILVUS_URI,
                "user": self.MILVUS_USER,
                "password": self.MILVUS_PASSWORD,
                "db_name": "ke_known_db",
            }
        )

    def doc_store(self, docs: list[Document]) -> list[str]:
        """保存向量"""
        return self.storer.add_documents(docs)


    def delete_store(self):
        """删除向量库"""
        pass

if __name__ == '__main__':
    milvus_doc_storer = MilvusDocStorerImpl()

    test_docs = [
        Document(
            page_content="""
            测试数据1
            """,
            metadata={
                'source': 'https://news.qq.com/rain/a/20260325A01O1K00',
                'owner': 'yahoo',
                'date': '2026-03-24 11:40'
            }
        ),
        Document(
            page_content="""
            测试数据2        
            """,
            metadata={
                'source': 'https://news.qq.com/rain/a/20260324A03FEI00',
                'owner': 'yahoo',
                'date': '2026-03-25 07:39'
            }
        )
    ]

    milvus_doc_storer.doc_store(test_docs)