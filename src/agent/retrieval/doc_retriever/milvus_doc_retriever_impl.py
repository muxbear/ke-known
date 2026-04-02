import os

from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import MilvusException, IndexType
from pymilvus.client.types import MetricType

from agent.model.vm import zhipuai_em
from agent.retrieval.doc_storer.doc_storer import DocStorer


class MilvusDocStorerImpl(DocStorer):

    milvus_storer: Milvus = None

    def __init__(self):
        self.MILVUS_URI = os.environ.get("MILVUS_URI")
        self.MILVUS_USER = os.environ.get("MILVUS_USER")
        self.MILVUS_PASSWORD = os.environ.get("MILVUS_PASSWORD")

    def connect_collection(self, db_name: str, collection_name: str):
        index_params = [
            {
                "field_name": "sparse",
                "index_name": "sparse_inverted_index",
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",  # 度量类型
                "params": {
                    "inverted_index_algo": "DAAT_MAXSCORE",  # 构建和查询索引的算法（DAAT_MAXSCORE 相关性评分的最高得分）
                    "bm25_k1": 1.6,  # 控制词频的饱和度，取值范围：1.2 ~ 2.0，数值越大，术语词频在文档中的排名的重要性越高
                    "bm25_b": 0.75,  # 控制文档长度的标准化程度，取值范围：0 ~ 1，值为1表示不进行归一化，值为0表示完全进行归一化
                }
            },
            {
                "field_name": "dense",
                "index_name": "dense_vector_index",
                "index_type": IndexType.HNSW,  # 一种图的近似最近邻搜索算法
                "metric_type": MetricType.IP,  # 相似度的度量方式：1-IP(内积，包含了余弦相似度) 2-L2(欧式距离)
                "params": {
                    "M": 16,  # M: 紧邻的节点数(值：4 ~ 64 之间)
                    "efConstruction": 100  # 即搜索范围 (50 ~ 200 之间）值越大，搜索时间越长，搜索范围越大
                }
            }
        ]

        try:
            # 如果 collection 没有会自动创建，否则就会加载
            # 注意，langchain-milvus 的版本只有 2.6.9 报错：TypeError: pymilvus.milvus_client.index.IndexParams.add_index() got multiple values for keyword argument 'field_name'
            # 2.6.10, 2.6.11 均报错：
            self.milvus_storer = Milvus(
                embedding_function=zhipuai_em,
                collection_name=collection_name,
                vector_field=["sparse", "dense"],
                builtin_function=BM25BuiltInFunction(
                    output_field_names="sparse"  # 明确指定输出字段名
                ),
                index_params=index_params,
                consistency_level="Strong",
                auto_id=True,
                connection_args={
                    "uri": self.MILVUS_URI,
                    "user": self.MILVUS_USER,
                    "password": self.MILVUS_PASSWORD,
                    "db_name": db_name
                }
            )

            print(f"在数据库 '{db_name}' 连接集合 '{collection_name}' 完成")
        except MilvusException as e:
            print(f"连接时发生错误: {e}")
            raise

    def hybrid(self, question: str):
        """混合检索"""
        return self.milvus_storer.similarity_search_with_score(
            query=question,
            k=3,
            ranker_type="rrf",
            ranker_params={"k": 60}
        )

    def add_docs(self, documents: list[Document]):
        try:
            # 使用下面的插入报错，问题尚未解决
            return self.milvus_storer.add_documents(documents)
        except MilvusException as e:
            print(f"连接时发生错误: {e}")
            raise

if __name__ == '__main__':
    milvus_doc_storer = MilvusDocStorerImpl()
    milvus_doc_storer.connect_collection(db_name="ke_known_db", collection_name="ke_known_collection")

#     test_docs = [
#         Document(
#             page_content="""
# As of November 1st, 2021 Yahoo’s suite of services will no longer be accessible from mainland China.
#         """,
#             metadata={
#                 'source': 'https://news.qq.com/rain/a/20260325A01O1K00',
#                 'owner': 'yahoo',
#                 'date': '2026-03-24 11:40'
#             }
#         ),
#         Document(
#             page_content="""
# Yahoo products and services remain unaffected in all other global locations. We thank you for your support and readership.
#             """,
#             metadata={
#                 'source': 'https://news.qq.com/rain/a/20260324A03FEI00',
#                 'owner': 'yahoo',
#                 'date': '2026-03-25 07:39'
#             }
#         )
#     ]
#
#     result = milvus_doc_storer.add_docs(documents=test_docs)
#     print(result)

    result = milvus_doc_storer.hybrid("Java 语言")
    print(result)

    # result = milvus_doc_storer.milvus_storer.similarity_search(
    #     query="yahoo",
    #     k=2
    # )
    # print(result)