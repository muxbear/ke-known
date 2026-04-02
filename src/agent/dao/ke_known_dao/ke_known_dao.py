from typing import Any, Union

from pymilvus import DataType, Function, FunctionType, IndexType, AnnSearchRequest, RRFRanker
from pymilvus.client.types import MetricType

from agent.dao.base_milvus_dao import BaseMilvusDao
from agent.model.vm import dashscope_em


class KeKnownDao(BaseMilvusDao):
    collection_name = "ke_known_collection"

    def __init__(self, db_name = "ke_known_db"):
        super().__init__()
        self.client.using_database(db_name)
        print(f"切换到数据库 {db_name}")

    def create_collection(self, collection_name: str, **kwargs):
        """抽象方法：创建集合，子类实现"""
        # 检查 Collection 是否存在
        if self.client.has_collection(collection_name):
            print(f"Collection '{collection_name}' 已存在")
            return

        try:
            ke_known_schema = self.client.create_schema()
            ke_known_schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
            ke_known_schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000, enable_analyzer=True, analyzer_params={'tokenizer': 'jieba', 'filter': ["cnalphanumonly"]})
            ke_known_schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR, nullable=False)

            # 创建 schema
            # 稀疏嵌入函数：从一个字段中读取原始数据（input_field_name=['text'] 对应下面 test_schema.add_field(field_name="text")）
            # 通过BM25算法转换成向量，再把稀疏向量存储到输出字段(output_field_names=['sparse'] 对应下面 test_schema.add_field(field_name="sparse"))
            bm25_function = Function( # 稀疏嵌入函数：从 text 字段中读取原始数据通过 bm25 算法，转换为稀疏向量，再存在 sparse 字段
                name="text_bm25_emb",  # 函数名
                input_field_names=["text"],  #
                output_field_names=["sparse"],
                function_type=FunctionType.BM25
            )
            ke_known_schema.add_function(function=bm25_function)

            ke_known_schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=1024)

            index_params = self.client.prepare_index_params()
            index_params.add_index( # 稀疏向量索引
                field_name="sparse",  # 对字段 sparse 进行索引
                index_name="sparse_inverted_index",
                index_type="SPARSE_INVERTED_INDEX",  # 索引类型
                metric_type="BM25",  # 度量类型
                params={
                    "inverted_index_algo": "DAAT_MAXSCORE",  # 构建和查询索引的算法（DAAT_MAXSCORE 相关性评分的最高得分）
                    "bm25_k1": 1.6,  # 控制词频的饱和度，取值范围：1.2 ~ 2.0，数值越大，术语词频在文档中的排名的重要性越高
                    "bm25_b": 0.75,  # 控制文档长度的标准化程度，取值范围：0 ~ 1，值为1表示不进行归一化，值为0表示完全进行归一化
                }
            )

            index_params.add_index( # 密集向量索引
                field_name="dense",
                index_name="dense_vector_index",
                index_type=IndexType.HNSW,
                metric_type=MetricType.IP,
                params={
                    "M": 16,
                    "efConstruction": 64,
                }
            )

            self.client.create_collection(collection_name=collection_name,
                                          schema=ke_known_schema,
                                          index_params=index_params)

            print(f"使用原生 pymilvus 创建 Collection '{collection_name}' 成功")
        except Exception as e:
            print(f"使用原生 pymilvus 创建 Collection 失败: {e}")

    def get_collection_info(self, collection_name: str) -> dict | None:
        try:
            info = self.client.get_collection_stats(collection_name)
            print(f"Collection 信息: {info}")
            return info
        except Exception as e:
            print(f"获取 Collection 信息失败: {e}")

    def get_collection_desc(self, collection_name: str) -> dict | None:
        """返回表结构"""
        return self.client.describe_collection(collection_name)

    def get_collection_indexes(self, collection_name: str) -> dict | None:
        """获取表索引"""
        return self.client.list_indexes(collection_name)

    def search(self, collection_name: str,
               data: list[str],
               search_params: dict[str, Any]) -> list[str] | None:
        return self.client.search(
            collection_name=collection_name,
            data=data,
            anns_field='sparse',
            limit=3,
            search_params=search_params,
            output_fields=["text"]
        )

    def hybrid(self, collection_name: str, question: str) -> list[list[dict]]:
        """混合检索"""

        # 稀疏向量检索
        search_params_1 = {
            'data': [question],
            'anns_field': 'sparse',
            'param': {
                'metric_type': 'BM25',
            },
            'limit': 3,
        }
        req1 = AnnSearchRequest(** search_params_1)

        # 稠密向量检索
        search_params_2 = {
                'data': dashscope_em.embed_documents(question),
                'anns_field': 'dense',
                'param': {
                    'metric_type': 'IP',
                    'params': {'nprobe': 10},
                },
                'limit': 3
            }
        req2 = AnnSearchRequest(** search_params_2)

        # 混合检索
        return self.client.hybrid_search(
            collection_name=collection_name,
            reqs=[req1, req2], # 搜索请求对象列表，每一个搜索请求对象都是 AnnSearchRequest
            ranker=RRFRanker(60), # 重排序，有2中 rerank: RRFRanker, WeightedRanker
            limit=3,
            output_fields=['id', 'text']
        )

    def drop_collection(self, collection_name: str):
        """删除 Collection """
        # 检查 Collection 是否存在
        if not self.client.has_collection(collection_name):
            print(f"Collection '{collection_name}' 已经不存在")
            return

        self.client.drop_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' 删除成功")

    def insert_documents(self, documents: list[Any]):
        try:
            # 插入数据
            res = self.client.insert(
                collection_name=self.collection_name,
                data=documents
            )
            print(f"成功插入 {res['insert_count']} 条文档")

            # 刷新数据（确保数据可搜索）
            self.client.flush(self.collection_name)
            print("数据已刷新")
        except Exception as e:
            print(f"插入文档失败: {e}")

    def delete_documents_by_ids(self, collection_name: str, ids: list[Union[str, int]]) -> dict[str, int]:
        return self.client.delete(collection_name=collection_name, ids=ids)

if __name__ == "__main__":
    ke_known_dao = KeKnownDao(db_name="ke_known_db")

    # ke_known_dao.create_collection("ke_known_collection")

    # collection_desc = ke_known_dao.get_collection_desc(collection_name='ke_known_collection')
    # print(f"表结构：{collection_desc}")

    # collection_indexes = ke_known_dao.get_collection_indexes(collection_name="ke_known_collection")
    # print(f"表索引：{collection_indexes}")

    # 模拟知识库文档
    # knowledge_docs = [
    #     {
    #         "text": "Python是一种高级编程语言，以其简洁易读的语法而闻名。它广泛应用于Web开发、数据分析、人工智能等领域。"
    #     },
    #     {
    #         "text": "JavaScript是Web开发的核心语言，用于创建交互式网页。它可以在浏览器和服务器端运行。"
    #     },
    # ]
    #
    # for doc in knowledge_docs:
    #     doc['dense'] = zhipuai_em.embed_documents(doc['text'])[0]
    #
    # ke_known_dao.insert_documents(knowledge_docs)

    # 过滤查询
    # response = ke_known_dao.client.query(
    #     collection_name="ke_known_collection",
    #     filter="id == 464160133250830688",
    #     output_fields=['id', 'text']
    # )
    # print(f"查询结果：{response}")

    # 匹配搜索（全文检索）
    # custom_search_params = {
    #     "params": {"drop_ratio_search": 0.2} # 搜索时忽略低重要度的比例
    #
    # }
    # result = ke_known_dao.search(
    #     collection_name="ke_known_collection",
    #     data=['高级编程语言'],
    #     search_params=custom_search_params
    # )


    # 混合检索
    result = ke_known_dao.hybrid(collection_name="ke_known_collection", question="浏览器")
    for hits in result:
        print(f"Top N 的结果")
        for item in hits:
            print(f"检索结果：{item}")

    # result = ke_known_dao.delete_documents_by_ids(collection_name="ke_known_collection",
    #                                      ids=[464160133250830526])
    # print(result)

    # ke_known_dao.drop_collection("ke_known_collection")
