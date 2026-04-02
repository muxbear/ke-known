from agent.component.doc_retriever.milvus_doc_retriever_impl import MilvusDocStorerImpl

if __name__ == '__main__':
    milvus_doc_storer = MilvusDocStorerImpl()

    # 相似性检索
    result = milvus_doc_storer.milvus_storer.similarity_search(
        query="基础",
        k=2
    )

    for doc in result:
        print(doc)