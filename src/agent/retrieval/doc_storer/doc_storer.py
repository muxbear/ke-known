from langchain_core.documents import Document


class DocStorer(object):

    def doc_store(self, docs: list[Document]) -> list[str]:
        """保存向量"""
        pass

    def delete_store(self):
        """删除向量库"""
        pass