from langchain_core.documents import Document


class DocSplitter(object):

    def split_doc(self, docs: list[Document]) -> list[Document]:
        pass
