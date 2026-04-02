from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

from agent.retrieval.doc_loader.doc_loader import DocLoader


class UnstructuredMarkdownDocImpl(DocLoader):

    def load_doc(self, path) -> list[Document]:
        loader = UnstructuredMarkdownLoader(
            file_path=path,
            mode="elements"
        )

        return loader.load()

if __name__ == '__main__':
    file = r'/resource/markdown/Arthas.md'
    markdown_loader = UnstructuredMarkdownDocImpl()
    docs = markdown_loader.load_doc(file)
    for doc in docs:
        print(doc)