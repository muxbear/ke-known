from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from agent.retrieval.doc_loader.doc_loader import DocLoader


class PyPdfDocLoaderImpl(DocLoader):
    """简单的 PDF 加载器，只能加载 PDF 中的文本"""

    def load_doc(self, path) -> list[Document]:
        loader = PyPDFLoader(path)
        return loader.load()

if __name__ == '__main__':
    pypdf_loader = PyPdfDocLoaderImpl()
    docs = pypdf_loader.load_doc(r"E:\document\ebook\alibaba\Java开发手册-嵩山版.pdf")
    print(f"docs 数量 {len(docs)}")

    print(f"docs[1].metadata = {docs[1].metadata}")
    print(f"docs[1].page_content = {docs[1].page_content}")