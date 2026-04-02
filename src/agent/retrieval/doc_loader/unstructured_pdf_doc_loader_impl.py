import json
from typing import Iterator

from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader

from agent.retrieval.doc_loader.doc_loader import DocLoader


class UnstructuredDocLoaderImpl(DocLoader):
    loader = None

    def load_doc(self, path) -> list[Document]:
        self.loader = UnstructuredLoader(
            file_path=path,
            strategy="hi_res",  # fast, hi_res, auto
            coordinates=True,
            partition_via_api=True,
            api_key='IhWKAZRBmZ14c8tmCsOLabqwIKLJ2e'
        )

        return self.loader.load()

    def lazy_load_doc(self, path) -> Iterator[Document]:
        lazy_loader = UnstructuredLoader(
            file_path=path,
            strategy="hi_res",  # fast, hi_res, auto
            coordinates=True,
            partition_via_api=True,
            api_key='IhWKAZRBmZ14c8tmCsOLabqwIKLJ2e'
        )
        return lazy_loader.lazy_load()

    def writer_json(self, data, file_name):
        with open(file_name, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    unstructured_loader = UnstructuredDocLoaderImpl()
    # docs = unstructured_loader.load_doc(r"E:\document\ebook\alibaba\Java开发手册-嵩山版.pdf")

    counter = 0
    docs = []
    for doc in unstructured_loader.lazy_load_doc(r"E:\document\ebook\alibaba\Java开发手册-嵩山版.pdf"):
        docs.append(doc)
        json_file_name = str(doc.metadata.get('page_number')) + '_' + str(counter) + '.json'
        counter += 1
        file_name = f"D:/work/PyCharmProjects/ke-known/{json_file_name}"
        unstructured_loader.writer_json(doc.to_json(), file_name)

