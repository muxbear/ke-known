import multiprocessing
import os
from multiprocessing import Queue

from agent.component.doc_loader.unstructured_markdown_doc_impl import UnstructuredMarkdownDocImpl
from agent.component.doc_retriever.milvus_doc_retriever_impl import MilvusDocStorerImpl

def file_parser_handler(path: str, out_queue: Queue, batch_size: int = 20):
    """解析目录 path 下的所有 markdown 文件，并分批放入队列"""
    # 获取目录下的所有 markdown 文件
    markdown_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.md')]
    print(markdown_files)

    if not markdown_files:
        out_queue.put(None)

    # 将每一个 markdown 文件解析出来的 docs 保存在 doc_list 中
    docs_list = []
    for markdown_file in markdown_files:
        try:
            # 解析 markdown_file 输出 document 并添加到 doc_list
            unstructured_markdown_doc_impl = UnstructuredMarkdownDocImpl()
            docs = unstructured_markdown_doc_impl.load_doc(markdown_file)
            docs_list.extend(docs)

            # doc_list 的数量达到指定的大小，就将其发送到 queue
            if len(docs_list) >= batch_size:
                out_queue.put(docs_list.copy())
                docs_list.clear()  # 清空列表
        except Exception as e:
            print(f"解析文件{markdown_file}失败，原因：{e}")

    # 发送剩余的 docs
    if docs_list:
        out_queue.put(docs_list)

    # 所有 docs 发送完成之后，发送终止信号
    out_queue.put(None)

def lan_to_str (metadata):
    if 'languages' in metadata and isinstance(metadata['languages'], list):
        metadata['languages'] = ', '.join(metadata['languages'])
    return metadata

def ensure_metadata_fields(metadata):
    # 确保 category_depth 字段存在且为整数
    if 'category_depth' not in metadata:
        metadata['category_depth'] = 0  # 设置默认值
    elif not isinstance(metadata['category_depth'], int):
        try:
            metadata['category_depth'] = int(metadata['category_depth'])
        except:
            metadata['category_depth'] = 0
    return metadata

def milvus_write_handler(input_queue: Queue):
    """从队列  Queue 读取数据插入到 Milvus"""
    milvus_doc_storer = MilvusDocStorerImpl()
    milvus_doc_storer.connect_collection("ke_known_db", "ke_known_collection")

    total_count = 0
    while True:
        docs = input_queue.get() # 阻塞函数
        if docs is None: # 收到终止信号
            break
        if isinstance(docs, list):
            for doc in docs:
                if hasattr(doc, "metadata"):
                    doc.metadata = lan_to_str(doc.metadata)
                    # 确保 category_depth 字段存在且类型正确
                    doc.metadata = ensure_metadata_fields(doc.metadata)
            # 存储清理后的文档
            milvus_doc_storer.doc_store(docs)
            total_count += len(docs)

if __name__ == "__main__":
    file_path = r"../../../../static/resource/markdown"
    docs_queue = Queue()

    file_parser_process = multiprocessing.Process(target=file_parser_handler, args=(file_path, docs_queue, 40))
    milvus_write_process = multiprocessing.Process(target=milvus_write_handler, args=(docs_queue,))

    file_parser_process.start()
    milvus_write_process.start()

    file_parser_process.join()
    milvus_write_process.join()

    print("写入 doc 结束")