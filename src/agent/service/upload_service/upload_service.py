import os
from fastapi import UploadFile
from pathlib import Path

class UploadService(object):
    static_root = os.getenv("STATIC_ROOT")
    static_url = os.getenv("STATIC_URL")
    doc_path = os.getenv("DOC_PATH")
    file_path = "txt/"

    def __init__(self):
        if not os.path.exists(self.static_root):
            print("请在 .env 中配置 static_root")
            return

        temp_doc_path = os.path.join(self.static_root, self.doc_path)
        if not os.path.exists(temp_doc_path):
            os.makedirs(temp_doc_path)
            print(f"自动创建了文档存储目录{temp_doc_path}")
            return

    def save_to_local(self, upload_file: UploadFile, file_name: str) -> dict[str, str]:
        """
        保存文件到本地磁盘。自动根据文件扩展名，将文件保存在对应目录
            txt: resource/doc/
            pdf: resource/pdf/
        :return:
        """

        # 获取文档扩展名
        ext_name = Path(file_name).suffix[1:] + "/"

        full_file_name = self.static_root + self.doc_path + ext_name + file_name
        with open(full_file_name, "wb") as file:
            while content := upload_file.file.read(2048):
                file.write(content)

        return {"file_path": file_name}




