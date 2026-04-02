from fastapi import APIRouter, UploadFile
from agent.service.upload_service.upload_service import UploadService

upload_router = APIRouter()

@upload_router.post("/upload_single")
async def upload_single_file(file: UploadFile) -> dict:
    """
    上传单个文件
    :param file: 文件
    :return: 上传结果
    """
    # print(file)
    # 保存文档服务
    save_doc_service = UploadService()
    result = save_doc_service.save_to_local(file, file.filename)
    result["id"] = "1" # TODO
    return result

