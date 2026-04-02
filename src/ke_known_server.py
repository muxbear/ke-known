from fastapi import FastAPI

from api.rag_index_api.rag_index_api import rag_index_router
from api.upload_api.upload_api import upload_router

# 主应用
app = FastAPI(title="可知后台服务")
app.include_router(upload_router, prefix="/upload", tags=["文件上传服务"])
app.include_router(rag_index_router, prefix="/rag_index", tags=["RAG 文档索引服务"])

@app.get("/hello")
async def hello():
    return {"hello": "world"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
