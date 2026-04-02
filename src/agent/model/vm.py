from agent.config.config import DASHSCOPE_API_KEY

from langchain_community.embeddings import DashScopeEmbeddings

# 千问文本向量模型
dashscope_em = DashScopeEmbeddings(client='',
                                   dashscope_api_key=DASHSCOPE_API_KEY,
                                   model='text-embedding-v4')

if __name__ == "__main__":
    texts = [
        "人工智能正在改变世界。",
        "Python 是一种非常流行的编程语言。",
        "今天天气不错。"
    ]

    # 千问
    response = dashscope_em.embed_documents(texts)

    print(response)
    print(f"len: {len(response)}")
    print(f"len: {len(response[0])}")
