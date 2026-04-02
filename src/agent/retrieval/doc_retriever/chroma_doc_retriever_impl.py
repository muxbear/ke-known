import os

from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory

from agent.model.llm import deepseek_llm
from agent.retrieval.doc_retriever.doc_retriever import DocRetriever


class ChromaDocRetriever(DocRetriever):
    persist_directory = "./chroma_langchain_db"

    collection_name = "ke_known_db"  # TODO

    RAG_PROMPT = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:
        """

    # 此提示告诉模型接收聊天历史记录和用户的最新问题，然后重新表述问题，以便可以独立于聊天历史记录来理解问题。
    CONTEXTUALIZE_Q_SYSTEM_PROMPT = """
        Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. 
        Do NOT answer the question just reformulate it if needed and otherwise return it as is.
        """

    CONTEXTUALIZE_Q_SYSTEM_PROMPT_CN = """
        给定一段聊天记录以及最新的用户问题，该问题可能与聊天记录中的某些内容有关。
        请重新构建一个独立的问题，使其无需参考聊天记录也能被理解。
        如果无需重新构建，则直接返回原问题。
        """

    QA_SYSTEM_PROMPT = """
        You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise. \
        {context} 
        """

    chroma: Chroma = None

    store = {}

    def __init__(self, embeddings):
        static_root = os.getenv("STATIC_ROOT")
        vdb_path = os.getenv("VDB_PATH")

        self.persist_directory = f'{static_root}{vdb_path}{self.collection_name}'
        print(f"Chroma 数据存储目录：{self.persist_directory}")

        self.embeddings = embeddings
        self.__get_vector_store()

    def __get_vector_store(self):
        self.chroma = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,  # Where to save data locally, remove if not necessary
            create_collection_if_not_exists=True
        )

    def retrieve_doc(self, question: str, session_id: str) -> str:
        # result = self.chroma.similarity_search(query=question, k=3)

        rag_prompt = PromptTemplate.from_template(self.RAG_PROMPT)
        retriever = self.chroma.as_retriever() # 获取 chroma 的检索器

        rag_chain = (
            {"context": retriever | self.__format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | deepseek_llm
            | StrOutputParser()
        )

        result = rag_chain.invoke(question)
        # print(f'检索结果：{result}')
        return result

    def retrieve_doc_with_history(self, question: str, session_id: str) -> str:
        # 该模板包括带说明的系统消息，聊天历史记录和占位符，以及 {input} 标记的最新的用户输入
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.CONTEXTUALIZE_Q_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        retriever = self.chroma.as_retriever()  # 获取 chroma 的检索器

        # 历史信息感知检索器
        history_aware_retriever = create_history_aware_retriever(deepseek_llm,
                                                                 retriever,
                                                                 contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.QA_SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # 构建问答链
        question_answer_chain = create_stuff_documents_chain(deepseek_llm, qa_prompt)

        # 组装 RAG 链：代表完整的工作流程。其中历史感知检索器首先处理查询以合并任何相关的历史上下文，
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        self.__invoke_and_save(session_id, question)

        response = conversational_rag_chain.invoke(
            {
                "input": question,
            },
            config={
                "configurable": {"session_id": session_id}
            }
        )

        # print(response['answer'])

        # 使用基本字典结构管理历史聊天记录
        store = {}


        return response['answer']

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()

        return self.store[session_id]

    def __format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def __invoke_and_save(self, session_id, input_text):
        self.__save_message(session_id, "human", input_text)
        pass

    def __save_message(self, session_id, role, input_text):
        pass