from pymilvus import MilvusClient

from agent.config.config import MILVUS_URI, MILVUS_USER, MILVUS_PASSWORD


class BaseMilvusDao:

    def __init__(self):
        self.client = MilvusClient(
            uri=MILVUS_URI,
            user=MILVUS_USER,
            password=MILVUS_PASSWORD,
        )

    def create_database(self, db_name: str = "default"):
        """创建数据库"""
        # 检查数据库是否存在
        try:
            if not self.is_exist_db(db_name):
                self.client.create_database(db_name)
                print(f"数据库 '{db_name}' 创建成功")
                self.client.using_database(db_name)
                print(f"切换数据库 '{db_name}' 成功")
            else:
                print(f"数据库 '{db_name}' 已存在，直接使用")
        except Exception as e:
            print(f"数据库操作失败: {e}")

    def is_exist_db(self, db_name: str) -> bool | None:
        """检查数据库是否存在"""
        try:
            # 获取所有数据库
            databases = self.client.list_databases()
            if db_name in databases:
                return True

            return False
        except Exception as e:
            print(f"数据库操作失败: {e}")

    def drop_database(self, db_name: str = "default"):
        try:
            if not self.is_exist_db(db_name):
                print(f"数据库 {db_name} 不存在")
                return

            self.client.using_database(db_name)

            collections = self.client.list_collections()
            for collection in collections:
                self.client.drop_collection(collection)
                print(f"删除数据库 {db_name} 下的集合 {collection}")

            self.client.drop_database(db_name)
            print(f"删除数据库 {db_name} 成功")
        except Exception as e:
            print(f"数据库操作失败: {e}")

    def create_collection(self, collection_name: str, **kwargs):
        """抽象方法：创建集合，由子类实现"""
        pass

    def drop_collection(self, collection_name: str):
        """抽象方法：删除集合，由子类实现"""
        pass

    def __del__(self):
        self.client.close()
        print("client 已关闭")

if __name__ == '__main__':
    base_milvus_dao = BaseMilvusDao()

    custom_db_name = "ke_known_db"
    base_milvus_dao.create_database(custom_db_name)

    # base_milvus_dao.drop_database(custom_db_name)
