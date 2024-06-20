from langchain_community.vectorstores import FAISS
from embedding_model import EmbeddingModel


class KnowledgeRetriever:
    knowledge_db_root_name = "knowledge_db_"

    def __init__(self,
                 use_public_embedding=True):
        self.embedding_model = EmbeddingModel(use_public=use_public_embedding).get_embeddings()
        if use_public_embedding:
            knowledge_db_pth = self.knowledge_db_root_name + "public_embedding"
        else:
            knowledge_db_pth = self.knowledge_db_root_name + "api_embedding"
        self.db = FAISS.load_local(knowledge_db_pth, self.embedding_model, allow_dangerous_deserialization=True)

    def retrieve(self, query, top_k=3):
        relevant_information = ""
        docs = self.db.similarity_search(query, k=top_k)
        for doc in docs:
            relevant_information += doc.page_content
        return relevant_information


if __name__ == "__main__":
    retriever = KnowledgeRetriever(use_public_embedding=True)
    results = retriever.retrieve("what's the meaning of service charter")
    print(results)
