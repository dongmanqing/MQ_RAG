from retrieval import KnowledgeRetriever


class RAGInfo:
    def __init__(self,
                 use_public_embedding=True,
                 top_k=3,
                 **kwargs):
        self.retriever = KnowledgeRetriever(use_public_embedding=use_public_embedding, **kwargs)
        self.top_k = top_k

    def get_info(self, query):
        results = self.retriever.retrieve(query, top_k=self.top_k)
        return results

    def get_prompt(self, query):
        related_info = self.get_info(query)
        prompt = f"""
        The user is asking about the query: ***{query}***. 
        The related information is: ***{related_info}***. 
        
        Please answer the user's question based on the related information. 
        If the given related information is not relevant to the user's question, please just ignore it.
        
        Now, please response to the query: ***{query}***. 
        """
        return prompt


if __name__ == "__main__":
    info = RAGInfo(use_public_embedding=True, top_k=3)
    print(info.get_prompt("what's the meaning of service charter"))
