from retrieval import KnowledgeRetriever
from openai import OpenAI
client = OpenAI(api_key="")


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
        print(related_info)
        prompt = f"""
        Your name is Ameca, you are answering questions from Macquarie University's students and staffs.
        Especially from the faculty of science and engineering.
        The user is asking about the query: ***{query}***. 
        The related information is: ***{related_info}***. 
        
        Please answer the user's question based on the related information. 
        If the given related information is not relevant to the user's question, please just ignore it.
        You should answer the questions with short sentences in a speaker tone, and do not ask questions. 
        Your response should no longer than 50 words. 
        You should not mention something like "according to the related information". 
        Sometimes the questions may include wrong speech recognized words, for example, "please give me an example of agility action in the factory" is actually asking "please give me an example of agility action in the falculty", then you should answer the question "falculty" instead of "factory", all questions are related to professional services at Universities. 
        If the user is asking about the agility examples or stories at Macquarie University, you can also provide the example with names.
        
        Now, please response to the query:
        """
        return prompt

    def get_response(self, query):
        prompt = self.get_prompt(query)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        )
        print(response)
        response_content = response.choices[0].message.content
        return response_content


if __name__ == "__main__":
    info = RAGInfo(use_public_embedding=True, top_k=5)
    print(info.get_response("Can you show us some examples for service charter in the faculty?  "))
