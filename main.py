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
        The user is asking about the query: ***{query}***. 
        The related information is: ***{related_info}***. 
        
        Please answer the user's question based on the related information. 
        If the given related information is not relevant to the user's question, please just ignore it.
        You should answer the questions with short sentences in a speaker tone. No longer than 50 words. 
        If the user is asking about the agility examples, you can also provide the example with names.
        
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
    info = RAGInfo(use_public_embedding=True, top_k=3)
    print(info.get_response("Which member of the Faculty Executive Committee best exemplifies the Service Charter?  "))
