from transformers import AutoTokenizer, AutoModel
from openai import OpenAI


class EmbeddingModel:
    public_model = "sentence-transformers/all-MiniLM-L6-v2"
    api_model = OpenAI(api_key="")

    def __init__(self, use_public=True):
        self.use_public = use_public
        if use_public:
            self.tokenizer = AutoTokenizer.from_pretrained(self.public_model)
            self.model = AutoModel.from_pretrained(self.public_model)
        else:
            self.model = self.api_model

    def embed(self, doc):
        return self.public_embed(doc) if self.use_public else self.api_embed(doc)

    def public_embed(self, doc):
        inputs = self.tokenizer(doc, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def api_embed(self, doc):
        response = self.api_model.embeddings.create(
            model="text-embedding-ada-002",
            input=[doc]
        )
        return response.data[0].embedding
