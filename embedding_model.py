from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings


class EmbeddingModel:
    public_model = "sentence-transformers/all-MiniLM-L6-v2"
    api_model_key = ""

    def __init__(self, use_public=True, **kwargs):
        self.use_public = use_public
        if use_public:
            self.embedding_model = self.public_embed(**kwargs)
        else:
            self.embedding_model = self.api_embed()

    def get_embeddings(self):
        return self.embedding_model

    def public_embed(self, **kwargs):
        model_kwargs = kwargs.get("model_kwargs", {'device': 'cuda'})
        encode_kwargs = kwargs.get("encode_kwargs", {'normalize_embeddings': False})
        embeddings_model = HuggingFaceEmbeddings(
            model_name=self.public_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return embeddings_model

    def api_embed(self):
        embeddings_model = OpenAIEmbeddings(api_key=self.api_model_key)
        return embeddings_model
