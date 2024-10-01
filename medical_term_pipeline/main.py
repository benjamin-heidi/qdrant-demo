from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os

class MedicalTermPipeline:
    def __init__(self):
        load_dotenv()
        self.llm = OpenAI(model="gpt-3.5-turbo-0613")
        self.client = QdrantClient(url=os.getenv("QDRANT_URL"))
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.vector_store = QdrantVectorStore(collection_name="my_collection", client=self.client)
        self.index = None

    def load_data(self, directory: str):
        documents = SimpleDirectoryReader(directory).load_data()
        self.index = VectorStoreIndex.from_documents(vector_store=self.vector_store, documents=documents, embedding=Settings.embed_model)

    def create_query_engine(self):
        return self.index.as_query_engine(streaming=True, similarity_top_k=1, llm=self.llm)

    async def stream_medical_terms(self, query: str):
        query_engine = self.create_query_engine()
        results = query_engine.query(query)
        for response in results:
            yield response

# Example usage
if __name__ == "__main__":
    pipeline = MedicalTermPipeline()
    pipeline.load_data("./data")
    for response in pipeline.stream_medical_terms("What are the symptoms of diabetes?"):
        print(response)
