from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from llama_index.core import StorageContext
import os

OPENAI_API_TOKEN = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN


import os

class DataIndexer:
    def __init__(self, output_folder="./mixed_data/"):
        self.output_folder = output_folder
        self.retriever_engine = None  # Initialize to None
        self.setup()
        self.index_flag_path = "./index_complete.flag"  # Path to index completion flag

    def setup(self):
        self.text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
        self.image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")
        self.storage_context = StorageContext.from_defaults(vector_store=self.text_store, image_store=self.image_store)

    def index_data(self):
        try:
            documents = SimpleDirectoryReader(self.output_folder).load_data()
            # Always initialize index, update your logic as needed to ensure it's correctly configured
            self.index = MultiModalVectorStoreIndex.from_documents(documents, storage_context=self.storage_context)
            #self.retriever_engine = self.index.as_retriever(similarity_top_k=5, image_similarity_top_k=5)
            print("Data indexing complete and retriever engine is ready.")
            with open(self.index_flag_path, 'w') as f:
                f.write("Indexing completed")
        except Exception as e:
            print(f"An error occurred during indexing: {e}")

    def get_retriever(self):
        self.retriever_engine= self.index.as_retriever(similarity_top_k=5, image_similarity_top_k=5)
        return self.retriever_engine

