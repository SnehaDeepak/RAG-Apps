import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import torch
import os
import glob
import re
from langchain_community.document_loaders import TextLoader
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunctionRegistry
from PIL import Image
from pathlib import Path
import pandas as pd
import lancedb
from lancedb.embeddings import get_registry
from typing import ClassVar
from pydantic import Field
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
import nltk
import shutil
import random

VECTOR_SIZE = 512  
nltk.download('punkt')

GOOGLE_API_KEY = "AIzaSyD9FRF932rrA79k17EoTLf11LwOQqOjhBM"  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["COHERE_API_KEY"] = "MWG16sNqDiQzl0gSE904RUmUKbeuykqz0be2BK0O"

# Function to check the length of the file content
def is_valid_file(file_path, min_length=20):
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()
            return len(content) >= min_length
    except FileNotFoundError:
        return False
    

    
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}
        
class ParentDocumentRetrieverRAG:
    def __init__(self, db_directory="./JohnWick_db_parentsRD"):
        file1 = 'mixed_data/output_text.txt'
        file2 = 'captions2.txt'

        valid_file = file1 if is_valid_file(file1) else file2
        if not is_valid_file(valid_file):
            raise ValueError("Neither file contains valid content of the required length.")

        self.text_content = TextLoader(valid_file).load()
        self.parent_docs = [Document(page_content=doc.page_content) for doc in self.text_content]
        self.store = InMemoryStore()
        self.vectorstore = Chroma(embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), collection_name="fullDoc", persist_directory=db_directory)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter
        )
        self.retriever.add_documents(self.parent_docs, ids=None)
        print(f"Added {len(self.parent_docs)} documents to the retriever.")
        for doc in self.parent_docs:
            print(f"Document: {doc.page_content}")

    def get_retrieved_documents(self, query):
        parent_retrieved_docs = self.retriever.invoke(query)
        print(f"Retrieved {len(parent_retrieved_docs)} parent documents for query: {query}")
        for doc in parent_retrieved_docs:
            print(f"Parent Document: {doc.page_content}")
        child_retrieved_docs = self.vectorstore.similarity_search(query)
        for doc in child_retrieved_docs:
            print(f"Child Document: {doc.page_content}")
        return parent_retrieved_docs + child_retrieved_docs


class ContextualCompressionRetrieverRAG:
    def __init__(self, db_directory="./JohnWick_db_parentsRD"):
        file1 = 'mixed_data/output_text.txt'
        file2 = 'captions2.txt'

        # Determine which file to use based on content length
        valid_file = file1 if is_valid_file(file1) else file2
        # Check if the selected file is valid
        if not is_valid_file(valid_file):
            raise ValueError("Neither file contains valid content of the required length.")

        # Ensure writable directory
        os.makedirs(db_directory, exist_ok=True)

        # Load and split documents
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        documents = TextLoader(valid_file).load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        texts = text_splitter.split_documents(documents)

        # Initialize vectorstore and retriever
        self.vectorstore = Chroma(persist_directory=db_directory, embedding_function=embeddings, collection_name="fullDoc")
        self.vectorstore.add_documents(texts)
        naive_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        compressor = CohereRerank(top_n=3)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=naive_retriever
        )
        print(f"Added {len(texts)} documents to the retriever.")

    def get_retrieved_documents(self, query):
        retrieved_docs = self.compression_retriever.invoke(query)
        print(f"Retrieved {len(retrieved_docs)} documents for query: {query}")

        # Debugging: Print details of each retrieved document
        for doc in retrieved_docs:
            print(f"Document ID: {doc.metadata.get('doc_id', 'N/A')}")
            print(f"Document Content: {doc.page_content[:200]}...")  # Print first 200 characters
        
        return retrieved_docs




class ChromaDBStore:
    def __init__(self, path, collection_name, embedding_function, data_loader):
        self.client = chromadb.PersistentClient(path=path)
        if collection_name in self.client.list_collections():
            self.client.delete_collection(name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            data_loader=data_loader
        )
        self._initialize_data()

    def _initialize_data(self):
        self.add_image_data()
        self.add_text_data()

    def add_image_data(self):
        IMAGE_FOLDER = 'mixed_data'
        image_uris = sorted(glob.glob(os.path.join(IMAGE_FOLDER, 'frame_*.png')), key=lambda x: int(re.search(r'frame_(\d+).png', x).group(1)))
        self.image_ids = [str(i) for i in range(len(image_uris))]
        
        self.collection.add(ids=self.image_ids, uris=image_uris)

    def add_text_data(self):
        # Paths to the output files
        file1 = 'mixed_data/output_text.txt'
        file2 = 'captions2.txt'

        # Determine which file to use based on content length
        valid_file = file1 if is_valid_file(file1) else file2
        # Check if the selected file is valid
        if not is_valid_file(valid_file):
            raise ValueError("Neither file contains valid content of the required length.")

        raw_documents = TextLoader(valid_file).load()
        start_id = len(self.image_ids)  # Start from the next number after the last image ID
        ids = [str(start_id + 1)]
        
        raw_documents = [doc.page_content for doc in raw_documents]
        self.collection.add(ids=ids, documents=raw_documents)

    # Naive retriever
    def query(self, query_texts, n_results, include):
        return self.collection.query(
            query_texts=query_texts,
            n_results=n_results,
            include=include
        )

class Pets(LanceModel):
    registry: ClassVar = EmbeddingFunctionRegistry.get_instance()
    clip: ClassVar = registry.get("open-clip").create()
    vector: Vector(clip.ndims()) = clip.VectorField()
    image_uri: str = clip.SourceField()

    class Config:
        arbitrary_types_allowed = True

class TextModel(LanceModel):
    model: ClassVar = get_registry().get("gemini-text").create()
    text: str = model.SourceField()
    vector: Vector(model.ndims()) = model.VectorField()

    class Config:
        arbitrary_types_allowed = True

class LanceDBStore:
    def __init__(self):
        self.registry = EmbeddingFunctionRegistry.get_instance()
        self.clip = self.registry.get("open-clip").create()
        self.model = get_registry().get("gemini-text").create()
        self.db = lancedb.connect("lancedb")
        self._initialize_data()

    def _initialize_data(self):
        self.add_image_data()
        self.add_text_data()

    def add_image_data(self):
        print("------Performing Image embedding------")
        table = self.db.create_table("images", schema=Pets, mode="overwrite")
        p = Path("mixed_data").expanduser()
        uris = [str(f) for f in p.glob("*.png")]
        table.add(pd.DataFrame({"image_uri": uris}))

    def add_text_data(self):
        # Paths to the output files
        file1 = 'mixed_data/output_text.txt'
        file2 = 'captions2.txt'

        # Determine which file to use based on content length
        valid_file = file1 if is_valid_file(file1) else file2
        # Check if the selected file is valid
        if not is_valid_file(valid_file):
            raise ValueError("Neither file contains valid content of the required length.")

        p = Path(valid_file).expanduser()
        if p.exists():
            with p.open('r') as file:
                content = file.read()
                sentences = nltk.sent_tokenize(content)
        
        print("------Performing Text embedding------")
        table = self.db.create_table("text", schema=TextModel, mode="overwrite")
        table.add(pd.DataFrame({"text": sentences}))

    def query(self, query_texts, n_results, include):
        self.db = lancedb.connect("lancedb")
        query_text = query_texts[0] if query_texts else ""
        combined_text = ""
        images = []

        try:
            table_text = self.db["text"]
            text_results = table_text.search(query_text).to_pydantic(TextModel)
            combined_text = " ".join([word.text for word in text_results])
        except Exception as e:
            print(f"Text query error: {e}")

        try:
            table_image = self.db["images"]
            image_results = table_image.search(query_text).limit(n_results).to_pydantic(Pets)
            images = [pet.image_uri for pet in image_results]
        except Exception as e:
            print(f"Image query error: {e}")

        return {
            "combined_text": combined_text,
            "images": images
        }


class VectorStoreFactory:
    @staticmethod
    def create_vector_store(vector_store_type, retriever_type):
        random_number = random.randint(1000, 9999)
        path = f"test{random_number}"
        
        if vector_store_type == "chroma":
            if retriever_type == "naive":
                return ChromaDBStore(
                    path=path,
                    collection_name='multimodal_collection67',
                    embedding_function=OpenCLIPEmbeddingFunction(),
                    data_loader=ImageLoader()
                )
            elif retriever_type == "parent_document":
                return ParentDocumentRetrieverRAG()
            elif retriever_type == "contextual_compression":
                return ContextualCompressionRetrieverRAG()
            else:
                raise ValueError("Unsupported retriever type for Chroma")
        elif vector_store_type == "lance":
            return LanceDBStore()
        else:
            raise ValueError("Unsupported vector store type")
            



def initialize_vector_store(vector_store_type, retriever_type=None):
    global vector_store, current_vector_store_type, current_retriever_type
    current_vector_store_type = vector_store_type
    current_retriever_type = retriever_type
    if vector_store_type == "chroma":
        random_number = random.randint(1000, 9999)
        path = f"test{random_number}"
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            print(f"Deleted the database directory at {db_path}.")
        else:
            print(f"No database directory found at {db_path} to delete.")
        vector_store = VectorStoreFactory.create_vector_store(vector_store_type, retriever_type)
    elif vector_store_type == "lance":
        vector_store = VectorStoreFactory.create_vector_store(vector_store_type, retriever_type="naive")
    return f"Initialized {vector_store_type} with {retriever_type if retriever_type else 'default settings'}"



# Example usage
#vector_store_type = "chroma"
#retriever_type = "naive"  # Options: "naive", "parent_document", "contextual_compression"

#vector_store = VectorStoreFactory.create_vector_store(vector_store_type, retriever_type)

"""

# Example query
query = "for iit and aiims what are opportunities"

# Perform query based on retriever type
if retriever_type == "naive":
    result = vector_store.query(query_texts=[query], n_results=5, include=["documents", "uris"])
    print(result)
elif retriever_type == "parent_document" or retriever_type == "contextual_compression":
    result = vector_store.get_retrieved_documents(query)
    for doc in result:
        print(doc.page_content)
"""
