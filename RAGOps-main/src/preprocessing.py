from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, load_index_from_storage, StorageContext


def preprocess(path):
    documents = SimpleDirectoryReader(path).load_data()
    return documents
