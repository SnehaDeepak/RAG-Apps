import nest_asyncio
nest_asyncio.apply()

from llama_index.core.evaluation import generate_question_context_pairs, EmbeddingQAFinetuneDataset
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, load_index_from_storage, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core.evaluation import RetrieverEvaluator
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
import asyncio
import os
from datetime import datetime
import pandas as pd
import torch
import json
import mlflow
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.evaluation import RelevancyEvaluator
from llama_index.core.evaluation import BatchEvalRunner
from evaluation import eval_rag
from preprocessing import preprocess


system_prompt="""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""
## Default format supportable by LLama2
query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

## read config data
data_config_path = 'config/data.json'
with open(data_config_path, 'r') as f:
    data_config = json.load(f)

rag_config_path = 'config/rag.json'
with open(rag_config_path, 'r') as f:
    rag_config = json.load(f)

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y_%H:%M:%S")
run_name = "run_"+dt_string
mlflow.set_tag("mlflow.runName", run_name)

## log parameters
mlflow.log_param("granularity", data_config['type_granularity'])
if data_config['type_granularity'] == 'sentence':
    mlflow.log_param("sentence_window_size", data_config['sentence_window_size'])
else:
    mlflow.log_param("chunk_size", data_config['chunk_size'])
mlflow.log_param("Embeddings", data_config['Embeddings'])
mlflow.log_param("similarity_top_k", rag_config['similarity_top_k'])
mlflow.log_param("rerank_top_n", rag_config['rerank_top_n'])
mlflow.log_param("LLM", rag_config['LLM'])
mlflow.log_param("context_window", rag_config['context_window'])
mlflow.log_param("max_new_tokens", rag_config['max_new_tokens'])
mlflow.log_param("temperature", rag_config['temperature'])



if data_config['Embeddings']=='all-mpnet-base-v2':
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
elif data_config['Embeddings']=='bge-small-en-v1.5':
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

if rag_config['LLM']=='llama2':
    llm_rag = HuggingFaceLLM(
        context_window=rag_config['context_window'],
        max_new_tokens=rag_config['max_new_tokens'],
        generate_kwargs={"temperature": rag_config['temperature'], "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
        # uncomment this if using CUDA to reduce memory usage
        model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
    )
elif rag_config['LLM']=='mistral':
    llm_rag = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=None,
        temperature=rag_config['temperature'],
        max_new_tokens=rag_config['max_new_tokens'],
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=rag_config['context_window'],
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": -1},
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )


service_context_rag=ServiceContext.from_defaults(
    llm=llm_rag,
    embed_model=embeddings
)

# preprocess and load documents
path = '/workspace/RAGOps_artifacts/collections/'+data_config['collection']
documents = preprocess(path)

# create chunks for eval dataset 
# node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)
# eval_nodes = node_parser.get_nodes_from_documents(documents)

## create retriever and query engine
if data_config['type_granularity']=='chunk':
    print('###################### chunk data granularity #####################')
    node_parser = SimpleNodeParser.from_defaults(chunk_size=data_config['chunk_size'])
    nodes = node_parser.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex(nodes, service_context = service_context_rag)
    rerank = SentenceTransformerRerank(top_n=rag_config['rerank_top_n'], model="BAAI/bge-reranker-base")
    retriever = vector_index.as_retriever(similarity_top_k=rag_config['similarity_top_k'], node_postprocessors=[rerank])
    query_engine = vector_index.as_query_engine(similarity_top_k=rag_config['similarity_top_k'], node_postprocessors=[rerank])

elif data_config['type_granularity']=='sentence':
    print('###################### sentence data granularity ####################')
    node_parser = SentenceWindowNodeParser(
          window_size = data_config['sentence_window_size'],
          window_metadata_key = "window",
          original_text_metadata_key = "original_text")
    nodes = node_parser.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex(nodes, service_context = service_context_rag)

    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(top_n=rag_config['rerank_top_n'], model="BAAI/bge-reranker-base")
    retriever = vector_index.as_retriever(similarity_top_k=rag_config['similarity_top_k'], node_postprocessors=[postproc, rerank])
    query_engine = vector_index.as_query_engine(similarity_top_k=rag_config['similarity_top_k'], node_postprocessors=[postproc, rerank])

## save indexing and create vectordb
run_id = mlflow.active_run().info.run_id
save_dir = f'/workspace/RAGOps_artifacts/vectorstores/{run_id}/vector_store'
os.makedirs(save_dir)
vector_index.storage_context.persist(persist_dir=save_dir)

## evaluate rag pipeline
eval_rag(nodes, retriever, query_engine)

mlflow.set_tag("Eval-LLM", "Llama3")
mlflow.set_tag("Atrifact_URI", f"{mlflow.get_artifact_uri()}")
