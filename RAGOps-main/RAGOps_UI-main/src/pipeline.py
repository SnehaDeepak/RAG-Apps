from llama_index.core import ServiceContext, load_index_from_storage, StorageContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
import torch
import json

system_prompt="""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""
## Default format supportable by LLama2
query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")



global INDEX
global GRANULARITY

def deploy_rag(deploydata):
    
    global INDEX
    global GRANULARITY
    GRANULARITY = deploydata['granularity']
    
    # load llm
    if deploydata['LLM']=='llama2':
        llm = HuggingFaceLLM(
            context_window=int(deploydata['context_window']),
            max_new_tokens=int(deploydata['max_new_tokens']),
            generate_kwargs={"temperature": float(deploydata['temperature']), "do_sample": False},
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
            model_name="meta-llama/Llama-2-7b-chat-hf",
            device_map="auto",
            # uncomment this if using CUDA to reduce memory usage
            model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
        )

    elif deploydata['LLM']=='mistral':
        llm = LlamaCPP(
            # You can pass in the URL to a GGML model to download it automatically
            model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf',
            # optionally, you can set the path to a pre-downloaded model instead of model_url
            model_path=None,
            temperature=float(deploydata['temperature']),
            max_new_tokens=int(deploydata['max_new_tokens']),
            # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
            context_window=int(deploydata['context_window']),
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

    # load embeddings
    if deploydata['Embeddings']=='all-mpnet-base-v2':
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    elif deploydata['Embeddings']=='bge-small-en-v1.5':
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
    
    Service_Context=ServiceContext.from_defaults(
        llm=llm,
        embed_model=embeddings
    )
        
    save_dir = '/workspace/RAGOps_artifacts/vectorstores/'+deploydata['Run_id']+'/vector_store'
    INDEX = load_index_from_storage(
    StorageContext.from_defaults(persist_dir=save_dir),
    service_context=Service_Context)


def run_query(userdata):
    
    global INDEX
    global GRANULARITY
    
    query = userdata['query']
    
    if GRANULARITY == 'chunk':
        rerank = SentenceTransformerRerank(top_n=int(userdata['rerank_top_n']), model="BAAI/bge-reranker-base")
        engine = INDEX.as_query_engine(similarity_top_k=int(userdata['similarity_top_k']), node_postprocessors=[rerank])
        response = engine.query(query)
        
    elif GRANULARITY == 'sentence':
        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        rerank = SentenceTransformerRerank(top_n=int(userdata['rerank_top_n']), model="BAAI/bge-reranker-base")
        engine = INDEX.as_query_engine(similarity_top_k=int(userdata['similarity_top_k']), node_postprocessors=[postproc, rerank])
        response = engine.query(query)
    print(response.response)
    return response.response
