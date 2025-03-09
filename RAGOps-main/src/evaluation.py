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

system_prompt="""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""
## Default format supportable by LLama2
query_wrapper_prompt=SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")


def eval_rag(nodes, retriever, query_engine):
    
    tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct")
    
    stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),]
    
    llm_eval = HuggingFaceLLM(
    context_window=8192,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.1, "do_sample":
    False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="auto",
    stopping_ids=stopping_ids,
    tokenizer_kwargs={"max_length": 4096},
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16}
    )

    qa_dataset = generate_question_context_pairs(
        nodes,
        llm=llm_eval,
        num_questions_per_chunk=1
    )
    run_id = mlflow.active_run().info.run_id
    save_dir = '/workspace/RAGOps_artifacts/evaldata/'+run_id
    os.makedirs(save_dir)
    # [optional] save
    qa_dataset.save_json(save_dir+'/dataset.json')
    # [optional] load
    #qa_dataset = EmbeddingQAFinetuneDataset.from_json("eval_dataset.json")
    queries = list(qa_dataset.queries.values())

    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retriever
    )

    async def eval_context():
        eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
        return eval_results

    eval_results = asyncio.run(eval_context())

    def display_results(eval_results):
        """Display results from evaluate."""

        metric_dicts = []
        for eval_result in eval_results:
            metric_dict = eval_result.metric_vals_dict
            metric_dicts.append(metric_dict)

        full_df = pd.DataFrame(metric_dicts)

        hit_rate = full_df["hit_rate"].mean()
        mrr = full_df["mrr"].mean()

        metric_df = pd.DataFrame(
            {"Hit Rate": [hit_rate], "MRR": [mrr]}
        )
        metric_df.to_csv('metrics/context.csv', index=False)
        print('########################################################')
        print(metric_df)
        print('########################################################')

        return hit_rate, mrr

    hit_rate, mrr = display_results(eval_results)
    mlflow.log_metric('Hit Rate' , hit_rate)
    mlflow.log_metric('MRR', mrr)

    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    service_context_eval = ServiceContext.from_defaults(
      llm = llm_eval,
      embed_model = embed_model
    )

    faithfulness_llama3 = FaithfulnessEvaluator(service_context=service_context_eval)

    relevancy_llama3 = RelevancyEvaluator(service_context=service_context_eval)

    # Let's pick queries to do evaluation
    no_queries = len(queries)
    batch_eval_queries = queries[:no_queries]

    # Initiate BatchEvalRunner to compute FaithFulness and Relevancy Evaluation.
    runner = BatchEvalRunner(
        {"faithfulness": faithfulness_llama3, "relevancy": relevancy_llama3},
        workers=8,
    )

    # Compute evaluation
    async def eval_response():
        eval_results = await runner.aevaluate_queries(
            query_engine, queries=batch_eval_queries
        )
        return eval_results

    eval_results = asyncio.run(eval_response())

    # Let's get faithfulness score
    faithfulness_score = sum(result.passing for result in eval_results['faithfulness']) / len(eval_results['faithfulness'])

    # Let's get relevancy score
    relevancy_score = sum(result.passing for result in eval_results['relevancy']) / len(eval_results['relevancy'])

    metric_df = pd.DataFrame(
        {"faithfulness": [faithfulness_score], "relevancy": [relevancy_score]}
    )
    metric_df.to_csv('metrics/response.csv', index=False)
    print('########################################################')
    print(metric_df)
    print('########################################################')

    mlflow.log_metric('faithfulness' , faithfulness_score)
    mlflow.log_metric('relevancy', relevancy_score)
