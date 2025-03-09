# import metrics
from ragas.metrics import faithfulness, answer_relevancy, context_utilization
from ragas.metrics.critique import harmfulness, maliciousness, coherence, correctness, conciseness
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas.run_config import RunConfig
# wrappers
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
proxy_client = get_proxy_client('gen-ai-hub')
from langfuse import Langfuse
import asyncio
#langfuse_secret_key = 'sk-lf-990a8e06-e8f8-421f-b20d-74d36848cc12'
#langfuse_public_key = 'pk-lf-69d8b91f-6c12-41cd-af62-74e792b5540c'

'''# util function to init Ragas Metrics
def init_ragas_metrics(metrics, llm, embedding):
    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            metric.llm = llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = embedding
        run_config = RunConfig()
        metric.init(run_config)

async def score_with_ragas(query, chunks, answer):
    scores = {}
    for m in metrics:
        print(f"calculating {m.name}")
        scores[m.name] = await m.ascore(
            row={"question": query, "contexts": chunks, "answer": answer}
        )
        #await asyncio.sleep(1)
    print(scores)
    return scores

# metrics you chose
metrics = [faithfulness, answer_relevancy, context_precision, harmfulness]

llm = ChatOpenAI(temperature=0, proxy_model_name='gpt-4', max_tokens=1024)
emb = OpenAIEmbeddings(proxy_model_name='text-embedding-ada-002', proxy_client=proxy_client)

init_ragas_metrics(
    metrics,
    llm=LangchainLLMWrapper(llm),
    embedding=LangchainEmbeddingsWrapper(emb),
)
langfuse = Langfuse(
  secret_key="sk-lf-990a8e06-e8f8-421f-b20d-74d36848cc12",
  public_key="pk-lf-69d8b91f-6c12-41cd-af62-74e792b5540c",
  host="https://cloud.langfuse.com"
)
langfuse.auth_check()

question = "what is RAG"
context = "Retrieval-Augmented Generation (RAG) is a concept that enhances Language Models (LLMs) by integrating additional information from an external knowledge source. This approach allows LLMs to generate more accurate and contextual answers by bridging the gap between the model's general knowledge and specific details needed for certain queries. RAG functions similarly to an open-book exam, where external materials can be referenced to provide relevant information, thus improving the reasoning capabilities of the model without relying solely on its internal knowledge."
answer = "Retrieval-Augmented Generation (RAG) is a concept that enhances Language Models (LLMs) by integrating additional information from an external knowledge source."

asyncio.run(score_with_ragas(question, context, answer))
#print(scores)'''

# Import statements omitted for brevity

# Util function to init Ragas Metrics
def init_ragas_metrics(metrics, llm, embedding):
    print("Initializing metrics...")
    for metric in metrics:
        print("metrics", metric.name)
        if isinstance(metric, MetricWithLLM):
            metric.llm = llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = embedding
        run_config = RunConfig()
        metric.init(run_config)
    print("initialization done")

# Function to score with Ragas metrics (now synchronous)
async def score_with_ragas_async(query, chunks, answer, metrics):
    print("scoring started....")
    scores = {}
    for m in metrics:
        print(f"calculating {m.name}")
        scores[m.name] = await m.ascore(row={"question": query, "contexts": chunks, "answer": answer})
        # Simulating async behavior with sleep (remove in production)
        await asyncio.sleep(1)
    print("scoring completed")
    #print(scores)
    return scores

def ragas_evaluator(question, context, answer):
    # Metrics you chose
    metrics = [faithfulness, answer_relevancy, harmfulness, maliciousness, coherence, correctness, conciseness]

    llm = ChatOpenAI(temperature=0, proxy_model_name='gpt-4', max_tokens=1024)
    emb = OpenAIEmbeddings(proxy_model_name='text-embedding-ada-002', proxy_client=proxy_client)

    init_ragas_metrics(
        metrics,
        llm=LangchainLLMWrapper(llm),
        embedding=LangchainEmbeddingsWrapper(emb),
    )

    langfuse = Langfuse(
        secret_key="sk-lf-990a8e06-e8f8-421f-b20d-74d36848cc12",
        public_key="pk-lf-69d8b91f-6c12-41cd-af62-74e792b5540c",
        host="https://cloud.langfuse.com"
    )
    langfuse.auth_check()

    #question = "what is RAG"
    #context = "Retrieval-Augmented Generation (RAG) is a concept that enhances Language Models (LLMs) by integrating additional information from an external knowledge source. This approach allows LLMs to generate more accurate and contextual answers by bridging the gap between the model's general knowledge and specific details needed for certain queries. RAG functions similarly to an open-book exam, where external materials can be referenced to provide relevant information, thus improving the reasoning capabilities of the model without relying solely on its internal knowledge."
    #answer = "Retrieval-Augmented Generation (RAG) is a concept that enhances Language Models (LLMs) by integrating additional information from an external knowledge source."

    # Run the scoring function synchronously
    scores = asyncio.run(score_with_ragas_async(question, context, answer, metrics))
    #print("ragas", scores)
    return scores
    #print(scores)
