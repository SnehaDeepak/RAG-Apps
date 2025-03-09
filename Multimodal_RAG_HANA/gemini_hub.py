from pypdf import PdfReader
import time
import shutil
import os
import re
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.google_gemini import ChatGoogleGenerativeAI
proxy_client = get_proxy_client('gen-ai-hub')

def load_pdf_chroma(wiki_space):
    docs =[]
    print("^^^^^^^^^^^^^^^^^^^^^^", wiki_space)
    if wiki_space != None:
        for file in os.listdir(wiki_space):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(wiki_space,file)
                # Logic to read pdf
                reader = PdfReader(pdf_path)
                # Loop over each page and store it in a variable
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                split_text = re.split('\n \n', text)
                docs.extend([i for i in split_text if i != ""])
            print("file name ------>",pdf_path)
    elif wiki_space == None:
        for file in os.listdir("data"):
            if file.endswith(".pdf"):
                pdf_path = "./data/" +file
                # Logic to read pdf
                reader = PdfReader(pdf_path)
                # Loop over each page and store it in a variable
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                split_text = re.split('\n \n', text)
                docs.extend([i for i in split_text if i != ""])
            print("file name ------>",pdf_path)
    
    return docs

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        proxy_client = get_proxy_client('gen-ai-hub')

        embedding_model = OpenAIEmbeddings(proxy_model_name='text-embedding-ada-002', proxy_client=proxy_client)
        embd = embedding_model.embed_documents(input)
        return embd

def create_chroma(docs):
    shutil.rmtree('gemini_vec_db1', ignore_errors=True)
    os.makedirs('gemini_vec_db1', exist_ok=True)

    path = "gemini_vec_db1"
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(name='rag_gemini', embedding_function=GeminiEmbeddingFunction())

    for i, d in enumerate(docs):
        db.add(documents=d, ids=str(i))

    return 'Vector DB created!!!'

def load_chroma_collection():
    persist_directory = "gemini_vec_db1"
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    db = chroma_client.get_collection(name='rag_gemini', embedding_function=GeminiEmbeddingFunction())

    return db

def get_relevant_passage(query, db):
  passage = db.query(query_texts=[query], n_results=3)['documents'][0]
  return passage

def make_rag_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and converstional tone. \
  If the passage is irrelevant to the answer, you may ignore it.
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

  ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt

'''def infer(query):
    docs = load_pdf()
    create_chroma(docs)
    print("vector db created")
    db = load_chroma_collection()
    print("vec db loaded")
    relevant_text = get_relevant_passage(query,db)
    print("retrievr")
    prompt = make_rag_prompt(query, 
                             relevant_passage="".join(relevant_text)) # joining the relevant chunks to create a single passage
    model = ChatGoogleGenerativeAI(proxy_model_name='gemini-1.0-pro')
    print('model loaded')
    #answer = model.generate_content(prompt)
    try:
        response = model.invoke(prompt)
        #response = model.generate_content(prompt)
    except Exception as e:
        time.sleep(120)
        response = model.invoke(prompt)
        #response = model.generate_content(prompt)
    print("********",response.content)
    #resp = response.text
    #print("answer from LLM", resp)

infer("what is llm-commons")'''
