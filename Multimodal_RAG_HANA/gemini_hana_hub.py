from langchain_community.document_loaders import PyPDFLoader
import time
import shutil
import json
import os
import re
#from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
#from gen_ai_hub.proxy.langchain.openai import OpenAI
#from langchain.memory import ConversationBufferMemory
#from langchain.chains import ConversationalRetrievalChain
#from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.langchain.google_gemini import ChatGoogleGenerativeAI
from hdbcli import dbapi
from langchain_community.vectorstores.hanavector import HanaDB
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
proxy_client = get_proxy_client('gen-ai-hub')

def load_pdf_hana(wiki_space):
    docs =[]
    print("^^^^^^^^^^^^^^^^^^^^^^", wiki_space)
    if wiki_space != None:
        for file in os.listdir(wiki_space):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(wiki_space,file)
                loader = PyPDFLoader(pdf_path)
                docs.extend(loader.load())
            print("file name ------>",pdf_path)
    elif wiki_space == None:
        for file in os.listdir("data"):
            if file.endswith(".pdf"):
                pdf_path = "./data/" +file
                loader = PyPDFLoader(pdf_path)
                docs.extend(loader.load())
            print("file name ------>",pdf_path)
    return docs

def hana_initialize():
    with open("cds-hana-vectordb-instance-key.json") as json_file:
        hana_credential = json.load(json_file)

    [hana_db_address, hana_db_port, hana_db_user, hana_db_password] = \
        map(lambda var: hana_credential[var], ["host", "port", "user", "password"])

    # Use connection settings from the environment
    connection = dbapi.connect(
        address=hana_db_address,
        port=hana_db_port,
        user=hana_db_user,
        password=hana_db_password,
        autocommit=True,
        sslValidateCertificate=False,
    )
    print("Connected to HANA DB!!")
    return connection

def create_embd(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=5)
    texts = text_splitter.split_documents(docs)
    #print("^^^^^^^^ splitted texts ^^^^^^^^^^")
    #print(texts)

    embedding_model = OpenAIEmbeddings(proxy_model_name='text-embedding-ada-002', proxy_client=proxy_client)
    #response = embedding_model.embed_documents(doc)
    '''shutil.rmtree('gpt_vec_db', ignore_errors=True)
    os.makedirs('gpt_vec_db', exist_ok=True)

    persist_directory = "gpt_vec_db"
    #vectordb = FAISS.from_documents(documents=texts, embedding=embeddings)
    #vectordb.save_local('gemini_vec_db') #save document locally
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding_model, persist_directory=persist_directory)
    vectordb.persist() #save document locally'''
    connection = hana_initialize()
    vector = HanaDB(embedding=embedding_model, connection=connection, table_name="RAG_POC_CDS")
    vector.delete(filter={})
    vector.add_documents(texts)
    print("********---------> vector db created")
    return vector

'''def infer(query):
    #persist_directory = 'gpt_vec_db'
    model = ChatGoogleGenerativeAI(proxy_model_name='gemini-1.0-pro')
    embedding_model = OpenAIEmbeddings(proxy_model_name='text-embedding-ada-002', proxy_client=proxy_client)
    docs = load_pdf()
    vectordb = create_embd(docs)
    #vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

    qa_template = """ Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Only return the helpful answer below and nothing else.
    Helpful answer: 
    """

    QA_CHAIN_PROMPT = PromptTemplate(template=qa_template,
                                        input_variables=['context','question'])
    qa_chain = RetrievalQA.from_chain_type(model,
                                            retriever = vectordb.as_retriever(search_kwargs={'k':5}),
                                            return_source_documents=True,
                                            chain_type_kwargs={'prompt':QA_CHAIN_PROMPT}
                                            )
    try:
        response2 = qa_chain({'query': query})
    except Exception as e:
        time.sleep(120)
        response2 = qa_chain({'query': query})

    print(f"Answer from LLM : {response2['result']}")

infer("what is rag")'''
