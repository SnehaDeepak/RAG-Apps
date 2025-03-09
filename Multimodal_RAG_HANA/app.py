import gradio as gr
from gradio.themes import *
import pandas as pd
import os
import shutil
import time
from gemini_hub import *
from gemini_hana_hub import *
from gpt_hub import *
from get_wiki_pgs import *
from pdf_extractor import *
from multi_vector_chroma import *
from multi_vector_hana import *
from evaluator import RAGEvaluator
from ragas_test import *
global MODEL
global vectordb
global multi_vectordb_chroma, multi_vectorstore_hana
global text_summaries, table_summaries
global img_base64_list, image_summaries
global tables, texts
global metrics
global all_score
MODEL ='none'
vectordb = 'none'
multi_vectordb_chroma, multi_vectorstore_hana = 'none', 'none'
text_summaries, table_summaries = 'none', 'none'
img_base64_list, image_summaries = 'none', 'none'
tables, texts = 'none', 'none'
all_score =pd.DataFrame()
###### css style #######
css = """
h1 {
text-align:center;
}
"""
# upload files
def upload_file(files):
    file_paths = [file.name for file in files]
    print("********",file_paths)
    shutil.rmtree('data')
    os.mkdir('data')

    for file_path in file_paths:
        cmd = 'cp ' +file_path+ ' data/'
        print("&&&&&&&&",cmd)
        os.system(cmd)
    return file_paths

#get data from wiki spaces
def get_wiki_data(space_key):
    main(space_key)
    return "Data downloaded from selected wiki space"

#create llm agent & vector db
def createagent(modelname, db_type, wiki_space=None):
    global MODEL
    MODEL = modelname
    global vectordb

    print("creating LLM agent & vector db for-------->", MODEL)
    if modelname == 'Gemini-Pro':
        if db_type == 'HANA':
            docs = load_pdf_hana(wiki_space)
            vectordb = create_embd(docs)
        elif db_type == 'Chroma':
            docs = load_pdf_chroma(wiki_space)
            create_chroma(docs)

    elif modelname == 'GPT-4':
        docs = load_pdf(wiki_space)
        if db_type == 'HANA':
            vectordb = create_embd_hana(docs)
        elif db_type == 'Chroma':
            create_embd_chroma(docs)

    return "VectorDB & LLM Agent is ready!!!"

#for multimodal
def createagent_multi(modelname, db_type):
    global MODEL
    MODEL = modelname
    global multi_vectordb_chroma, multi_vectorstore_hana
    global text_summaries, table_summaries
    global img_base64_list, image_summaries
    global tables, texts

    print("creating LLM agent & vector db for-------->", MODEL)
    if modelname == 'Gemini-Pro':
        pass

    elif modelname == 'GPT-4':
        if db_type == 'HANA':
            connection = hana_connect()
            all_texts, tables, texts = partition_pdf()
            # Get text, table summaries
            text_summaries, table_summaries = generate_text_summaries(
                all_texts, tables, summarize_texts=True
            )
            # Image summaries
            img_base64_list = generate_img_summaries('mixed_data')
            captioner = ImageCaptioner()
            image_summaries = list(captioner.run("mixed_data"))
            # The vectorstore to use to index the summaries
            embedding_model = OpenAIEmbeddings(proxy_model_name='text-embedding-ada-002', proxy_client=proxy_client)
            multi_vectorstore_hana = HanaDB(embedding=embedding_model, connection=connection, table_name="RAG_POC_CDS")
            multi_vectorstore_hana.delete(filter={})

        elif db_type == 'Chroma':
            shutil.rmtree('multi_vector', ignore_errors=True)
            os.makedirs('multi_vector', exist_ok=True)
            path = 'multi_vector'
            multi_vectordb_chroma = create_multi_store(path)

    return "VectorDB & LLM Agent is ready!!!"

#model inference or LLM response generation
def model_infer(input_text,db_type):
    global MODEL
    global vectordb
  
    print("Inference of LLM Model ---->", MODEL)
    print("Input text ------>", input_text)

    if MODEL == 'none':
        return "LLM model not selected. Select model first"
    
    elif MODEL == 'GPT-4':
        if db_type == 'Chroma':
            persist_directory = 'gpt_vec_db'
            embedding_model = OpenAIEmbeddings(proxy_model_name='text-embedding-ada-002', proxy_client=proxy_client)
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
        elif db_type == 'HANA':
            vectordb = vectordb

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0, proxy_model_name='gpt-4'),
                                           vectordb.as_retriever(), 
                                           memory=memory)

        response = qa.run(input_text)
        #print(answer)
        return response
    
    elif MODEL == 'Gemini-Pro':
        if db_type == 'Chroma':
            db = load_chroma_collection()
            print("vec db loaded")
            relevant_text = get_relevant_passage(input_text,db)
            print("retrievr")
            prompt = make_rag_prompt(input_text, 
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
            return response.content
        
        elif db_type == 'HANA':
            model=ChatGoogleGenerativeAI(proxy_model_name='gemini-1.0-pro')
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
                response2 = qa_chain({'query': input_text})
            except Exception as e:
                time.sleep(120)
                response2 = qa_chain({'query': input_text})
            
            return response2['result']

def infer_multi_modal(query, db_type):
    global multi_vectordb_chroma, multi_vectorstore_hana
    global text_summaries, table_summaries
    global img_base64_list, image_summaries
    global tables, texts
    global metrics
    global all_score
    if db_type == 'Chroma':
        response_multi, context_str = perform_query(multi_vectordb_chroma, query)
        # Initialize evaluator
        evaluator = RAGEvaluator()
        metrics = evaluator.evaluate_all(str(response_multi.content),str(context_str))
        #print("normal", metrics)
        ragas_score = ragas_evaluator(query, str(context_str), str(response_multi.content))
        #all_score = {**metrics, **ragas_score}
        descriptions=['Measures the similarity between machine-generated text and reference text based on n-gram overla',
                                 'Evaluates the overlap of unigram units (words) between machine-generated and reference texts to assess summarization quality.',
                                 'Measures the proportion of relevant items among the retrieved items by a BERT-based model.',
                                 'Measures the proportion of relevant items that have been retrieved by a BERT-based model.',
                                 'Harmonic mean of precision and recall scores for assessing the overall performance of a BERT-based model.',
                                 'Quantifies how well a language model predicts a sample of text by measuring the uncertainty of predicting the next word.',
                                 'Measures the variety and distinctiveness of generated text to ensure it covers a wide range of topics or perspectives.',
                                 'Evaluates the presence and impact of biases related to race or ethnicity in generated or processed text.',
                                 'faithfulness', 'answer_relevancy', 'harmfulness', 'maliciousness', 'coherence', 'correctness', 'conciseness']
        data = [{'metric': k, 'value': v, 'description': descriptions[i]} for i, (k, v) in enumerate({**metrics, **ragas_score}.items())]
        all_score = pd.DataFrame(data)
        print(all_score)
        return response_multi.content
    elif db_type == 'HANA':
        response_multi = perform_query_multi(query,multi_vectorstore_hana,
                                             text_summaries, texts,
                                            table_summaries, tables,
                                        image_summaries, img_base64_list)
        all_context = text_summaries + table_summaries + image_summaries
        context_str = '\n'.join(all_context)
        # Initialize evaluator
        evaluator = RAGEvaluator()
        metrics = evaluator.evaluate_all(str(response_multi),str(context_str))
        ragas_score = ragas_evaluator(query, str(context_str), str(response_multi))
        descriptions=['Measures the similarity between machine-generated text and reference text based on n-gram overla',
                                 'Evaluates the overlap of unigram units (words) between machine-generated and reference texts to assess summarization quality.',
                                 'Measures the proportion of relevant items among the retrieved items by a BERT-based model.',
                                 'Measures the proportion of relevant items that have been retrieved by a BERT-based model.',
                                 'Harmonic mean of precision and recall scores for assessing the overall performance of a BERT-based model.',
                                 'Quantifies how well a language model predicts a sample of text by measuring the uncertainty of predicting the next word.',
                                 'Measures the variety and distinctiveness of generated text to ensure it covers a wide range of topics or perspectives.',
                                 'Evaluates the presence and impact of biases related to race or ethnicity in generated or processed text.',
                                 'faithfulness', 'answer_relevancy', 'harmfulness', 'maliciousness', 'coherence', 'correctness', 'conciseness']
        data = [{'metric': k, 'value': v, 'description': descriptions[i]} for i, (k, v) in enumerate({**metrics, **ragas_score}.items())]
        all_score = pd.DataFrame(data)
        return response_multi

def display_images(directory_path='fetched_images'):
    image_paths = glob.glob(os.path.join(directory_path, '*.png')) + glob.glob(os.path.join(directory_path, '*.jpg'))
    if not image_paths:
        return "Images related to query could not be found!! Try with different query", False
    images = [os.path.join(directory_path, os.path.basename(img_path)) for img_path in image_paths]
    return images, True


def show_img(directory_path):
    result, success = display_images(directory_path)
    if not success:
        return [], result  # Return empty list for images and error message
    else:
        return result, "Here are your related images from relevant documents"  # Return image paths and empty message

def display_eval_score():
    df = all_score
    return df

######### gradio UI design ##############

with gr.Blocks(gr.themes.Monochrome()) as demo:
    #gr.Markdown("""#  Document analysis using RAG """)
    with gr.Row():
        with gr.Column(scale=2, variant="compact"):
            gr.Image("sap_logo.jpeg", show_download_button=False, show_label=False,container=False, width=1)
        with gr.Column(scale=8, variant="compact"):
            gr.HTML("<h2 style=\"color=#0e15cc;\">RAG Application to Analyze Wiki data</h2>")

        # file upload
    with gr.Tab("Simple RAG"):
        file_output = gr.File()
        upload_button = gr.UploadButton("Click to upload data file", file_types=["text", ".json", ".pdf", ".csv"], file_count="multiple")
        upload_button.upload(upload_file, upload_button, file_output)

        #wiki data
        '''space_list = get_all_space()
        wiki_space = gr.Dropdown(space_list, label="Available Unrestricted Spaces", info="Please select space")
        out3 = gr.Textbox(label="Status", value="No space is selected")
        btn3 = gr.Button("Get data")
        btn3.click(fn=get_wiki_data, inputs=wiki_space, outputs=out3)'''
        # create llm agent & vector db
        with gr.Row():
            inp1 = gr.Dropdown(["Gemini-Pro", "GPT-4"], label="Models", info="Please select Model")
            inp2 = gr.Dropdown(['Chroma', 'HANA'], label="Vector Database", info="Please select vector database to be used")
            out1 = gr.Textbox(label="Status", value="No agent is created")
        btn1 = gr.Button("Create LLM agent & Vector DB")
        btn1.click(fn=createagent, inputs=[inp1,inp2], outputs=out1)
        # run LLM model
        with gr.Row():
            inp3 = gr.Textbox(label='Write a query')
            out2 = gr.Textbox(label='Response')
        btn2 = gr.Button("Run LLM")
        btn2.click(fn=model_infer, inputs=[inp3,inp2], outputs=out2)
    
    with gr.Tab("Multimodal RAG"):
        file_output = gr.File()
        upload_button = gr.UploadButton("Click to upload data file", file_types=["text", ".json", ".pdf", ".csv"], file_count="multiple")
        upload_button.upload(upload_file, upload_button, file_output)

        #wiki data
        space_list = get_all_space()
        wiki_space = gr.Dropdown(space_list, label="Available Unrestricted Spaces", info="Please select space")
        out_1 = gr.Textbox(label="Status", value="No space is selected")
        out_2 = gr.Textbox(label="Status", value="Data not processed")
        btn_1 = gr.Button("Get data")
        btn_1.click(fn=get_wiki_data, inputs=wiki_space, outputs=out_1)
        btn_2 = gr.Button("Process data")
        btn_2.click(fn=process_multi_pdf, inputs=wiki_space, outputs=out_2)

        # create llm agent & vector db
        with gr.Row():
            inp_1 = gr.Dropdown(["GPT-4"], label="Models", info="Please select Model")
            inp_2 = gr.Dropdown(['Chroma', 'HANA'], label="Vector Database", info="Please select vector database to be used")
            out_3 = gr.Textbox(label="Status", value="No agent is created")
        btn_3 = gr.Button("Create LLM agent & Vector DB")
        btn_3.click(fn=createagent_multi, inputs=[inp_1,inp_2], outputs=out_3)
        # run LLM model
        with gr.Row():
            inp_3 = gr.Textbox(label='Write a query')
            out_4 = gr.Textbox(label='Response')
        btn_4 = gr.Button("Run LLM")
        btn_4.click(fn=infer_multi_modal, inputs=[inp_3,inp_2], outputs=out_4)
        
        #display evaluation metrics
        btn_6 = gr.Button("Evaluate")
        out_6 = gr.DataFrame()
        btn_6.click(fn=display_eval_score,outputs=out_6)

        #display retrived images
        img_dir_path = gr.Textbox(value='fetched_images', label='Directory Path')
        btn_5 = gr.Button("Display related images")
        gallery = gr.Gallery()
        error_label = gr.Label()
        btn_5.click(fn=show_img, inputs=img_dir_path, outputs=[gallery,error_label])

demo.queue().launch(share=True)

