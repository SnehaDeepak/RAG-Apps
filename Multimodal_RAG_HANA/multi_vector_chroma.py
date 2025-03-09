import os
import glob
import random
import json
import time
import base64
import re
from PIL import Image
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.hanavector import HanaDB
from hdbcli import dbapi
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
#from gen_ai_hub.proxy.native.google.clients import GenerativeModel
from gen_ai_hub.proxy.langchain.google_gemini import ChatGoogleGenerativeAI
proxy_client = get_proxy_client('gen-ai-hub')

#from langchain_community.vectorstores.hanavector import OpenCLIPEmbeddingFunction
#from langchain_community.vectorstores.hanavector import ImageLoader, TextLoader
#from langchain_community.vectorstores.hanavector.utils.embedding_functions import OpenCLIPEmbeddingFunction
#from langchain_community.vectorstores.hanavector.utils.data_loaders import ImageLoader, TextLoader


# Function to check the length of the file content
def is_valid_file(file_path, min_length=20):
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()
            return len(content) >= min_length
    except FileNotFoundError:
        return False
    
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
        #image_uris = sorted(glob.glob(os.path.join(IMAGE_FOLDER, 'frame_*.png')), key=lambda x: int(re.search(r'frame_(\d+).png', x).group(1)))
        image_uris = glob.glob(os.path.join(IMAGE_FOLDER, 'frame_*.png'))
        # Check if any images were found
        if not image_uris:
            print("No image files found in the folder:", IMAGE_FOLDER)
            # Optionally, handle the case where no images are found
            # For example, you might want to initialize self.image_ids and self.collection in a specific way
            self.image_ids = []
        else:
            print("*************", image_uris)
            self.image_ids = [str(i) for i in range(len(image_uris))]
        
            self.collection.add(ids=self.image_ids, uris=image_uris)

    def add_text_data(self):
        # Paths to the output files
        file1 = 'mixed_data/output_text.txt'
        file2 = 'mixed_data/captions2.txt'

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
        print("************** Chroma db created for multi data ************")

    # Naive retriever
    def query(self, query_texts, n_results, include):
        return self.collection.query(
            query_texts=query_texts,
            n_results=n_results,
            include=include
        )
    
def create_multi_store(path):
        vector_store = ChromaDBStore(
                        path=path,
                        collection_name='multimodal_collection67',
                        embedding_function=OpenCLIPEmbeddingFunction(),
                        data_loader=ImageLoader()
                    )
        return vector_store
    
def extract_documents_and_images(results):
    if results is None:
        raise ValueError("Results are None. Query might have failed or returned no results.")
    
    documents = []
    image_locations = []

    # Ensure results.get('documents') is not None
    if results.get('documents') is not None:
        for doc in results.get('documents', [[]])[0]:  # Extract text documents
            if doc:
                documents.append(doc)
    else:
        print("No documents found in results.")

    # Ensure results.get('uris') is not None
    if results.get('uris') is not None:
        for uri in results.get('uris', [[]])[0]:  # Extract image URIs
            if uri and uri.endswith('.png'):
                image_locations.append(uri)
    else:
        print("No URIs found in results.")

    return documents, image_locations

def delete_images(directory_path, file_extension='*.png'):
    # Build the path pattern to match all images with the specified extension
    pattern = os.path.join(directory_path, file_extension)
    
    # Find all files in the directory that match the pattern
    image_files = glob.glob(pattern)
    
    # Delete each file found
    for file_path in image_files:
        os.remove(file_path)
        #print(f"Deleted: {file_path}")

def save_images(image_paths, directory_path):
    """
    Loads image files from given paths and saves them to a specified directory with their original filenames.

    Parameters:
        image_paths (list of str): A list of paths to image files.
        directory_path (str): The directory where the images will be saved.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    # Loop through the list of image file paths
    for img_path in image_paths:
        try:
            # Open the image file from the path
            img = Image.open(img_path)
            # Extract the filename from the image path
            filename = os.path.basename(img_path)
            # Define the full file path for each image
            file_path = os.path.join(directory_path, filename)
            
            # Save the image
            img.save(file_path, 'PNG')  # Ensure the format matches the file extension
            print(f"Saved: {file_path}")
        except IOError as e:
            print(f"Error opening or saving image {img_path}: {e}")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_image}"

def perform_query(vector_store, query):
    #vector_store = create_multi_store(path)
    # retrive list of relevant images from vectordb
    img_results =  vector_store.query(
        query_texts=[query],
        n_results=2,
        include=["uris"],
    )
    
    # Debugging: Print img_results to see its structure
    print("Image results:", img_results)
    
    # Ensure img_results is not None
    if img_results is None:
        raise ValueError("Image results are None. Query might have failed or returned no results.")
    

    _, img = extract_documents_and_images(img_results) #get image summaries for retrieved images
    # retrive text from vector db
    text_results = vector_store.query(
                query_texts=[query],
                n_results=20,
                include=["documents", "metadatas", "distances"],
            )
    documents, _ = extract_documents_and_images(text_results) # get text summary for retrieved text

    print("$$$$$$$$$$$$$ Retrived Results:")
    #print(documents)
    directory = 'fetched_images'
    delete_images(directory)
    save_images(img, directory)
    #print("################",img)

    context_str = documents  
    #kwargs = dict({'model_name':'gemini-1.0-pro'})
    #model = GenerativeModel(proxy_client=proxy_client, **kwargs)
    model = ChatOpenAI(temperature=0, proxy_model_name='gpt-4')
    qa_tmpl_str = (
        "Given the provided information, including relevant images, retrieved context from the video, and metadata for timestamps of various frames, "
        "accurately and precisely answer the query without adding any external information.\n"
        "Ensure your response is honest and responsible, avoiding any racist or sexist remarks.\n"
        "---------------------\n"
        "Context: {context_str}\n"
        "---------------------\n"
        "Query: {query_str}\n"
        "-----------------\n"
        "Answer: "
    )
    prompt = qa_tmpl_str.format(context_str=context_str, query_str=query)
    content_list = [] 
    content_list.append(prompt) # context info of retrived text
    retrived_img = os.listdir('fetched_images')
    for image_path in retrived_img:
        img = os.path.join('fetched_images',image_path)
        print("************", img)
        enc_img = encode_image(img)
        content_list.append(enc_img) # add img info to context in base64 encoded format
    success = False
    retries = 5
    while not success and retries > 0:
        try:
            print("""*****""")
            print(content_list)
            print("********")
            response = model.invoke(content_list, stream=True)
            #response = model.generate_content(content_list)
            #response.resolve()
            success = True
            #print(response.text)
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in {6 - retries} seconds...")
            time.sleep(6 - retries)
            retries -= 1
            if retries == 0:
                raise e
    print("&&&&&&&&&&& Answer &&&&&&&", response)
    return response, context_str 

'''random_number = random.randint(1000, 9999)
path = f"test{random_number}"
create_multi_store(path)
perform_query("what is RAG",path)'''