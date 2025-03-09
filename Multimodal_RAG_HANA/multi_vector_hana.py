import os
import glob
import random
import json
import shutil
import time
import base64
import re
import io
from io import BytesIO
from PIL import Image
from langchain_community.vectorstores.hanavector import HanaDB
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from hdbcli import dbapi
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import time
import base64
from PIL import Image
import google.generativeai as genai
os.environ["GEMINI_API_KEY"] = "AIzaSyA38G_UIGzKQbELOLdtXajXaBIRNrgz9I0"  
proxy_client = get_proxy_client('gen-ai-hub')

def hana_connect():
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
    print("HANA Connection successful")
    return connection

def partition_pdf():
    file2 = 'mixed_data/captions2.txt'
    with open(file2, 'r', encoding='utf-8') as file:
        text = file.read()
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=4000, chunk_overlap=0
        )
        texts_4k_token = text_splitter.split_text(text)
        tables=[]

    '''all_texts = []

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        tables =[]
        texts = [d.page_content for d in docs]

        # Optional: Enforce a specific token size for texts
        text_splitter = CharacterTextSplitter.from_token_encoder(
            chunk_size=4000, chunk_overlap=0
        )
        joined_texts = " ".join(texts)
        texts_4k_token = text_splitter.split_text(joined_texts)
        
        all_texts.extend(texts_4k_token)'''

    return texts_4k_token, tables, text

# Generate summaries of text elements
def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    Summarize text elements
    texts: List of str
    tables: List of str
    summarize_texts: Bool to summarize texts
    """

    # Prompt
    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements. \
    Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Text summary chain
    model = ChatOpenAI(temperature=0, proxy_model_name='gpt-4')
    #model = ChatOpenAI(temperature=0, model="gpt-4")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Initialize empty summaries
    text_summaries = []
    table_summaries = []

    # Apply to text if texts are provided and summarization is requested
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts

    # Apply to tables if tables are provided
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    return text_summaries, table_summaries

#image summaries
def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt):
    """Make image summary"""
    #chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)
    chat = ChatOpenAI(proxy_model_name='gpt-4', max_tokens=1024)

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content


def generate_img_summaries(path):
    """
    Generate summaries and base64 encoded strings for images
    path: Path to list of .jpg files extracted by Unstructured
    """

    # Store base64 encoded images
    img_base64_list = []

    # Store image summaries
    image_summaries = []

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""

    # Apply to images
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".png"):
            print(img_file)
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            #summ = image_summarize(base64_image, prompt)
            #image_summaries.append(summ)

    return img_base64_list

#image summary with captioning
class ImageCaptioner:
    def __init__(self, model_name="gemini-1.5-pro-latest", batch_size=100):
        self.batch_size = batch_size
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 0.5,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        self.model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
    
    def is_image_file(self, filename):
        """
        Check if a file is an image based on its extension.
        """
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        return any(filename.lower().endswith(ext) for ext in valid_extensions)
    
    def natural_sort_key(self, filename):
        # Extract the number from the filename
        num_part = ''.join(filter(str.isdigit, filename))
        return int(num_part) if num_part else float('inf')
    
    def process_images(self, directory):
        """
        Process images in batches, convert them to PIL Image objects, and invoke an LLM for captioning.
        """
        files = [f for f in sorted(os.listdir(directory), key=self.natural_sort_key) if os.path.isfile(os.path.join(directory, f)) and self.is_image_file(f)]
        total_files = len(files)
        all_captions = []
    
        for i in range(0, total_files, self.batch_size):
            earlier_response=[]
            batch_files = files[i:i+self.batch_size]
            images = [Image.open(os.path.join(directory, f)) for f in batch_files]

            content_list = []
            for img in images:
                content_list.append(img)

            prompt = (
        f"Given the sequence of images extracted from a video, your task is to analyze each frame meticulously and compose a comprehensive and accurate summary. "
        "Ensure that the summary reflects the exact details visible in the images without adding any external information or assumptions. Focus on capturing the progression of events, key actions, and significant details depicted in the frames. "
        "Additionally, if any uncertain or unusual events occur, describe these anomalies clearly in your summary"
        f"Consider the following as the memory and result of earlier images to maintain continuity in the new responses: {earlier_response}. So, Start from where the previous story ended and form an accurate summary by taking into consideration the earlier responses and hence there should not be a drastic mismatch and discontinuty from previous response."
            )


            content_list.insert(0, prompt)

            success = False
            retries = 1
            while not success and retries > 0:
                try:
                    response = self.model.generate_content(content_list, stream=True)
                    response.resolve()
                    print(response.text)
                    all_captions.append(response.text)  # Collecting all captions
                    earlier_response.append(response.text) 
                    success = True
                except Exception as e:
                    print(f"Error occurred: {e}. Retrying in {63 - retries} seconds...")
                    time.sleep(63 - retries)
                    retries -= 1
                    if retries == 0:
                        raise e

        return all_captions
    
    def save_captions(self, captions, output_file='captions4.txt'):
        with open(output_file, 'w') as file:
            for caption in captions:
                file.write(caption + '\n')
        print(f"Captions have been saved to {output_file}")

    def run(self, directory):
        captions = self.process_images(directory)
        return captions
        #self.save_captions(captions, output_file)

#vector store
def create_multi_vector_retriever(
    vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """

    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    # Add texts, tables, and images
    # Check that text_summaries is not empty before adding
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    # Check that table_summaries is not empty before adding
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    # Check that image_summaries is not empty before adding
    if image_summaries:
        add_documents(retriever, image_summaries, images)

    return retriever

#build RAG retriver
def plt_img_base64(img_base64):
    """Disply base64 encoded string as image"""
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    display(HTML(image_html))


def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}


def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    '''if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)'''

    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide answers related to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


def multi_modal_rag_chain(retriever):
    """
    Multi-modal RAG chain
    """

    # Multi-modal LLM
    #model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)
    model = ChatOpenAI(temperature=0, proxy_model_name='gpt-4', max_tokens=1024)

    # RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain

def base64_to_image(base64_string):
    # Remove the data URI prefix if present
    if "data:image" in base64_string:
        base64_string = base64_string.split(",")[1]

    # Decode the Base64 string into bytes
    image_bytes = base64.b64decode(base64_string)
    return image_bytes

def create_image_from_bytes(image_bytes):
    # Create a BytesIO object to handle the image data
    image_stream = BytesIO(image_bytes)

    # Open the image using Pillow (PIL)
    image = Image.open(image_stream)
    return image

def delete_images(directory_path, file_extension='*.png'):
    # Build the path pattern to match all images with the specified extension
    pattern = os.path.join(directory_path, file_extension)
    
    # Find all files in the directory that match the pattern
    image_files = glob.glob(pattern)
    
    # Delete each file found
    for file_path in image_files:
        os.remove(file_path)
        #print(f"Deleted: {file_path}")

def is_base64_encoded(s):
    """Check if the string is a valid Base64 encoding."""
    try:
        # Try to decode the Base64 string
        base64.b64decode(s, validate=True)
        return True
    except (TypeError, ValueError):
        return False

def is_image_base64(encoded_str):
    """Check if the Base64 string represents a valid image."""
    try:
        # Decode the Base64 string and attempt to open it as an image
        image_data = base64.b64decode(encoded_str)
        with BytesIO(image_data) as img_file:
            Image.open(img_file).verify()  # Verify if it's a valid image
        return True
    except (base64.binascii.Error, IOError):
        return False

def save_base64_image(encoded_str, file_path):
    """Save a Base64-encoded image as a PNG file."""
    try:
        image_data = base64.b64decode(encoded_str)
        with BytesIO(image_data) as img_file:
            with Image.open(img_file) as img:
                img.save(file_path, format='PNG')
        print(f"Image saved as {file_path}")
    except (base64.binascii.Error, IOError) as e:
        print(f"Failed to save image: {e}")

def check_and_save_image_list(base64_list, output_folder):
    """Check if the list contains valid Base64-encoded images and save them as PNG files."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, item in enumerate(base64_list):
        if isinstance(item, str) and is_base64_encoded(item):
            if is_image_base64(item):
                file_path = os.path.join(output_folder, f'image_{idx}.png')
                save_base64_image(item, file_path)
            else:
                print(f"Base64 string at index {idx} is not a valid image.")
        else:
            print(f"Item at index {idx} is not a valid Base64 string.")

def perform_query_multi(query, vectorstore,text_summaries, texts,
                        table_summaries, tables,
                        image_summaries, img_base64_list):
        # Create retriever
    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        texts,
        table_summaries,
        tables,
        image_summaries,
        img_base64_list,
    )
    # Create RAG chain
    chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)
    # Run RAG chain
    response = chain_multimodal_rag.invoke(query)
    print("***** Response ******", response)
    docs = retriever_multi_vector_img.invoke(query, limit=6)
    print("&&&&&&&&&&&&&& list of retrived images &&&&&", docs)
    #print("$$$$$$$$$", docs)
    delete_images('fetched_images')
    check_and_save_image_list(docs, 'fetched_images')
    #shutil.rmtree('fetched_images', ignore_errors=True)
    #os.makedirs('fetched_images', exist_ok=True)
    #img.save(os.path.join('fetched_images','ret_img.png'))
    return response

'''connection = hana_connect()
all_texts, tables, texts = partition_pdf()
# Get text, table summaries
text_summaries, table_summaries = generate_text_summaries(
    all_texts, tables, summarize_texts=True
)
print("text summary done")
# Image summaries
img_base64_list = generate_img_summaries('mixed_data')
captioner = ImageCaptioner()
image_summaries = list(captioner.run("mixed_data"))
# The vectorstore to use to index the summaries
embedding_model = OpenAIEmbeddings(proxy_model_name='text-embedding-ada-002', proxy_client=proxy_client)
vectorstore = HanaDB(embedding=embedding_model, connection=connection, table_name="RAG_POC_CDS")
vectorstore.delete(filter={})

# Create retriever
retriever_multi_vector_img = create_multi_vector_retriever(
    vectorstore,
    text_summaries,
    texts,
    table_summaries,
    tables,
    image_summaries,
    img_base64_list,
)
# Create RAG chain
chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)
# Run RAG chain
response = chain_multimodal_rag.invoke("what is Retriver Augmented Generation")
print("***** Response ******", response)
docs = retriever_multi_vector_img.invoke("what is RAG", limit=6)
#print("$$$$$$$$$", docs)
# Convert Base64 to image bytes
image_bytes = base64_to_image(docs[0])
# Create an image from bytes
img = create_image_from_bytes(image_bytes)
img.save(os.path.join('fetched_images','ret_img.png'))'''


