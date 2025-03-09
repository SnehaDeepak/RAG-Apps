import gradio as gr
from moviepy.editor import VideoFileClip, AudioFileClip
from pytube import YouTube
import speech_recognition as sr
from pathlib import Path
from langchain_community.document_loaders import TextLoader
import os
import json
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
#from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.schema import ImageNode
from llama_index.core.response.notebook_utils import display_source_node
from data_indexer import DataIndexer 
from llama_index.embeddings.openai import OpenAIEmbedding
import glob
from gradio import Gallery
from PIL import Image
import random
import shutil
import lancedb
from lancedb.embeddings import EmbeddingFunctionRegistry
from PIL import Image
from lancedb.pydantic import LanceModel, Vector
from pathlib import Path
import pandas as pd
import lancedb
from lancedb.embeddings import get_registry
import pandas as pd
from random import sample
import nltk

import imageio
import os
import base64
from base64 import b64encode


from Image_caption_latest import ImageCaptioner
#from lancedb import Vector

from langchain_core.messages import HumanMessage
from vectorstore_latest import VectorStoreFactory
from pdf_extractor import PDFExtractor




OPENAI_API_TOKEN = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN

GOOGLE_API_KEY = ""  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

#Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=GOOGLE_API_KEY)
#Settings.llm = Gemini(api_key=GOOGLE_API_KEY)

from llama_index.multi_modal_llms.gemini import GeminiMultiModal
gemini_pro = GeminiMultiModal(model_name="models/gemini-pro-vision")



registry = EmbeddingFunctionRegistry.get_instance()
clip = registry.get("open-clip").create()
db = lancedb.connect("lancedb1")


#model = get_registry().get("huggingface").create(name='facebook/bart-base')
model = get_registry().get("gemini-text").create()


#video_url = "https://www.youtube.com/watch?v=d_qvLDhkg00"
output_video_path = "./video_data/"
output_folder = "./mixed_data/"
output_audio_path = "./mixed_data/output_audio.wav"

filepath = output_video_path + "input_vid.mp4"
Path(output_folder).mkdir(parents=True, exist_ok=True)
Path(output_video_path).mkdir(parents=True, exist_ok=True)




from PIL import Image
import matplotlib.pyplot as plt
import os


def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 7:
                break
                
def download_video(url, output_path):
    """
    Download a video from a given url and save it to the output path.

    Parameters:
    url (str): The url of the video to download.
    output_path (str): The path to save the video to.

    Returns:
    dict: A dictionary containing the metadata of the video.
    """
    yt = YouTube(str(url))
    metadata = {"Author": yt.author, "Title": yt.title, "Views": yt.views}
    yt.streams.get_highest_resolution().download(
        output_path=output_path, filename="input_vid.mp4"
    )
    return metadata


def video_to_images(video_path, output_folder,fps):
    """
    Convert a video to a sequence of images, save them to the output folder,
    and also store metadata including timestamps for each frame.

    Parameters:
    video_path (str): The path to the video file.
    output_folder (str): The path to the folder to save the images to.
    """
    clip = VideoFileClip(video_path)
    duration = clip.duration  # Total duration of the video in seconds
    frame_rate = fps # Frames per second
    total_frames = int(duration * frame_rate)  # Total number of frames to extract

    metadata = []

    for i, frame in enumerate(clip.iter_frames(fps=frame_rate, dtype="uint8")):
        frame_time = i / frame_rate
        frame_filename = f"frame_{i}.png"  # Filename is sequentially numbered
        frame_path = os.path.join(output_folder, frame_filename)
        imageio.imwrite(frame_path, frame)  # Save frame as image
        metadata.append({
            "frame_time": frame_time,
            "filename": frame_filename
        })

    # Description of the metadata structure
    description = "This JSON file contains metadata for each frame extracted from the video. " \
                  "Each entry lists the timestamp of the frame in seconds and the filename of the frame image."

    metadata_with_description = {
        "description": description,
        "data": metadata
    }

    metadata_filename = os.path.join(output_folder, "metadata.json")
    with open(metadata_filename, "w") as f:
        json.dump(metadata_with_description, f, indent=4)
    # Debugging: print the metadata structure before saving
    print(json.dumps(metadata_with_description, indent=4))

    print("Frames and metadata saved successfully.")
    
    


def video_to_audio(video_path, output_audio_path):
    """
    Convert a video to audio and save it to the output path.

    Parameters:
    video_path (str): The path to the video file.
    output_audio_path (str): The path to save the audio to.

    """
    clip = VideoFileClip(video_path)
    audio = clip.audio

    if audio is None:
        # Create a blank audio file
        print("No audio track found in the video file, creating a blank audio file.")
        silence = AudioFileClip("blank_audio.mp3")
        silence.write_audiofile(output_audio_path)
    else:
        audio.write_audiofile(output_audio_path)

def audio_to_text(audio_path):
    """
    Convert audio to text using the SpeechRecognition library.

    Parameters:
    audio_path (str): The path to the audio file.

    Returns:
    test (str): The text recognized from the audio.

    """
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_path)

    with audio as source:
        # Record the audio data
        audio_data = recognizer.record(source)

        try:
            # Recognize the speech
            text = recognizer.recognize_whisper(audio_data)
        except sr.UnknownValueError:
            print("Speech recognition could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from service; {e}")

    return text




def delete_images(directory_path, file_extension='*.png'):
    # Build the path pattern to match all images with the specified extension
    pattern = os.path.join(directory_path, file_extension)
    
    # Find all files in the directory that match the pattern
    image_files = glob.glob(pattern)
    
    # Delete each file found
    for file_path in image_files:
        os.remove(file_path)
        #print(f"Deleted: {file_path}")




def delete_directory(directory):
    """Deletes the specified directory along with all its contents."""
    try:
        # Remove the directory and all its contents
        shutil.rmtree(directory)
        #print(f"Directory '{directory}' has been deleted successfully.")
    except FileNotFoundError:
        print(f"Error: The directory '{directory}' does not exist.")
    except PermissionError:
        print(f"Error: Permission denied to delete the directory '{directory}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


    
def setup_retriever():
    documents = SimpleDirectoryReader(output_folder).load_data()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index.as_retriever(similarity_top_k=3, image_similarity_top_k=5)

#retriever_engine = setup_retriever()

def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)

    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            display_source_node(res_node, source_length=200)
            retrieved_text.append(res_node.text)

    return retrieved_image, retrieved_text


def image_to_data_uri(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded_string}"
    
    
def save_images1(image_paths, directory_path, file_prefix="image"):
    """
    Loads image files from given paths and saves them to a specified directory with a unique filename.

    Parameters:
        image_paths (list of str): A list of paths to image files.
        directory_path (str): The directory where the images will be saved.
        file_prefix (str): The prefix for naming the saved image files.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    # Loop through the list of image file paths
    for index, img_path in enumerate(image_paths):
        try:
            # Open the image file from the path
            img = Image.open(img_path)
            # Define the filename for each image
            filename = f"{file_prefix}_{index+1}.png"  # Save as PNG
            file_path = os.path.join(directory_path, filename)
            
            # Save the image
            img.save(file_path, 'PNG')  # Ensure the format matches the file extension
            #print(f"Saved: {file_path}")
        except IOError as e:
            print(f"Error opening or saving image {img_path}: {e}")
            
            
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


# Example usage:
# Assuming image_paths is a list of file paths to images


def check_keywords_in_response(response, keywords):
    """
    Checks if any of the specified keywords are in the response.
    
    Args:
        response (str): The response text.
        keywords (list of str): Keywords to check in the response.
        
    Returns:
        bool: True if any keyword is found, False otherwise.
    """
    return any(keyword in response for keyword in keywords)




def load_png_images_from_directory(directory_path):
    """
    Loads all PNG images from the specified directory into a list.

    Parameters:
    directory_path (str): The path to the directory containing the PNG images.

    Returns:
    list: A list of PIL Image objects.
    """
    image_list = []
    # Loop through all the files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".png"):
            # Construct the full path to the image
            image_path = os.path.join(directory_path, filename)
            # Open the image
            img = Image.open(image_path)
            # Append the image to the list
            image_list.append(img)

    return image_list

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


    

def process_video(video_file=None, video_url=None,fps=4):
    try:
        #delete_directory("lancedb")
        delete_directory("mixed_data")
        #delete_directory("video_data")
        output_video_path = "./video_data/"
        output_folder = "./mixed_data/"
        output_audio_path = "./mixed_data/output_audio.wav"
        
        filepath = output_video_path + "input_vid.mp4"
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        Path(output_video_path).mkdir(parents=True, exist_ok=True)
        
        # Check if a URL is provided and download video accordingly
        if video_url is not None:
            metadata = download_video(video_url, output_video_path)
            file_path = os.path.join(output_video_path, "input_vid.mp4")
        else:
            # Use the provided video file if URL is not provided
            file_path = video_file
            metadata = {'file_path': file_path}  # You might want to include additional metadata manually
        
        video_to_images(file_path, output_folder,fps)
        video_to_audio(file_path, output_audio_path)
        text_data = audio_to_text(output_audio_path)

        with open(output_folder + "output_text.txt", "w") as file:
            file.write(text_data)
        print("Text data saved to file")
        file.close()
        os.remove(output_audio_path)
        print("Audio file removed")
        with open(os.path.join(output_video_path, "metadata.json"), "w") as meta_file:
            json.dump(metadata, meta_file, indent=4)
            print(metadata)
        print("-------------------------------------------------------")
        print("Video processing finished!!")
        print("-------------------------------------------------------")
        #captioner = ImageCaptioner()
        #captioner.run("mixed_data")

    except Exception as e:
        raise e
    
    if len(text_data) < 25:
        text_data = "Video transcript not available as it has no audio file"
        captioner = ImageCaptioner()
        captioner.run("mixed_data")
    print(text_data)
    print(metadata)
    return metadata, text_data

def process_pdf_file(pdf_file):
    text_data ="NA"
    delete_directory("mixed_data")  
    output_folder = "./mixed_data/"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    with open(output_folder + "output_text.txt", "w") as file:
            file.write(text_data)
    pdf_extractor = PDFExtractor()
    extracted_text, image_paths = pdf_extractor.process_pdf(pdf_file.name)
    print("Extracted Text:")
    print(extracted_text)
    print("\nExtracted Images:")
    for image_path in image_paths:
        print(image_path)
    return "PDF processed successfully."
    




def initialize_vector_store(vector_store_type, retriever_type=None):
    global vector_store, current_vector_store_type, current_retriever_type

    current_vector_store_type = vector_store_type
    current_retriever_type = retriever_type

    # Handle directory deletion if the vector store type is chroma
    if vector_store_type == "chroma":
        db_path = "test2"
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            print(f"Deleted the database directory at {db_path}.")
        else:
            print(f"No database directory found at {db_path} to delete.")
        vector_store = VectorStoreFactory.create_vector_store(vector_store_type, retriever_type)
    elif vector_store_type == "lance":
        vector_store = VectorStoreFactory.create_vector_store(vector_store_type,retriever_type="naive")

    return f"Initialized {vector_store_type} with {retriever_type if retriever_type else 'default settings'}"


    
import os
import time
import json
from PIL import Image
import google.generativeai as genai

# Set the environment variable (ensure to set this in your actual environment)
os.environ["GEMINI_API_KEY"] = "AIzaSyA38G_UIGzKQbELOLdtXajXaBIRNrgz9I0"  

# Configure the API with the environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}



def is_image_file(filename):
    """
    Check if a file is an image based on its extension.
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def natural_sort_key(filename):
    # Extract the number from the filename
    num_part = ''.join(filter(str.isdigit, filename))
    return int(num_part) if num_part else float('inf')

def delete_images(directory):
    # Function to delete images in a directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def save_images(images, directory):
    # Function to save images in a directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    for idx, image_path in enumerate(images):
        with open(image_path, "rb") as img_file:
            image_data = img_file.read()
        save_path = os.path.join(directory, f"image_{idx}.jpg")
        with open(save_path, "wb") as save_file:
            save_file.write(image_data)

def check_keywords_in_response(response, keywords):
    # Function to check if any keywords are present in the response
    return any(keyword in response for keyword in keywords)

    
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}
        
global_question = ""
global_context = ""
global_generated_output = ""



def perform_query(query):
    global global_question, global_context, global_generated_output
    global vector_store, current_vector_store_type, current_retriever_type
    output_video_path = "./video_data/"
    output_folder = "./mixed_data/"
    output_audio_path = "./mixed_data/output_audio.wav"

    if 'vector_store' not in globals() or vector_store is None:
        raise ValueError("Vector store is not initialized. Call initialize_vector_store first.")
        
    naive_retriever = VectorStoreFactory.create_vector_store("chroma", "naive")
    img_results =  naive_retriever.query(
        query_texts=[query],
        n_results=10,
        include=["uris"],
    )
    
    # Debugging: Print img_results to see its structure
    print("Image results:", img_results)
    
    # Ensure img_results is not None
    if img_results is None:
        raise ValueError("Image results are None. Query might have failed or returned no results.")
    

    _, img = extract_documents_and_images(img_results)

    # Use selected retriever for text extraction
   # Use selected retriever for text extraction
    documents = []
    if current_vector_store_type == "chroma":
        if current_retriever_type in ["parent_document", "contextual_compression"]:
            text_results = vector_store.get_retrieved_documents(query)
            print(f"Retrieved {len(text_results)} documents for query: {query}")
            for doc in text_results:
                if isinstance(doc, dict):  # Assuming doc is a dictionary
                    documents.append(doc['page_content'])
                    print(f"Text Document: {doc['page_content']}")
                elif hasattr(doc, 'page_content'):  # Check for page_content attribute
                    documents.append(doc.page_content)
                    print(f"Text Document: {doc.page_content}")
                else:
                    print(f"Retrieved object is not an instance of Document or a dictionary: {doc}")
            
            if not documents:
                print("No documents were appended to the documents list.")
            else:
                print(f"Appended {len(documents)} documents to the documents list.")
        else:
            text_results = vector_store.query(
                query_texts=[query],
                n_results=20,
                include=["documents", "metadatas", "distances"],
            )
            documents, _ = extract_documents_and_images(text_results)

    elif current_vector_store_type == "lance":
        print("LanceDB Results:")
        text_results = vector_store.query(
            query_texts=[query],
            n_results=20,
            include=["documents"]
        )
        if text_results:
            documents = text_results['combined_text']
        else:
            print("No results found.")
            documents = []

    print("Results:")
    print(documents)
    directory = 'fetched_images'
    delete_images(directory)
    save_images(img, directory)
    print(img)

    with open('mixed_data/metadata.json', 'r') as file:
        metadata_str = json.load(file)
 
    context_str = documents   
    print("*******************************************")
    #print(context_str)
    print("******************************************")
   
    os.environ["GEMINI_API_KEY"] = "AIzaSyD9FRF932rrA79k17EoTLf11LwOQqOjhBM"

    # Configure the API with the environment variable
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Load safety settings
    safety_settings = load_safety_settings()
 

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        safety_settings=safety_settings,
        generation_config=generation_config,
    )



    qa_tmpl_str = (
        "Given the provided information, including relevant images, retrieved context from the video, and metadata for timestamps of various frames, "
        "accurately and precisely answer the query without adding any external information.\n"
        "Ensure your response is honest and responsible, avoiding any racist or sexist remarks.\n"
        "---------------------\n"
        "Context: {context_str}\n"
        "---------------------\n"
        "Query: {query_str}\n"
        "Also look at the images and try to explain what you see in the images based on Safety settings \n"
        "For safety settings, consider the following thresholds:\n"
        "- If the threshold is BLOCK_LOW_AND_ABOVE for any of the HARASSMENT, HATE SPEECH, SEXUALLY EXPLICIT content, or DANGEROUS CONTENT in saftety setting, then restrict from giving any response if you find any serious and explicit content in the images or context based on respectives category HARASSMENT, HATE SPEECH, SEXUALLY EXPLICIT content, or DANGEROUS CONTENT. Just say 'Can't comment on that because of safety measure'\n"
        "- If the threshold is BLOCK_NONE, you may give a detail response regardless of the content seen in the images or context.\n"
        "- If the threshold is BLOCK_ONLY_HIGH, give a response if the content is not extremely violent or explicit, otherwise restrict your response.\n"
        "Safety settings: {safety_settings}.\n"
        "Answer: "
    )


    prompt = qa_tmpl_str.format(context_str=context_str, query_str=query,safety_settings=safety_settings)
    content_list = []
    content_list.append(prompt)
    images = [Image.open(image_path) for image_path in img]
    for img in images:
         content_list.append(img)
    success = False
    retries = 5
    while not success and retries > 0:
        try:
            print("""*****""")
            print(content_list)
            print("********")
            response = model.generate_content(content_list, stream=True)
            response.resolve()
            success = True
            print(response.text)
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in {6 - retries} seconds...")
            time.sleep(6 - retries)
            retries -= 1
            if retries == 0:
                raise e

    keywords = ["safety measure", "Information not provided", "not provided", "does not provide", "Information not provided in video", "The video does not provide any information"]

    if check_keywords_in_response(response.text, keywords):
        delete_images("fetched_images")
    
    # Set global variables for the evaluation
    global_question = query
    global_context = context_str
    global_generated_output = response.text
    safety_settings = load_safety_settings()
    print(safety_settings)
    

    return response.text



#metadata, text= process_video(video_file="video_data/Arrest013_x264.mp4", video_url=None)
#response=perform_query("At what time or the timestamps and date the arrest is happening can you look into the cctv footage ")

#metadata, text= process_video(video_file="video_data/Arson024_x264.mp4", video_url=None)

def perform_query_pdf(query):
    global global_question, global_context, global_generated_output
    global vector_store, current_vector_store_type, current_retriever_type
   

    if 'vector_store' not in globals() or vector_store is None:
        raise ValueError("Vector store is not initialized. Call initialize_vector_store first.")
        
    naive_retriever = VectorStoreFactory.create_vector_store("chroma", "naive")
    img_results =  naive_retriever.query(
        query_texts=[query],
        n_results=2,
        include=["uris"],
    )
    
    # Debugging: Print img_results to see its structure
    print("Image results:", img_results)
    
    # Ensure img_results is not None
    if img_results is None:
        raise ValueError("Image results are None. Query might have failed or returned no results.")
    

    _, img = extract_documents_and_images(img_results)

    # Use selected retriever for text extraction
   # Use selected retriever for text extraction
    documents = []
    if current_vector_store_type == "chroma":
        if current_retriever_type in ["parent_document", "contextual_compression"]:
            text_results = vector_store.get_retrieved_documents(query)
            print(f"Retrieved {len(text_results)} documents for query: {query}")
            for doc in text_results:
                if isinstance(doc, dict):  # Assuming doc is a dictionary
                    documents.append(doc['page_content'])
                    print(f"Text Document: {doc['page_content']}")
                elif hasattr(doc, 'page_content'):  # Check for page_content attribute
                    documents.append(doc.page_content)
                    print(f"Text Document: {doc.page_content}")
                else:
                    print(f"Retrieved object is not an instance of Document or a dictionary: {doc}")
            
            if not documents:
                print("No documents were appended to the documents list.")
            else:
                print(f"Appended {len(documents)} documents to the documents list.")
        else:
            text_results = vector_store.query(
                query_texts=[query],
                n_results=2,
                include=["documents", "metadatas", "distances"],
            )
            documents, _ = extract_documents_and_images(text_results)

    elif current_vector_store_type == "lance":
        print("LanceDB Results:")
        text_results = vector_store.query(
            query_texts=[query],
            n_results=2,
            include=["documents"]
        )
        if text_results:
            documents = text_results['combined_text']
        else:
            print("No results found.")
            documents = []

    print("Results:")
    print(documents)
    directory = 'fetched_images'
    delete_images(directory)
    save_images(img, directory)
    print(img)

   
 
    context_str = documents   
    print("*******************************************")
    #print(context_str)
    print("******************************************")
   
    os.environ["GEMINI_API_KEY"] = "AIzaSyD9FRF932rrA79k17EoTLf11LwOQqOjhBM"

    # Configure the API with the environment variable
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Load safety settings
    safety_settings = load_safety_settings()
 

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        safety_settings=safety_settings,
        generation_config=generation_config,
    )



    qa_tmpl_str = (
        "Given the provided information, including relevant images, retrieved context from the pdf, "
        "accurately and precisely answer the query without adding any external information.\n"
        "Ensure your response is honest and responsible, avoiding any racist or sexist remarks.\n"
        "---------------------\n"
        "Context: {context_str}\n"
        "---------------------\n"
        "Query: {query_str}\n"
        "Also look at the images and try to explain what you see in the images based on Safety settings \n"
        "For safety settings, consider the following thresholds:\n"
        "- If the threshold is BLOCK_LOW_AND_ABOVE for any of the HARASSMENT, HATE SPEECH, SEXUALLY EXPLICIT content, or DANGEROUS CONTENT in saftety setting, then restrict from giving any response if you find any serious and explicit content in the images or context based on respectives category HARASSMENT, HATE SPEECH, SEXUALLY EXPLICIT content, or DANGEROUS CONTENT. Just say 'Can't comment on that because of safety measure'\n"
        "- If the threshold is BLOCK_NONE, you may give a detail response regardless of the content seen in the images or context.\n"
        "- If the threshold is BLOCK_ONLY_HIGH, give a response if the content is not extremely violent or explicit, otherwise restrict your response.\n"
        "Safety settings: {safety_settings}.\n"
        "Answer: "
    )


    prompt = qa_tmpl_str.format(context_str=context_str, query_str=query,safety_settings=safety_settings)
    content_list = []
    content_list.append(prompt)
    images = [Image.open(image_path) for image_path in img]
    for img in images:
         content_list.append(img)
    success = False
    retries = 5
    while not success and retries > 0:
        try:
            print("""*****""")
            print(content_list)
            print("********")
            response = model.generate_content(content_list, stream=True)
            response.resolve()
            success = True
            print(response.text)
        except Exception as e:
            print(f"Error occurred: {e}. Retrying in {6 - retries} seconds...")
            time.sleep(6 - retries)
            retries -= 1
            if retries == 0:
                raise e

    keywords = ["safety measure", "Information not provided", "not provided", "does not provide", "Information not provided in video", "The video does not provide any information"]

    if check_keywords_in_response(response.text, keywords):
        delete_images("fetched_images")
    
    # Set global variables for the evaluation
    global_question = query
    global_context = context_str
    global_generated_output = response.text
    safety_settings = load_safety_settings()
    print(safety_settings)
    

    return response.text







def display_images(directory_path='fetched_images'):
    image_paths = glob.glob(os.path.join(directory_path, '*.png')) + glob.glob(os.path.join(directory_path, '*.jpg'))
    if not image_paths:
        return "Images related to query could not be found!! Try with different query", False
    images = [os.path.join(directory_path, os.path.basename(img_path)) for img_path in image_paths]
    return images, True


def show_images(directory_path):
    result, success = display_images(directory_path)
    if not success:
        return [], result  # Return empty list for images and error message
    else:
        return result, "Here are your related images from video"  # Return image paths and empty message
    
def set_vector_store_type(selected_type):
    global vector_store_type
    vector_store_type = selected_type
    initialize_vector_store(vector_store_type)

import json
import os


import os
import json

from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
import os
import json
from google.generativeai import generative_models
from google.generativeai.types import HarmCategory, HarmBlockThreshold

def map_safety_setting(value):
    return {
        "Block none": HarmBlockThreshold.BLOCK_NONE,
        "Block few": HarmBlockThreshold.BLOCK_ONLY_HIGH,
        "Block some": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        "Block most": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    }.get(value, HarmBlockThreshold.BLOCK_NONE)

def map_safety_setting_reverse(value):
    return {
        HarmBlockThreshold.BLOCK_NONE: "Block none",
        HarmBlockThreshold.BLOCK_ONLY_HIGH: "Block few",
        HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE: "Block some",
        HarmBlockThreshold.BLOCK_LOW_AND_ABOVE: "Block most"
    }.get(value, "Block none")

def save_safety_settings(enable_safety, harassment, hate_speech, sexually_explicit, dangerous_content):
    if enable_safety:
        safety_settings = [
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT.name,
                "threshold": map_safety_setting(harassment).name
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH.name,
                "threshold": map_safety_setting(hate_speech).name
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT.name,
                "threshold": map_safety_setting(sexually_explicit).name
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT.name,
                "threshold": map_safety_setting(dangerous_content).name
            }
        ]
    else:
        safety_settings = []

    # Save settings to a file
    with open("safety_settings.json", "w") as file:
        json.dump(safety_settings, file)
        
    return "Selected safety setting enabled"

def load_safety_settings():
    if os.path.exists("safety_settings.json"):
        with open("safety_settings.json", "r") as file:
            settings = json.load(file)
            print(f"Loaded settings from file: {settings}")
            safety_settings = {
                HarmCategory[setting["category"]]: HarmBlockThreshold[setting["threshold"]]
                for setting in settings
            }
            return safety_settings
    return {}

# Save default safety settings to the file if it does not exist
if not os.path.exists("safety_settings.json"):
    save_safety_settings(
        enable_safety=True,
        harassment="Block most",
        hate_speech="Block most",
        sexually_explicit="Block most",
        dangerous_content="Block most"
    )

# Set the environment variable (ensure to set this in your actual environment)
os.environ["GEMINI_API_KEY"] = "AIzaSyD9FRF932rrA79k17EoTLf11LwOQqOjhBM"

# Configure the API with the environment variable
import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load safety settings
safety_settings = load_safety_settings()


# Correctly define the generation config as a dictionary
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    safety_settings=safety_settings,
    generation_config=generation_config,
)


css = """
/* Your custom CSS here */
"""

# Define the options for retrievers
retriever_options = ["naive", "parent_document", "contextual_compression"]



# Function to update retriever dropdown based on vector store selection
def update_retriever_options(vector_store_type):
    if vector_store_type == "chroma":
        return gr.update(choices=retriever_options, visible=True)
    else:
        return gr.update(choices=[], visible=False)

safety_settings = load_safety_settings()


# After processing the video, define the model with the safety settings
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    safety_settings=safety_settings,
    generation_config=generation_config,
)

def process_video_interface(video_file, video_url, fps):
    if video_url:
        return process_video(video_url=video_url, fps=fps)
    elif video_file is not None:
        return process_video(video_file=video_file.name, fps=fps)
    else:
        return {}, "No video provided"

from evaluation_module import RAGEvaluator

# Initialize evaluator
evaluator = RAGEvaluator()
    
def evaluate_response(question, context, generated_output):
    if question and context and generated_output:
        metrics = evaluator.evaluate_all(generated_output, context)
        return (
            metrics['BLEU'],
            metrics['ROUGE-1'],
            metrics['BERT P'],
            metrics['BERT R'],
            metrics['BERT F1'],
            metrics['Perplexity'],
            metrics['Diversity'],
            metrics['Racial Bias'],
        )
    else:
        return "Please provide all inputs to evaluate."
        
with gr.Blocks(css=css) as demo:
   
    with gr.Tab("RAG with Multi-modal LLM"):
        gr.Markdown("# Application Architecture")

        image_path = 'Untitled Diagram.drawio.png' 

        if os.path.exists(image_path):
            image_display = image_path   
        else:
            image_display = None

        gr.Image(value=image_display, label="Image from Path")


    with gr.Tab("Multimodal Application"):
        gr.Markdown("# Multimodal RAG with Video Data")

        with gr.Row():
            video_url = gr.Textbox(label="Enter Video URL")
            video_file = gr.File(label="Upload your video file", type="filepath")
            fps = gr.Number(label="Frames per Second (FPS)", value=4, precision=0)

        enable_safety = gr.Checkbox(label="Enable Extra Layer of Content Safety towards RAI")

        with gr.Row(visible=False) as safety_options_row:
            harm_category_harassment = gr.Radio(label="Harassment", choices=["Block none", "Block few", "Block most"])
            harm_category_hate_speech = gr.Radio(label="Hate Speech", choices=["Block none", "Block few", "Block most"])
            harm_category_sexually_explicit = gr.Radio(label="Sexually Explicit", choices=["Block none", "Block few", "Block most"])
            harm_category_dangerous_content = gr.Radio(label="Dangerous Content", choices=["Block none", "Block few","Block most"])

        enable_safety.change(
            fn=lambda enabled: gr.update(visible=enabled),
            inputs=[enable_safety],
            outputs=[safety_options_row]
        )

        save_safety_button = gr.Button("Save Safety Settings")
        save_safety_message = gr.Textbox(label="Save Status", interactive=False)

        save_safety_button.click(
            fn=save_safety_settings,
            inputs=[enable_safety, harm_category_harassment, harm_category_hate_speech, harm_category_sexually_explicit, harm_category_dangerous_content],
            outputs=[save_safety_message]
        )

        process_button = gr.Button("Process Video")
        output_metadata = gr.JSON(label="Metadata")
        output_text = gr.Textbox(label="Video Transcript")

        def process_video_interface(video_file, video_url, fps):
            if video_url:
                return process_video(video_url=video_url, fps=fps)
            elif video_file is not None:
                return process_video(video_file=video_file.name, fps=fps)
            else:
                return {}, "No video provided"

        process_button.click(
            fn=process_video_interface,
            inputs=[video_file, video_url, fps],
            outputs=[output_metadata, output_text]
        )

        with gr.Row():
            vector_store_type_dropdown = gr.Dropdown(label="Select Vector Store Type", choices=["chroma", "lance"], value="chroma")
            retriever_type_dropdown = gr.Dropdown(label="Select Retriever Type", choices=retriever_options, visible=True)
            initialize_button = gr.Button("Initialize Vector Store")
            status_message = gr.Textbox(label="Status", interactive=False)

        vector_store_type_dropdown.change(
            fn=update_retriever_options,
            inputs=[vector_store_type_dropdown],
            outputs=[retriever_type_dropdown]
        )

        initialize_button.click(
            fn=initialize_vector_store,
            inputs=[vector_store_type_dropdown, retriever_type_dropdown],
            outputs=[status_message]
        )

        query = gr.Textbox(label="Enter your query about the video")
        query_button = gr.Button("Chat with Video")
        output_response = gr.Textbox(label="Response")

        query_button.click(
            fn=perform_query,
            inputs=[query],
            outputs=[output_response]
        )

        directory_input = gr.Textbox(value='fetched_images', label="Directory Path")
        btn_display_images = gr.Button("Display Related Images to Video")
        gallery = gr.Gallery()
        error_label = gr.Label()

        btn_display_images.click(
            fn=show_images,
            inputs=[directory_input],
            outputs=[gallery, error_label]
        )
    
    with gr.Tab("Multimodal LLM for PDF"):
        gr.Markdown("# Multimodal RAG with PDF")

        with gr.Row():
        
            #video_file = gr.File(label="Upload your pdf file", type="filepath")
      
            pdf_file = gr.File(label="Upload your PDF file", type="filepath")
         

        enable_safety = gr.Checkbox(label="Enable Extra Layer of Content Safety towards RAI")

        with gr.Row(visible=False) as safety_options_row:
            harm_category_harassment = gr.Radio(label="Harassment", choices=["Block none", "Block few", "Block most"])
            harm_category_hate_speech = gr.Radio(label="Hate Speech", choices=["Block none", "Block few", "Block most"])
            harm_category_sexually_explicit = gr.Radio(label="Sexually Explicit", choices=["Block none", "Block few", "Block most"])
            harm_category_dangerous_content = gr.Radio(label="Dangerous Content", choices=["Block none", "Block few","Block most"])

        enable_safety.change(
            fn=lambda enabled: gr.update(visible=enabled),
            inputs=[enable_safety],
            outputs=[safety_options_row]
        )

        save_safety_button = gr.Button("Save Safety Settings")
        save_safety_message = gr.Textbox(label="Save Status", interactive=False)

        save_safety_button.click(
            fn=save_safety_settings,
            inputs=[enable_safety, harm_category_harassment, harm_category_hate_speech, harm_category_sexually_explicit, harm_category_dangerous_content],
            outputs=[save_safety_message]
        )

        process_button = gr.Button("Process PDF")
        
        process_button.click(   
            fn=process_pdf_file,
            inputs=pdf_file,
            outputs=gr.Textbox(label="Status")
        )



        with gr.Row():
            vector_store_type_dropdown = gr.Dropdown(label="Select Vector Store Type", choices=["chroma", "lance"], value="chroma")
            retriever_type_dropdown = gr.Dropdown(label="Select Retriever Type", choices=retriever_options, visible=True)
            initialize_button = gr.Button("Initialize Vector Store")
            status_message = gr.Textbox(label="Status", interactive=False)

        vector_store_type_dropdown.change(
            fn=update_retriever_options,
            inputs=[vector_store_type_dropdown],
            outputs=[retriever_type_dropdown]
        )

        initialize_button.click(
            fn=initialize_vector_store,
            inputs=[vector_store_type_dropdown, retriever_type_dropdown],
            outputs=[status_message]
        )

        query = gr.Textbox(label="Enter your query about the PDF")
        query_button = gr.Button("Chat with PDF")
        output_response = gr.Textbox(label="Response")

        query_button.click(
            fn= perform_query_pdf,
            inputs=[query],
            outputs=[output_response]
        )

        directory_input = gr.Textbox(value='fetched_images', label="Directory Path")
        btn_display_images = gr.Button("Display Related Images to PDF")
        gallery = gr.Gallery()
        error_label = gr.Label()

        btn_display_images.click(
            fn=show_images,
            inputs=[directory_input],
            outputs=[gallery, error_label]
        )
        
        
    with gr.Tab("RAG System Evaluation Dashboard"):
        gr.Markdown("# RAG System Evaluation Dashboard")

        question = gr.Textbox(label="Question", value="", interactive=False)
        context = gr.Textbox(label="Reference Context (top 'k' documents)", value="", interactive=False)
        generated_output = gr.Textbox(label="LLM Generated Output", value="", interactive=False)

        update_evaluation_button = gr.Button("Update Evaluation Inputs")


        update_evaluation_button.click(
            fn=lambda: (global_question, global_context, global_generated_output),
            outputs=[question, context, generated_output]
        )

        evaluate_button = gr.Button("Evaluate")
        bleu_score = gr.Textbox(label="BLEU Score (Low: <0.10, Medium: 0.10-0.29, High: ≥0.30)")
        rouge1_score = gr.Textbox(label="ROUGE-1 Score (Good: 0.6-1, Acceptable: 0.3-0.6, Poor: 0-0.3)")
        bert_p_score = gr.Textbox(label="BERT Precision (Good: 0.8-1, Acceptable: 0.5-0.8, Poor: 0-0.5)")
        bert_r_score = gr.Textbox(label="BERT Recall (Good: 0.8-1, Acceptable: 0.5-0.8, Poor: 0-0.5)")
        bert_f1_score = gr.Textbox(label="BERT F1 Score (Good: 0.8-1, Acceptable: 0.5-0.8, Poor: 0-0.5)")
        perplexity_score = gr.Textbox(label="Perplexity (Good: 0-20, Acceptable: 20-60, Poor: 60 and above)")
        diversity_score = gr.Textbox(label="Diversity (Good: 0.5-1, Acceptable: 0.3-0.5, Poor: 0-0.3)")
        racial_bias_score = gr.Textbox(label="Racial Bias (Good: 0-0.1, Acceptable: 0.1-0.3, Poor: 0.3 and above)")

        
        evaluate_button.click(
            fn=evaluate_response,
            inputs=[question, context, generated_output],
            outputs=[bleu_score, rouge1_score, bert_p_score, bert_r_score, bert_f1_score, perplexity_score, diversity_score, racial_bias_score]
        )
        

     
demo.launch(share=True)
