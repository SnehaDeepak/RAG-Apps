import os
import time
import base64
from PIL import Image
import google.generativeai as genai


os.environ["GEMINI_API_KEY"] = "AIzaSyA38G_UIGzKQbELOLdtXajXaBIRNrgz9I0"  

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
    
    def save_captions(self, captions, output_file='captions2.txt'):
        with open(output_file, 'w') as file:
            for caption in captions:
                file.write(caption + '\n')
        print(f"Captions have been saved to {output_file}")

    def run(self, directory, output_file='captions2.txt'):
        captions = self.process_images(directory)
        self.save_captions(captions, output_file)
        
