import fitz
import os
import glob
from pathlib import Path
import shutil

class PDFExtractor:
    def __init__(self, output_dir="mixed_data", output_file="captions2.txt"):
        self.output_dir = output_dir
        self.output_file = output_file
    
    def extract_images_from_pdf(self, pdf_path):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        pdf_document = fitz.open(pdf_path)
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            image_list = page.get_images(full=True)
            for image_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = os.path.join(self.output_dir, f"frame_{image_index + 1}_{os.path.basename(pdf_path)}.{image_ext}")

                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)

        pdf_document.close()
        print(f"Images successfully extracted to {self.output_dir}")

    def extract_text_and_tables_from_pdf(self, pdf_path):
        pdf_document = fitz.open(pdf_path)
        text = ""
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            text += page.get_text()

        pdf_document.close()
        '''with open(self.output_dir + self.output_file, "w") as f:
            f.write(text)'''
        return text

    def append_text_to_file(self, text):
        with open(os.path.join(self.output_dir, self.output_file), "a") as f:
            f.write(text + "\n\n")
        print(f"Text successfully appended to {self.output_file}")

    def process_pdf(self, pdf_path):
        self.extract_images_from_pdf(pdf_path)
        extracted_text = self.extract_text_and_tables_from_pdf(pdf_path)
        self.append_text_to_file(extracted_text)
        image_paths = glob.glob(os.path.join(self.output_dir, '*.png'))
        
        return extracted_text, image_paths

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

def process_multi_pdf(space_key):
    #pdf_file = 'data/4249142809_2af1437517504cf3b9a748973e61cce7-290524-1446-554.pdf'
    pdf_files = os.listdir(space_key)
    print("list of all files", pdf_files)
    text_data ="NA"
    delete_directory("mixed_data")  
    output_folder = "./mixed_data/"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    pdf_extractor = PDFExtractor(output_dir=output_folder)
    for pdf_file in pdf_files:
        print("***************", pdf_file)
        extracted_text, image_paths = pdf_extractor.process_pdf(os.path.join(space_key,pdf_file))
        
        print(f"\nPDF processed successfully: {pdf_file}")
        print("Extracted Images:")
        for image_path in image_paths:
            print(image_path)
    '''with open(output_folder + "output_text.txt", "w") as file:
            file.write(text_data)
    pdf_extractor = PDFExtractor()
    extracted_text, image_paths = pdf_extractor.process_pdf(pdf_file)
    #print("Extracted Text:")
    #print(extracted_text)
    print("\nExtracted Images:")
    for image_path in image_paths:
        print(image_path)'''
    return "All PDFs processed successfully!!!"

#process_multi_pdf('3DVE')