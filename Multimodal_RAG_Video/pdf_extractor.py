import fitz
import os
import glob

class PDFExtractor:
    def __init__(self):
        self.output_dir = "mixed_data"
        self.output_file = "captions2.txt"
    
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
                image_filename = os.path.join(self.output_dir, f"frame_{image_index + 1}.{image_ext}")

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
        with open(self.output_file, "w") as f:
            f.write(text)
        return text

    def process_pdf(self, pdf_path):
        self.extract_images_from_pdf(pdf_path)
        extracted_text = self.extract_text_and_tables_from_pdf(pdf_path)
        image_paths = glob.glob(os.path.join(self.output_dir, '*.png'))
        
        return extracted_text, image_paths