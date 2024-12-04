import os
import json
import logging
import time
import google.generativeai as genai
from tqdm import tqdm


class ResearchPaperProcessor:
    def __init__(self, api_key, pdf_folder="pdfs", response_folder="responses", json_folder="json"):
        """
        Initializes processor with API key and folder paths.
        Configures generative AI model with prompts and settings.
        Ensures required folders exist for processing data.
        """
        self.api_key = api_key
        self.pdf_folder = pdf_folder
        self.response_folder = response_folder
        self.json_folder = json_folder
        os.makedirs(self.response_folder, exist_ok=True)
        os.makedirs(self.json_folder, exist_ok=True)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Configure Generative AI
        genai.configure(api_key=self.api_key)
        self.gemini_model = genai.GenerativeModel(
            "models/gemini-1.5-flash",
            system_instruction=self._get_system_prompt(),
            generation_config=self._get_generation_config(),
        )

    def _get_system_prompt(self):
        SYSTEM_PROMPT = """You are a highly accurate research paper document extraction model. Your task is to extract content from PDFs while maintaining the original structure and formatting. When you encounter a plot or image, do not include any URLs or HTML tags. Instead, extract both the original title and description from the PDF, and generate a detailed description. Ensure the content is unaltered and free of unnecessary code."""
        return SYSTEM_PROMPT

    def _get_user_prompt(self):
        USER_PROMPT = """
        Extract the following information from the provided PDF and structure it as a JSON object with the following keys: 'title', 'authors', 'abstract', 'sections', and 'images'.  

        - The 'sections' key should contain a dictionary where the keys are the section headings (without any numbers) and the values are the corresponding text content.  
        - The 'references' key should be included in the 'sections' key and contain a list of objects where each object represents a referenced paper. Each reference object should include the title, authors, and a citation formatted as: "Authors. (Year). Title of the paper. Journal Name (if available), Volume(Issue), Pages."  
        - Include the images in the 'images' key, which should be structured as a dictionary. Each image or table should have an 'image_desc' (description) and 'image_location' (location within the document, e.g., Figure 1, Table 2).  
        - Do not include any URLs or HTML tags. Ensure the content is unaltered and free of unnecessary code. Provide only the JSON output without any additional text or explanations.  

        The final output JSON should strictly follow the form below:

        {
            "data": {
            "title": "Example Title",
            "authors": "Author 1, Author 2, Author 3",
            "abstract": "Abstract text goes here.",
            "sections": {
                "Section 1 Title": "Section 1 content...",
                "Section 2 Title": "Section 2 content...",
                "References": [
                {
                    "title": "Paper Title",
                    "authors": "Author A, Author B",
                    "citation": "Author A, Author B. (Year). Paper Title. Journal Name, Volume(Issue), Pages."
                }
                ]
            },
            "images": {
                "image_1": {
                "image_desc": "Description of image 1.",
                "image_location": "Figure 1"
                },
                "image_2": {
                "image_desc": "Description of image 2.",
                "image_location": "Figure 2"
                }
            }
            }
        }
        }
        """
        return USER_PROMPT

    def _get_generation_config(self):
        return {
            "temperature": 0.5,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }

    def upload_pdfs(self):
        """
        Uploads PDF files from the specified folder to the AI model.
        Filters files with `.pdf` extension only.
        Returns a list of uploaded file objects.
        """
        pdf_files = [os.path.join(self.pdf_folder, f) for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
        return [genai.upload_file(pdf, mime_type="application/pdf") for pdf in pdf_files]

    def generate_responses(self, files):
        """
        Generates AI responses for each uploaded PDF file.
        Saves the responses as text files in the response folder.
        Handles errors gracefully and ensures a delay between requests.
        """
        for pdf in tqdm(files, desc="Processing PDFs", unit="file"):
            base_name = pdf.display_name.replace('.pdf', '')
            try:
                response = self.gemini_model.generate_content([pdf, self._get_user_prompt()])
                response_text = response.text
                with open(os.path.join(self.response_folder, f"{base_name}.txt"), 'w') as f:
                    f.write(response_text)
                time.sleep(15)
            except Exception as e:
                logging.error(f"Error generating response for {base_name}: {e}")

    def convert_responses_to_json(self):
        """
        Converts AI text responses to JSON format for easy parsing.
        Saves each JSON object in the specified folder.
        Handles decoding errors and logs issues for debugging.
        """
        for text_file in os.listdir(self.response_folder):
            if text_file.endswith('.txt'):
                base_name = text_file.replace('.txt', '')
                with open(os.path.join(self.response_folder, text_file), 'r') as f:
                    try:
                        json_data = json.loads(f.read())
                        json_file_path = os.path.join(self.json_folder, f"{base_name}.json")
                        with open(json_file_path, 'w') as json_file:
                            json.dump(json_data, json_file, indent=4)
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON decoding error in {base_name}: {e}")

    def combine_json_files(self, output_file):
        """
        Combines all JSON files into a single consolidated file.
        Reads each JSON, extracts data, and merges it into one object.
        Saves the final combined data to the specified output file.
        """
        combined_data = {}
        for json_file in os.listdir(self.json_folder):
            if json_file.endswith('.json'):
                with open(os.path.join(self.json_folder, json_file), 'r') as f:
                    data = json.load(f)['data']
                    combined_data[json_file.replace('.json', '')] = data
        with open(output_file, 'w') as f:
            json.dump(combined_data, f, indent=4)

    def run(self):
        """
        Executes the entire processing workflow for PDFs.
        Handles uploading, response generation, JSON conversion, and merging.
        Produces a final consolidated JSON file for all processed papers.
        """
        files = self.upload_pdfs()
        self.generate_responses(files)
        self.convert_responses_to_json()
        self.combine_json_files("combined_data.json")
