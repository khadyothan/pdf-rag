import os
import logging
import requests
from tqdm import tqdm
from PyPDF2 import PdfReader
import pandas as pd
import arxiv
from io import BytesIO


class ArxivPaperDownloader:
    def __init__(self, query, date_range, max_results=200, max_pages=15, output_folder="pdfs"):
        """
        Initializes downloader with query, date range, and result limits.
        Configures output folder for saving downloaded PDFs.
        Ensures folder exists and logging is set up.
        """
        self.query = query
        self.date_range = date_range
        self.max_results = max_results
        self.max_pages = max_pages
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def fetch_papers(self):
        """
        Fetches research papers from arXiv using the specified query.
        Retrieves metadata such as title, authors, and PDF URLs.
        Returns a list of results from arXiv.
        """
        client = arxiv.Client()
        search = arxiv.Search(
            query=f"{self.query} AND submittedDate:[{self.date_range}]",
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        return list(client.results(search))

    def process_paper_metadata(self, papers):
        """
        Converts arXiv results into a structured DataFrame.
        Extracts metadata including title, authors, and abstract.
        Returns a DataFrame for further processing.
        """
        df = pd.DataFrame([
            {
                'id': paper.get_short_id(),
                'pdf_url': paper.pdf_url,
                'title': paper.title,
                'authors': ", ".join([author.name for author in paper.authors]),
                'abstract': paper.summary,
            }
            for paper in papers
        ])
        return df

    def download_and_filter_pdfs(self, df):
        """
        Downloads PDFs, filters based on page count, and trims unprocessed rows.
        Ensures valid PDFs are saved to the specified folder.
        Returns a DataFrame of filtered PDFs with page count <= max_pages.
        """
        valid_pdf_count = 0
        pdf_page_counts = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading PDFs"):
            url, file_name = row['pdf_url'], f"{row['id']}.pdf"
            try:
                response = requests.get(url)
                response.raise_for_status()
                pdf_reader = PdfReader(BytesIO(response.content))
                page_count = len(pdf_reader.pages)

                if page_count <= self.max_pages:
                    file_path = os.path.join(self.output_folder, file_name)
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    valid_pdf_count += 1

                pdf_page_counts.append(page_count)
                if valid_pdf_count >= 100:
                    break
            except Exception as e:
                logging.warning(f"Failed to process {url}: {e}")
                pdf_page_counts.append(0)

        # Ensure the lengths match by trimming the DataFrame
        processed_rows = len(pdf_page_counts)
        df = df.iloc[:processed_rows]
        df['pdf_page_count'] = pdf_page_counts

        # Filter and return the trimmed DataFrame
        filtered_df = df[df['pdf_page_count'] <= self.max_pages]
        logging.info(f"Filtered {len(filtered_df)} PDFs with {self.max_pages} or fewer pages.")
        return filtered_df
