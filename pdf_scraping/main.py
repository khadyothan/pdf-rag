from arxiv_downloader import ArxivPaperDownloader
from research_processor import ResearchPaperProcessor

# Step 1: Download and filter papers
downloader = ArxivPaperDownloader(query="cat:cs.DC", date_range="20230101 TO 20240101")
papers = downloader.fetch_papers()
metadata_df = downloader.process_paper_metadata(papers)
filtered_df = downloader.download_and_filter_pdfs(metadata_df)
filtered_df.to_csv("filtered_pdfs.csv", index=False)

# Step 2: Process filtered PDFs
processor = ResearchPaperProcessor(api_key="AIzaSyDO9vVOooSInMZ7fI4nT9GeUzfA-mMbRFQ")
processor.run()
