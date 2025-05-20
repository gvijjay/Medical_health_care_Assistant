# File: scrap_store.py

import os
import requests
import pickle
import time
import concurrent.futures
from typing import List, Union
from tqdm import tqdm
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from urls_list import links, pdf_urls

CONFIG = {
    "output_dir": "scraping_output",
    "batch_size": 50,
    "max_workers": 8,
    "request_delay": 0.5,
    "faiss_index_name": "combined_faiss_index",
    "processed_log": "processed_urls.log",
    "processed_pdfs_log": "processed_pdfs.log",
    "scraped_data_backup": "scraped_data.pkl",
    "embedding_model": "BAAI/bge-large-en-v1.5"
}

def setup_environment():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    for file in [CONFIG["processed_log"], CONFIG["processed_pdfs_log"], CONFIG["scraped_data_backup"]]:
        file_path = os.path.join(CONFIG["output_dir"], file)
        if not os.path.exists(file_path):
            open(file_path, 'w').close()

def get_processed_items(log_file: str) -> set:
    try:
        with open(os.path.join(CONFIG["output_dir"], log_file), 'r') as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()

def process_pdf(pdf_path: str) -> dict:
    try:
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"File path {pdf_path} is not valid.")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        full_text = "\n".join([page.page_content for page in pages])
        return {
            "source": pdf_path,
            "content": full_text,
            "type": "pdf",
            "timestamp": time.time()
        }
    except Exception as e:
        print(f"\nFailed to process {pdf_path}: {str(e)}")
        return {"source": pdf_path, "content": "", "error": str(e), "type": "pdf"}

def scrape_single_url(url: str) -> dict:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(["script", "style", "nav", "footer", "iframe", "header"]):
            element.decompose()
        text = soup.get_text(separator=' ', strip=True)
        time.sleep(CONFIG["request_delay"])
        return {
            "source": url,
            "content": text,
            "type": "url",
            "timestamp": time.time()
        }
    except Exception as e:
        print(f"\nFailed to scrape {url}: {str(e)}")
        return {"source": url, "content": "", "error": str(e), "type": "url"}

def process_batch(items: List[Union[str, dict]], is_pdf: bool = False) -> list:
    results = []
    log_file = CONFIG["processed_pdfs_log"] if is_pdf else CONFIG["processed_log"]
    processed_items = get_processed_items(log_file)
    items_to_process = [item for item in items if item not in processed_items]

    if not items_to_process:
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = {executor.submit(process_pdf if is_pdf else scrape_single_url, item): item for item in items_to_process}
        with tqdm(total=len(items_to_process), desc=f"Processing {'PDF' if is_pdf else 'URL'} batch") as pbar:
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                    if result and result.get("content"):
                        results.append(result)
                        with open(os.path.join(CONFIG["output_dir"], log_file), 'a') as f:
                            f.write(item + '\n')
                except Exception as e:
                    print(f"\nError processing {item}: {str(e)}")
                finally:
                    pbar.update(1)
    return results

def create_documents(scraped_data: list) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    documents = []
    for item in scraped_data:
        if not item["content"]:
            continue
        chunks = text_splitter.split_text(item["content"])
        for chunk in chunks:
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": item["source"],
                    "type": item["type"],
                    "timestamp": item["timestamp"]
                }
            ))
    return documents

def update_faiss_index(scraped_data: list, existing_index: FAISS = None) -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG["embedding_model"],
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    documents = create_documents(scraped_data)
    if existing_index:
        print("Updating existing FAISS index...")
        existing_index.add_documents(documents)
        return existing_index
    else:
        print("Creating new FAISS index...")
        return FAISS.from_documents(documents, embeddings)

def save_backup(data: list):
    backup_path = os.path.join(CONFIG["output_dir"], CONFIG["scraped_data_backup"])
    try:
        existing_data = []
        if os.path.exists(backup_path):
            with open(backup_path, 'rb') as f:
                existing_data = pickle.load(f)
        combined_data = existing_data + data
        with open(backup_path, 'wb') as f:
            pickle.dump(combined_data, f)
    except Exception as e:
        print(f"Error saving backup: {str(e)}")

def main(url_list: List[str], pdf_list: List[str] = None):
    setup_environment()

    index_path = os.path.join(CONFIG["output_dir"], CONFIG["faiss_index_name"])
    faiss_index = None
    if os.path.exists(index_path):
        try:
            embeddings = HuggingFaceEmbeddings(model_name=CONFIG["embedding_model"])
            faiss_index = FAISS.load_local(index_path, embeddings)
            print("Loaded existing FAISS index")
        except Exception as e:
            print(f"Error loading existing index: {str(e)}")
            faiss_index = None

    if url_list:
        print(f"\nProcessing {len(url_list)} URLs...")
        url_batches = [url_list[i:i + CONFIG["batch_size"]] for i in range(0, len(url_list), CONFIG["batch_size"])]
        for batch in url_batches:
            scraped_batch = process_batch(batch, is_pdf=False)
            if scraped_batch:
                faiss_index = update_faiss_index(scraped_batch, faiss_index)
                save_backup(scraped_batch)

    if pdf_list:
        print(f"\nProcessing {len(pdf_list)} PDFs...")
        pdf_batches = [pdf_list[i:i + CONFIG["batch_size"]] for i in range(0, len(pdf_list), CONFIG["batch_size"])]
        for batch in pdf_batches:
            scraped_batch = process_batch(batch, is_pdf=True)
            if scraped_batch:
                faiss_index = update_faiss_index(scraped_batch, faiss_index)
                save_backup(scraped_batch)

    if faiss_index:
        faiss_index.save_local(index_path)
        print(f"\nFinal combined FAISS index saved to {index_path}")

    if faiss_index:
        query = "What are the symptoms of abdominal aortic aneurysm?"
        print(f"\nRunning similarity search for: '{query}'")
        results = faiss_index.similarity_search(query, k=3)
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i} (Source: {doc.metadata['source']}, Type: {doc.metadata['type']}):")
            print(doc.page_content[:500] + "...")

if __name__ == "__main__":
    main(url_list=links, pdf_list=pdf_urls)
