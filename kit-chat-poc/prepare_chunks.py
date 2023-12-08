"""Prepares chunks of information for KIT CHat PoC from excel sheet links

  python prepare_chunks.py -m rcsplit -i input/

Requirements:
   To be filled later
"""
import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Callable, List

import pdfplumber
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich import print
from rich.progress import track

from datamodels import Chunk, KnowledgeFormat
from utils import count_tokens, get_filenames, load_names, to_json

DEFAULT_SEPARATORS = ["\n\n", "\n", "。", " ", ""]

IMPLEMENTED_METHODS = [
    "adhocsentencegrouping",
    "rcsplit",
    "automerging",
    "summary"
]


def process_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Select a strategy and chunk contents of PDF files accordingly"
    )
    parser.add_argument(
        "--method", "-m",
        dest="method",
        required=False,
        default="rcsplit",
        choices=IMPLEMENTED_METHODS + ["all"],
        type=str,
        help="Text chunking method",
    )
    parser.add_argument(
        "--input", "-i",
        dest="path_to_docs",
        required=False,
        type=str,
        default="input",
        help="Path to input documents (can be a folder or a JSON file)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        dest="path_to_chunkstores",
        required=False,
        type=str,
        default="interim",
        help="Path to output chunkstores",
    )
    return parser.parse_args()


def load_pdf_documents(path_to_docs: Path) -> List[KnowledgeFormat]:
    """Load PDF files from disk.

    Note that all target PDF files are loaded to memory at once (known as eager loading).

    Args:
        path_to_docs (str): directory containing PDF Files
        method (str): loader name

    Returns:
        List[PdfPage]: PDF pages
    """
    if path_to_docs.suffix.lower() == ".json":
        return load_documents_from_json(path_to_docs)

    filenames = get_filenames(path_to_docs.stem + "/pdfs/", "pdf", True)
    docs = []
    for filename in track(sequence=filenames, description="Loading PDF files..."):
        doc = []
        with pdfplumber.open(filename) as pdf:
            for page in pdf.pages:
                doc.append(
                    KnowledgeFormat(
                        content=page.extract_text(),
                        source_path=filename.as_posix(),
                        title=filename.stem[:-3],
                        page_num=page.page_number,
                        lang=filename.stem.split("_")[-1]
                    )
                )
        docs.extend(doc)
    return docs


def load_html_documents(path_to_docs: Path, html_lang: str = "/htmls_en") -> List[KnowledgeFormat]:
    """Load PDF files from disk.

    Note that all target HTML files are loaded to memory at once (known as eager loading).

    Args:
        path_to_docs (str): directory containing HTML Files
        method (str): loader name

    Returns:
        List[PdfPage]: PDF pages
    """
    if path_to_docs.suffix.lower() == ".json":
        return load_documents_from_json(path_to_docs)

    filenames = get_filenames(path_to_docs.stem + html_lang, "html", True)
    docs = []
    html_count = 1
    for filename in track(sequence=filenames, description="Loading HTML Sharepoint files..."):
        doc = []
        loader = UnstructuredHTMLLoader(filename)
        data_items = loader.load()
        for data in data_items:
            doc.append(
                KnowledgeFormat(
                    content=data.page_content,
                    source_path=filename.as_posix(),
                    title=filename.stem if html_lang[-2:] == "jp" else filename.stem[:-3],
                    page_num=html_count,
                    lang=html_lang.split("_")[-1],
                )
            )
            html_count = html_count + 1
        docs.extend(doc)
    return docs


def init_rc_text_splitter(
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    length_function: Callable = count_tokens,
    is_separator_regex: bool = False,
    keep_separator: bool = True,
    separators: List[str] = DEFAULT_SEPARATORS,
) -> List[dict]:
    """Initialize Recursive Char Text Splitter.

    Args:
        chunk_size (int): max number of tokens allowed in a single chunk
        chunk_overlap (int): overlap between adjacent chunks
        length_function (Callable): custom method used to count tokens
        is_separator_regex (bool): is separator regex
        keep_separator (bool): keep separator in splits
        separators (List[str]): list of separators
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        is_separator_regex=is_separator_regex,
        keep_separator=keep_separator,
        separators=separators,
    )


def chunk_with_rcsplit(documents: List[KnowledgeFormat]) -> List[Chunk]:
    """Chunk text by recursively looking at certain characters.

    Args:
        documents (List[PdfPage]): input documents to be chunked

    Returns:
        List[Chunk]: output chunks
    """
    splitter = init_rc_text_splitter(chunk_size=1024, chunk_overlap=256)
    chunkstore = []
    for doc in documents:
        texts = splitter.split_text(doc.content)
        parents = [
            Chunk(
                content=text,
                page_num=doc.page_num,
                source_path=doc.source_path,
                title=doc.title,
                lang=doc.lang
            )
            for text in texts
        ]
        chunkstore.extend(parents)
    return chunkstore


def is_input_dir_valid(path_to_docs: Path) -> bool:
    """Validate input directory."""
    if path_to_docs.exists():
        if path_to_docs.is_file() and path_to_docs.suffix.lower() == ".json":
            return True
        # print(path_to_docs.stem)
        # print(type(path_to_docs.stem))
        pdf_filenames = get_filenames(path_to_docs.stem + "/pdfs/", "pdf", True)
        # print(pdf_filenames)
        html_filenames = get_filenames(path_to_docs.stem + "/htmls_en/", "html", True)
        # print(html_filenames)
        if pdf_filenames and html_filenames:
            return True
    print(f"[ERROR] '{path_to_docs}' does not exist or does not contain any PDF files")
    return False


def is_input_dir_valid(path_to_docs: Path) -> bool:
    """Validate input directory."""
    if path_to_docs.exists():
        if path_to_docs.is_file() and path_to_docs.suffix.lower() == ".json":
            return True
        filenames = get_filenames(path_to_docs, recursively=True)
        if filenames:
            return True
    print(f"[ERROR] '{path_to_docs}' does not exist or does not contain any PDF files")
    return False


def get_chunks(
    doc_type: str, 
    documents: List[KnowledgeFormat], 
    path_to_chunkstores: Path,
    filternames: List[str],
    ) -> None:
    """Get chunks from input documents and save to disk."""

    # split with recursive method
    chunkstore = chunk_with_rcsplit(documents)

    # post process
    _ = postprocess_chunk(chunkstore, filternames)

    # save chunks to disk
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fname = path_to_chunkstores.joinpath(f"chunkstore_{doc_type}_{timestamp}.json")
    to_json(out_fname, chunkstore)
    print(f"[INFO] Chunks are stored in '{out_fname}'")


def load_documents_from_json(filename: Path) -> List[KnowledgeFormat]:
    """Load documents from a JSON file."""
    # Read file contents
    with open(filename, mode="r") as fin:
        docs = json.load(fin)

    # Create 'PdfPage' object for each record
    documents = []
    for doc in docs:
        documents.append(
            KnowledgeFormat(
                id=doc["id"],
                num_tokens=doc["num_tokens"],
                page_num=doc["page_num"],
                source_path=doc["source_path"],
                title=doc["title"],
                content=doc["content"]
            )
        )

    return documents


def display_doc_stats(
    loaded_data: List[KnowledgeFormat], 
    method: str = "rcsplit",
    source_type: str = "PDF"
) -> None:
    """
    Display chunk statistics and properties of each type

    Args:
        loaded_data (List[KnowledgeFormat]): List of loaded documents 
    """

    total_num_files = len(set(doc.source_path for doc in loaded_data))
    total_num_pages = len(loaded_data)
    num_tokens = [page.num_tokens for page in loaded_data]

    print(f"[INFO] Data Ingestion Statistics for {source_type} Files :......")
    print(f"[INFO] chunking_method: '{method}'")
    print(f"[INFO] total_num_files: {total_num_files}")
    print(f"[INFO] total_num_pages: {total_num_pages}")
    print(f"[INFO] min(num_tokens_per_page): {min(num_tokens)}")
    print(f"[INFO] mean(num_tokens_per_page): {int(statistics.mean(num_tokens))}")
    print(f"[INFO] median(num_tokens_per_page): {int(statistics.median(num_tokens))}")
    print(f"[INFO] max(num_tokens_per_page): {max(num_tokens)}")
    print("\n\n")


def postprocess_chunk(chunkstore: List[Chunk], filternames: List[str]) -> List[Chunk]:

    for chunk in chunkstore:
        for name in filternames:
            chunk.content = chunk.content.replace(name[:-1], "")

    # print(chunkstore[0])
    return chunkstore


def main():
    """Orchestrate chunking of texts in PDF files and HTML pages."""
    args = process_args()

    path_to_docs = Path(args.path_to_docs)
    path_to_chunkstores = Path(args.path_to_chunkstores)  # output folder

    if is_input_dir_valid(path_to_docs):
        path_to_chunkstores.mkdir(parents=True, exist_ok=True)
        pdf_documents = load_pdf_documents(path_to_docs)
        print(len(pdf_documents))
        html_en_documents = load_html_documents(path_to_docs, "/htmls_en")
        print(len(html_en_documents))
        html_jp_documents = load_html_documents(path_to_docs, "/htmls_jp")
        print(len(html_jp_documents))

        # Print chunk statistics
        display_doc_stats(pdf_documents)
        display_doc_stats(html_en_documents, source_type="HTML Sharepoint English")
        display_doc_stats(html_jp_documents, source_type="HTML Sharepoint Japanese")
        
        
        # Load emp names
        kit_names = load_names("input/pii_list.txt")

        # Create chunks for all knowledge types
        get_chunks("pdf", pdf_documents, path_to_chunkstores, kit_names)
        get_chunks("html_en", html_en_documents, path_to_chunkstores, kit_names)
        get_chunks("html_jp", html_jp_documents, path_to_chunkstores, kit_names)


if __name__ == "__main__":
    start_time = datetime.now() 
    main()
    time_elapsed = datetime.now() - start_time 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
