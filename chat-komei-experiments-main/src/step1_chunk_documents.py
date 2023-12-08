"""Implements different text splitting methods for RAG systems.

Currently, the following methods are implemented.

  1. adhocsentencegrouping
  2. rcsplit
  3. automerging
  4. summary
  5. all (executes all methods one by one)

Usage:

  python src/step1_chunk_documents.py -m adhocsentencegrouping -i input/prod/
  python src/step1_chunk_documents.py -m rcsplit -i input/prod/
  python src/step1_chunk_documents.py -m automerging -i input/prod/
  python src/step1_chunk_documents.py -m summary -i input/prod/
  python src/step1_chunk_documents.py -m all -i input/prod/

  python src/step1_chunk_documents.py -m adhocsentencegrouping -i prod.json
  python src/step1_chunk_documents.py -m rcsplit -i prod.json
  python src/step1_chunk_documents.py -m automerging -i prod.json
  python src/step1_chunk_documents.py -m summary -i prod.json
  python src/step1_chunk_documents.py -m all -i prod.json

Requirements:

  1. The following environment variables MUST BE set in '.env' file for '-m summary'.
     For other methods, Open AI service would not be needed.

    1. OPENAI_API_KEY
    2. OPENAI_API_ENDPOINT

  2. Documents (PDF files) MUST BE placed in a single directory or their parsed content
     MUST be stored in a single JSON file. The schema of the JSON file MUST BE as follows.

     ```json
     [
       {
         "id": str,
         "num_tokens": int,
         "page_num": str,
         "source_path": str,
         "title": str,
         "content": str
       },
       {
         ...
       },
       ...
     ]
     ```
"""
import argparse
import json
import os
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import openai
import pdfplumber
import tiktoken
from dotenv import find_dotenv, load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from rich import print
from rich.progress import track

try:
    from src.constants import CONTEXT_LENGTHS
    from src.datamodels import Chunk, PdfPage
    from src.utils import count_tokens, get_filenames, to_json
except ImportError:
    from constants import CONTEXT_LENGTHS
    from datamodels import Chunk, PdfPage
    from utils import count_tokens, get_filenames, to_json

load_dotenv(find_dotenv())

openai.api_type = "azure"
openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_base = os.environ.get("OPENAI_API_ENDPOINT")
openai.api_version = "2023-07-01-preview"
ENCODER = tiktoken.encoding_for_model("gpt-3.5-turbo")

SYSTEM_PROMPT = """下記の文章の要点を要約してください"""
NUM_TOKENS_SYSTEM_PROMPT = len(ENCODER.encode(SYSTEM_PROMPT))
SLEEP_BEFORE_TALKING_TO_LLM = 0.1

PDF_SENTENCE_DELIMITERS = "。|\n"
PDF_SENTENCES_PER_SECTION = 10
PDF_CONTENT_BLOCK_SEPARATOR = " "
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


def load_documents(path_to_docs: Path) -> List[PdfPage]:
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

    filenames = get_filenames(path_to_docs)
    docs = []
    for filename in track(sequence=filenames, description="Loading PDF files..."):
        doc = []
        with pdfplumber.open(filename) as pdf:
            for page in pdf.pages:
                doc.append(
                    PdfPage(
                        content=page.extract_text(),
                        source_path=filename.as_posix(),
                        title=filename.stem,
                        page_num=page.page_number,
                    )
                )
        docs.extend(doc)
    return docs


def summarize(
    text: str,
    deployment: str = "gpt-35-turbo",
    max_num_retries: int = 5,
    num_tokens_buffer: int = 32,
) -> Optional[str]:
    """Summarize text using LLM.

    To select the parameters, you may refer to the following resources:

    - https://platform.openai.com/docs/api-reference/chat/create
    - https://docs.cohere.com/docs/temperature
    - https://docs.cohere.com/docs/controlling-generation-with-top-k-top-p

    Args:
        text (str): input text

    Returns:
        str: summary
    """
    time.sleep(SLEEP_BEFORE_TALKING_TO_LLM)
    context_length = CONTEXT_LENGTHS[deployment]
    max_num_tokens = int(
        (context_length - NUM_TOKENS_SYSTEM_PROMPT - num_tokens_buffer) / 2
    )
    text = ENCODER.decode(ENCODER.encode(text)[:(max_num_tokens)])
    message = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}"},
        {"role": "user", "content": f"{text}"},
    ]
    retry = 0
    while retry < max_num_retries:
        try:
            response = openai.ChatCompletion.create(
                engine=deployment,
                messages=message,
                temperature=0.0,
                max_tokens=count_tokens(text),
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as err:
            timestamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
            print(f"[WARNING] [{retry + 1}] [{timestamp}] {err}. Retrying...")
            time.sleep(0.5)
        retry += 1
    return None


def init_rc_text_splitter(
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    length_function: Callable = count_tokens,
    is_separator_regex: bool = False,
    keep_separator: bool = True,
    separators: List[str] = DEFAULT_SEPARATORS,
) -> RecursiveCharacterTextSplitter:
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


def chunk_with_rcsplit(documents: List[PdfPage]) -> List[Chunk]:
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
            )
            for text in texts
        ]
        chunkstore.extend(parents)
    return chunkstore


def chunk_with_summarization(documents: List[PdfPage]) -> List[Chunk]:
    """Chunk the texts using DocSummary strategy.

    Args:
        documents (List[PdfPage]): input documents to be chunked

    Returns:
        List[Chunk]: output chunks
    """
    splitter = init_rc_text_splitter(chunk_size=1024, chunk_overlap=256)
    chunkstore = []

    for doc in track(sequence=documents, description="Processing page..."):
        summary = summarize(doc.content)
        if summary is None:
            print(f"[WARNING] Issue in doc_id: '{doc.id}'. Setting summary to empty string")
            summary = ""

        texts = splitter.split_text(doc.content)
        parent = Chunk(
            content=summary,
            page_num=doc.page_num,
            source_path=doc.source_path,
            title=doc.title,
            modified_from_source=True,
        )
        children = [
            Chunk(
                parent_id=parent.id,
                content=text,
                page_num=doc.page_num,
                source_path=doc.source_path,
                title=doc.title,
            )
            for text in texts
        ]
        chunkstore.extend([parent] + children)

    return chunkstore


def chunk_with_automerging(documents: List[PdfPage]) -> List[Chunk]:
    """Chunk document text using Auto Merging strategy.

    Args:
        documents (List[PdfPage]): input documents to be chunked

    Returns:
        List[Chunk]: output chunks
    """
    large_splitter = TokenTextSplitter(chunk_size=4096, chunk_overlap=1024)
    medium_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=256)

    chunkstore = []
    for doc in documents:
        large_chunks = large_splitter.split_text(doc.content)
        for large_text in large_chunks:
            medium_chunks = medium_splitter.split_text(large_text)
            parent = Chunk(
                content=large_text,
                page_num=doc.page_num,
                source_path=doc.source_path,
                title=doc.title,
            )
            children = [
                Chunk(
                    parent_id=parent.id,
                    content=medium_text,
                    page_num=doc.page_num,
                    source_path=doc.source_path,
                    title=doc.title,
                )
                for medium_text in medium_chunks
            ]
            chunkstore.extend([parent] + children)

    return chunkstore


def chunk_with_adhoc_sentence_grouping(documents: List[PdfPage]) -> List[Chunk]:
    """Create chunks by grouping sentences.

    As sentences are extracted, or predicted, using an ad-hoc rule,
    their boundaries may be inaccurate.

    Args:
        documents (List[PdfPage]): input documents to be chunked

    Returns:
        List[Chunk]: output chunks

    Steps:

    1. Split PDF page text into sentences (with an ad-hoc rule)
    2. Concatenate each PDF_SENTENCES_PER_SECTION number of sentences into a single block
        (by adding whitespace between each sentence within the block)
    3. Move on to the next page
    """
    chunkstore = []
    for doc in documents:
        sentences = re.split(PDF_SENTENCE_DELIMITERS, doc.content)
        block = []
        block_num = 1
        for block_idx, sentence in enumerate(sentences):
            block.append(sentence)
            if (block_idx + 1) % PDF_SENTENCES_PER_SECTION == 0:
                chunkstore.append(
                    Chunk(
                        content=PDF_CONTENT_BLOCK_SEPARATOR.join(block),
                        page_num=doc.page_num,
                        source_path=doc.source_path,
                        title=doc.title,
                    )
                )
                block = []
                block_num += 1
    return chunkstore


def is_input_dir_valid(path_to_docs: Path) -> bool:
    """Validate input directory."""
    if path_to_docs.exists():
        if path_to_docs.is_file() and path_to_docs.suffix.lower() == ".json":
            return True
        filenames = get_filenames(path_to_docs)
        if filenames:
            return True
    print(f"[ERROR] '{path_to_docs}' does not exist or does not contain any PDF files")
    return False


def get_chunks(method: str, documents: List[PdfPage], path_to_chunkstores: Path) -> None:
    """Get chunks from input documents and save to disk."""
    if method == "rcsplit":
        chunkstore = chunk_with_rcsplit(documents)

    elif method == "summary":
        chunkstore = chunk_with_summarization(documents)

    elif method == "automerging":
        chunkstore = chunk_with_automerging(documents)

    elif method == "adhocsentencegrouping":
        chunkstore = chunk_with_adhoc_sentence_grouping(documents)

    num_tokens = [chunk.num_tokens for chunk in chunkstore if chunk.parent_id == "0"]
    print(f"[INFO] num_parent_chunks: {len(num_tokens)}")
    print(f"[INFO] min(num_tokens_per_parent_chunk): {min(num_tokens)}")
    print(f"[INFO] mean(num_tokens_per_parent_chunk): {int(statistics.mean(num_tokens))}")
    print(f"[INFO] median(num_tokens_per_parent_chunk): {int(statistics.median(num_tokens))}")
    print(f"[INFO] max(num_tokens_per_parent_chunk): {max(num_tokens)}")

    num_tokens = [chunk.num_tokens for chunk in chunkstore if chunk.parent_id != "0"]
    if num_tokens:
        print(f"[INFO] num_child_chunks: {len(num_tokens)}")
        print(f"[INFO] min(num_tokens_per_child_chunk): {min(num_tokens)}")
        print(f"[INFO] mean(num_tokens_per_child_chunk): {int(statistics.mean(num_tokens))}")
        print(f"[INFO] median(num_tokens_per_child_chunk): {int(statistics.median(num_tokens))}")
        print(f"[INFO] max(num_tokens_per_child_chunk): {max(num_tokens)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fname = path_to_chunkstores.joinpath(f"{method}_chunkstore_{timestamp}.json")
    to_json(out_fname, chunkstore)
    print(f"[INFO] Chunks are stored in '{out_fname}'")


def load_documents_from_json(filename: Path) -> List[PdfPage]:
    """Load documents from a JSON file."""
    # Read file contents
    with open(filename, mode="r") as fin:
        docs = json.load(fin)

    # Create 'PdfPage' object for each record
    documents = []
    for doc in docs:
        documents.append(
            PdfPage(
                id=doc["id"],
                num_tokens=doc["num_tokens"],
                page_num=doc["page_num"],
                source_path=doc["source_path"],
                title=doc["title"],
                content=doc["content"]
            )
        )

    return documents


def main():
    """Execute text splitting methods conditionally."""
    args = process_args()

    chunking_method = args.method
    path_to_docs = Path(args.path_to_docs)
    path_to_chunkstores = Path(args.path_to_chunkstores)  # output folder

    if is_input_dir_valid(path_to_docs):
        path_to_chunkstores.mkdir(parents=True, exist_ok=True)
        documents = load_documents(path_to_docs)
        total_num_files = len(set(doc.source_path for doc in documents))
        total_num_pages = len(documents)
        num_tokens = [page.num_tokens for page in documents]

        print(f"[INFO] chunking_method: '{chunking_method}'")
        print(f"[INFO] total_num_files: {total_num_files}")
        print(f"[INFO] total_num_pages: {total_num_pages}")
        print(f"[INFO] min(num_tokens_per_page): {min(num_tokens)}")
        print(f"[INFO] mean(num_tokens_per_page): {int(statistics.mean(num_tokens))}")
        print(f"[INFO] median(num_tokens_per_page): {int(statistics.median(num_tokens))}")
        print(f"[INFO] max(num_tokens_per_page): {max(num_tokens)}")

        if chunking_method == "all":
            for method in IMPLEMENTED_METHODS:
                get_chunks(method, documents, path_to_chunkstores)
        else:
            get_chunks(chunking_method, documents, path_to_chunkstores)


if __name__ == "__main__":
    main()
