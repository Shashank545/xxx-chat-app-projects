"""Run Retrieval Augmented Generation (RAG) pipeline.

Usage:

  python src/step3_run_rag_pipeline.py -i input/questions.csv

Requirements:

  1. The following environment variables MUST BE set in '.env' file.

    1. OPENAI_API_KEY
    2. OPENAI_API_ENDPOINT
    3. SEARCH_ENDPOINT
    4. SEARCH_API_KEY

  2. Azure Cognitive Search (ACS) indexes whose names are specified in 'INDEX_NAMES'
     global variable MUST BE constructed beforehand.

  3. Semantic search feature of the ACS instance MUST BE enabled.
"""
import argparse
import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import openai
import tiktoken
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType, Vector
from dotenv import find_dotenv, load_dotenv
from rich import print
from rich.progress import track

try:
    from src.constants import (
        CONTEXT_LENGTHS,
        SYSTEM_PROMPT_GENERATE_ANSWER,
        USER_PROMPT_CONFIRMING_ANSWER,
        USER_PROMPT_GENERATE_ANSWER,
        SearchOption
    )
    from src.utils import generate_embeddings, nonewlines, to_csv
except Exception:
    from constants import (
        CONTEXT_LENGTHS,
        SYSTEM_PROMPT_GENERATE_ANSWER,
        USER_PROMPT_CONFIRMING_ANSWER,
        USER_PROMPT_GENERATE_ANSWER,
        SearchOption
    )
    from utils import generate_embeddings, nonewlines, to_csv


load_dotenv(find_dotenv())

# Set up Open AI
openai.api_type = "azure"
openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_base = os.environ.get("OPENAI_API_ENDPOINT")
openai.api_version = "2023-07-01-preview"
AZURE_OPENAI_GPT_DEPLOYMENT_NAME = "gpt-35-turbo"
ENCODER = tiktoken.encoding_for_model("text-embedding-ada-002")

# Set up clients for Azure Cognitive Search
# INDEX_NAMES = ["rcsplit", "summary", "automerging"]
# INDEX_NAMES = ["rcsplit-chunkstore"]
# INDEX_NAMES = ["adhocsentencegrouping", "automerging", "rcsplit"]
INDEX_NAMES = ["summary"]
SEARCH_CLIENTS = {
    index_name: SearchClient(
        endpoint=os.environ.get("SEARCH_ENDPOINT"),
        index_name=index_name,
        credential=AzureKeyCredential(os.environ.get("SEARCH_API_KEY")),
        api_version="2023-07-01-Preview",
    )
    for index_name in INDEX_NAMES
}
SLEEP_BEFORE_TALKING_TO_LLM = 1


def process_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the RAG pipeline end to end"
    )
    parser.add_argument(
        "--input", "-i",
        dest="path_to_questions",
        required=False,
        type=str,
        default="input/questions.csv",
        help="Path to questions",
    )
    parser.add_argument(
        "--output-dir", "-o",
        dest="path_to_answers",
        required=False,
        type=str,
        default="output",
        help="Path to generated answers",
    )
    return parser.parse_args()


def retrieve(
    *,
    question: str,
    index_name: str,
    top: int = 3,
    search_option: int = SearchOption.VectorSemantic,
    search_filter: str = None,
    verbose: bool = False,
) -> list:
    """Retreve documents from ACS."""
    # Construct the payload for ACS
    payload = {"top": top, "filter": search_filter}
    if search_option in {SearchOption.Semantic, SearchOption.VectorSemantic}:
        payload = {
            **payload,
            "query_type": QueryType.SEMANTIC,
            "query_language": "ja-jp",
            "semantic_configuration_name": "default",
            "query_caption": None,
        }
    if search_option in {SearchOption.Vector, SearchOption.VectorBM25, SearchOption.VectorSemantic}:
        payload["vectors"] = [
            Vector(
                value=generate_embeddings(question),
                k=top,
                fields="content_embedding",
            )
        ]
    if search_option == SearchOption.Vector:
        payload["search_text"] = None
        payload["search_fields"] = []
    else:
        payload["search_text"] = question
        payload["search_fields"] = ["content"]

    if verbose:
        print(f"[DEBUG] {search_option=} ({SearchOption(search_option).name})")
        print(f"[DEBUG] {payload=}")
        print(f"[DEBUG] search_index={index_name}")

    # Retrieve relevant documents from ACS
    search_results = SEARCH_CLIENTS[index_name].search(**payload)

    # Parse search results
    data_points = []
    contents = []
    for doc in search_results:
        data_points.append(
            {
                "score": doc["@search.score"],
                "id": doc["id"],
                "parent_id": doc["parent_id"],
                "title": doc.get("title", ""),
                "content": doc.get("content", ""),
                "source_path": doc.get("source_path", ""),
                "page_num": doc.get("page_num", ""),
                "num_tokens": doc.get("num_tokens", ""),
                "modified_from_source": doc.get("modified_from_source", ""),
            }
        )
        contents.append("- " + nonewlines(doc["content"]))

    out_text = "\n".join(contents)

    return data_points, out_text


def talk_to_llm(
    message: List[dict],
    temperature: int = 0.,
    deployment: str = "gpt-4",
    max_num_retries: int = 5,
    sleep_time: float = 0.7,
    max_output_tokens: int = 1024,
    num_tokens_buffer: int = 8,
) -> Optional[str]:
    """Talk to LLM to answer user question.

    Note that

    1. LLM context windows count the input and output combined.
    2. As API returns the following message, we introduced 'num_tokens_buffer',
       a variable to leave a buffer in token limits
       > This model's maximum context length is 4096 tokens. However, you requested 4096
       > tokens (2060 in the messages, 2036 in the completion). Please reduce the length
       > of the messages or completion..
    """
    time.sleep(sleep_time)
    context_length = CONTEXT_LENGTHS[deployment]
    num_tokens = sum([len(ENCODER.encode(m["content"])) for m in message])
    num_tokens_excess = - (context_length - num_tokens + max_output_tokens + num_tokens_buffer)
    if num_tokens_excess > 0:
        text = message[-1]["content"]
        text = ENCODER.decode(ENCODER.encode(text)[:-num_tokens_excess])
    retry = 0
    while retry < max_num_retries:
        try:
            completion = openai.ChatCompletion.create(
                engine=deployment,
                messages=message,
                temperature=temperature,
                max_tokens=max_output_tokens,
                n=1,
            )
            return completion.choices[0].message.content
        except Exception as err:
            timestamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
            print(f"[WARNING] [{retry + 1}] [{timestamp}] {err}. Retrying...")
            time.sleep(0.5)
        retry += 1
    return None


def clean_text(text: str) -> str:
    """Clean up input text."""
    return re.sub(r"^「(.*)」$", r"\1", text)


def retrieve_simple(index_name: str, search_filter: str):
    """Retrieve IDs of target chunks."""
    payload = {
        "filter": search_filter,
        "query_type": QueryType.SIMPLE,
        "query_language": "ja-jp",
        "search_text": "*",
    }
    results = list(SEARCH_CLIENTS[index_name].search(**payload))

    family = {}
    for doc in results:
        if doc["parent_id"] not in family:
            family[doc["parent_id"]] = []
        family[doc["parent_id"]].append(doc["id"])

    return results, family


def generate_answer(question: str, index_name: str, verbose: bool = False) -> Tuple[str]:
    """Execute prompting strategy."""
    # Step 1: Retrieve relevant chunks
    if index_name == "automerging":
        # First look at the children
        search_filter = "parent_id ne '0'"
        top = 3
    elif index_name == "summary":
        # First look at the parents
        search_filter = "parent_id eq '0'"
        top = 64
    else:
        search_filter = None
        top = 3

    retrieved_docs, retrieved = retrieve(
        question=question,
        index_name=index_name,
        search_filter=search_filter,
        top=top,
    )

    if verbose:
        print(f"[DEBUG] [1] doc_ids: {'; '.join([doc['id'] for doc in retrieved_docs])}")

    # Step 2: Replace parent and children chunks if necessary
    if index_name == "summary":
        if any([doc["parent_id"] != "0" for doc in retrieved_docs]):
            print("[ERROR] Children exist in search results")

        ids = ",".join([doc["id"] for doc in retrieved_docs])
        search_filter = f"search.in(parent_id, '{ids}', ',')"
        top = 3

        retrieved_docs, retrieved = retrieve(
            question=question,
            index_name=index_name,
            search_filter=search_filter,
            top=top,
        )

    elif index_name == "automerging":
        if any([doc["parent_id"] == "0" for doc in retrieved_docs]):
            print("[ERROR] Parents exist in search results")

        parent_ids = ",".join([doc["parent_id"] for doc in retrieved_docs])
        search_filter = f"search.in(parent_id, '{parent_ids}', ',')"
        _, family_available = retrieve_simple(index_name, search_filter)

        family_current = {}
        for doc in retrieved_docs:
            if doc["parent_id"] not in family_current:
                family_current[doc["parent_id"]] = []
            family_current[doc["parent_id"]].append(doc["id"])

        target_docs = []
        for doc in retrieved_docs:
            if len(family_current[doc["parent_id"]]) / len(family_available[doc["parent_id"]]) >= 0.5:
                target_docs.append(doc["parent_id"])
            else:
                target_docs.append(doc["id"])

        doc_ids = ",".join([doc_id for doc_id in target_docs])
        search_filter = f"search.in(id, '{doc_ids}', ',')"
        retrieved_docs, _ = retrieve_simple(index_name, search_filter)

        if verbose:
            print(f"[DEBUG] [2] doc_ids: {'; '.join([doc['id'] for doc in retrieved_docs])}")

    # Step 3: Create reference information
    references = []
    for doc in retrieved_docs:
        _key = f"{doc['title']}_P{doc['page_num']}"
        if _key not in references:
            references.append(_key)
    references = "\n".join(references)

    # Step 4: Generate answer based on the question & retrieved contents
    message = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_GENERATE_ANSWER
        },
        {
            "role": "user",
            "content": USER_PROMPT_GENERATE_ANSWER.format(source=retrieved, question=question)
        },
    ]
    answer = talk_to_llm(message)

    # Step 5: Confirm that the generated answer is acceptable
    message = [
        {
            "role": "user",
            "content": USER_PROMPT_CONFIRMING_ANSWER.format(
                answer=answer, question=question
            )
        },
    ]
    confirmation_response = talk_to_llm(message)
    cleaned_confirmation_response = clean_text(confirmation_response)

    if cleaned_confirmation_response == "はい":
        return answer, references

    print(f"[DEBUG] {cleaned_confirmation_response=}")
    return confirmation_response, references


def load_questions(path_to_questions: Path):
    """Load test cases from CSV file.

    - Each line represents a single question
    - There MUST BE only one column (question) in the CSV file
    - There MUST NOT BE column header
    """
    if not path_to_questions.is_file():
        print(f"[ERROR] Make sure '{path_to_questions}' exists")
        exit(1)

    with open(path_to_questions, mode="r") as file:
        questions = [unicodedata.normalize("NFKC", line.rstrip()) for line in file]

    return questions


def reorganize(results: List[dict], questions: List[str]) -> List[dict]:
    """Reorganize results.

    The new format will be as follows.

    | ID | Question | Answer (method-1) | References (method-1) | ...
    |  1 | ........ | ................. | ..................... | ...
    """
    reorganized = []
    for result in results:
        row_num = result["ID"] - 1
        index_name = list(result.keys())[1]

        if row_num >= len(reorganized):
            reorganized.append(
                {
                    "ID": result["ID"],
                    "質問": questions[row_num],
                    f"回答 ({index_name})": result[index_name],
                    f"参照先 ({index_name})": result[f"{index_name}_references"],
                }
            )
        else:
            reorganized[row_num] = {
                **reorganized[row_num],
                f"回答 ({index_name})": result[index_name],
                f"参照先 ({index_name})": result[f"{index_name}_references"],
            }

    return reorganized


def main():
    """Orchestrate RAG pipeline."""
    args = process_args()

    path_to_questions = Path(args.path_to_questions)
    path_to_answers = Path(args.path_to_answers)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = path_to_answers.joinpath(f"answers_{timestamp}.csv")

    questions = load_questions(path_to_questions)
    results = []
    for n_question in track(sequence=range(len(questions)), description="Answering questions..."):
        question = questions[n_question]
        for index_name in INDEX_NAMES:
            answer, references = generate_answer(question, index_name)
            results.append(
                {
                    "ID": n_question + 1,
                    index_name: answer,
                    f"{index_name}_references": references,
                }
            )

    results = reorganize(results, questions)
    to_csv(results, output_filename, encoding="utf_8_sig")
    print(f"[DEBUG]: Answers are stored in '{output_filename}'")


if __name__ == "__main__":
    main()
