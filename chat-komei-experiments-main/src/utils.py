"""Utilities for the project."""
import csv
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import List, Union

import openai
import tiktoken
from dotenv import find_dotenv, load_dotenv

try:
    from src.constants import (
        MAX_ALLOWED_TOKEN_COUNT_FOR_TEXT_EMBEDDING_ADA_002,
        OPENAI_DEPLOYMENT_TEXT_EMBEDDING_ADA_002,
        OPENAI_PRICING_PER_TOKEN
    )
except ImportError:
    from constants import (
        MAX_ALLOWED_TOKEN_COUNT_FOR_TEXT_EMBEDDING_ADA_002,
        OPENAI_DEPLOYMENT_TEXT_EMBEDDING_ADA_002,
        OPENAI_PRICING_PER_TOKEN
    )


TOKENIZER = tiktoken.get_encoding("cl100k_base")
ENCODER = tiktoken.encoding_for_model("text-embedding-ada-002")

load_dotenv(find_dotenv())
openai.api_type = "azure"
openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_base = os.environ.get("OPENAI_API_ENDPOINT")
openai.api_version = "2023-05-15"


def generate_embeddings(
    text: str,
    sleep_time_in_sec: float = 0.2,
    max_num_retries: int = 5
) -> List[float]:
    """Generate embedding for the input text using text-embedding-ada-002."""
    time.sleep(sleep_time_in_sec)
    text = ENCODER.decode(
        ENCODER.encode(text)[:MAX_ALLOWED_TOKEN_COUNT_FOR_TEXT_EMBEDDING_ADA_002]
    )
    retry = 0
    while retry < max_num_retries:
        try:
            response = openai.Embedding.create(
                input=text,
                engine=OPENAI_DEPLOYMENT_TEXT_EMBEDDING_ADA_002
            )
            return response["data"][0]["embedding"]
        except Exception as err:
            print(f"[WARNING] [{retry + 1}] {err}. Retrying...")
            time.sleep(0.5)
        retry += 1
    return None


def count_tokens(text: str) -> int:
    """Count tokens in a given piece of text.

    Args:
        text (str): Input text

    Returns:
        int: Number of tokens in the input text
    """
    tiktokens = TOKENIZER.encode(text, disallowed_special=())
    return len(tiktokens)


def get_filenames(
    inp_dir: Union[str, Path],
    ext: str = "pdf",
    recursively: bool = False
) -> List[Path]:
    """Get file names."""
    inp_dir = Path(inp_dir)
    if recursively:
        return list(inp_dir.glob(f"**/*.{ext.lower()}"))
    return list(inp_dir.glob(f"*.{ext.lower()}"))


def to_json(filename: str, data: list) -> None:
    """Save list of dictionary like objects to disk."""
    if not isinstance(data, list) or not data:
        # 'data' MUST BE of type list
        return None

    if not all(dataclasses.is_dataclass(record) or isinstance(record, dict) for record in data):
        # All elements of 'data' MUST BE of type dict or dataclass
        return None

    if dataclasses.is_dataclass(data[0]):
        data = [dataclasses.asdict(record) for record in data]

    with open(filename, mode="w") as fout:
        json.dump(
            data,
            fout,
            indent=4,
            sort_keys=False,
            ensure_ascii=False
        )


def load_json(filename: Path) -> List[dict]:
    """Load data from disk."""
    with open(filename, mode="r") as fin:
        data = json.load(fin)
    return data


def nonewlines(text: str) -> str:
    """Remove new lines from text."""
    return text.replace("\n", " ").replace("\r", " ")


def calculate_cost(model: str, usage: dict):
    """Calculate estimated cost based on the # of tokens."""
    cost = 0
    for key in OPENAI_PRICING_PER_TOKEN[model]:
        cost += OPENAI_PRICING_PER_TOKEN[model][key] * usage[key]
    return round(cost, 2)


def to_csv(data: List[dict], csv_filename: Path, encoding: str = "utf_8") -> None:
    """Save a list of dictionaries to a CSV file.

    - Keep column headers
    """
    fieldnames = list(data[0].keys())
    with open(csv_filename, mode="w", newline="", encoding="utf_8_sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    return None
