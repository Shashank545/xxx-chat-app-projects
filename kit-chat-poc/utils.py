"""Utilities for the project."""
import csv
import dataclasses
import json
from pathlib import Path
from typing import List, Union

import tiktoken


TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text) -> int:
    """Calculate length of tokens in a given input text using tiktoken tokenizer.

    Args:
        text (str): Input text

    Returns:
        int: Length of tokens calculated with tiktoken
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


def load_names(kitemp_filename: Path) -> List:
    """Parse list of KIT Employee names from given file

    Args:
        kitemp_filename (str): File conataining all KIT employees name
    """
    with open(kitemp_filename, "r") as foo:
        emp_names = foo.readlines()

    return emp_names
