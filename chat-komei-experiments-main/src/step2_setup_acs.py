"""Load chunkstores from disk and index them in Azure Cognitive Search (ACS).

Usage:

  python src/step2_setup_acs.py
"""
import json
import os
import pickle
import sys
import time
from datetime import datetime
from typing import List

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswVectorSearchAlgorithmConfiguration,
    PrioritizedFields,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    SimpleField,
    VectorSearch
)
from dotenv import find_dotenv, load_dotenv
from rich import print
from rich.progress import track

try:
    from src.utils import generate_embeddings, get_filenames, load_json
except ImportError:
    from utils import generate_embeddings, get_filenames, load_json

load_dotenv(find_dotenv())

ACS_ENDPOINT = os.environ.get("SEARCH_ENDPOINT")
ACS_CREDENTIAL = AzureKeyCredential(os.environ.get("SEARCH_API_KEY"))
EMBEDDING_DIMENSION = 1536
SLEEP_TIME_FOR_INDEXING_IN_SEC = 0.1
SAVE_CHUNKS_TO_DISK = True
ACS_FIELDS_PDF = [
    SimpleField(
        name="id",
        type="Edm.String",
        key=True,
        filterable=True,
    ),
    SimpleField(
        name="parent_id",
        type="Edm.String",
        filterable=True,
    ),
    SearchableField(
        name="title",
        type="Edm.String",
        analyzer_name="ja.lucene",
    ),
    SearchField(
        name="title_embedding",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=EMBEDDING_DIMENSION,
        vector_search_configuration="default",
    ),
    SearchableField(
        name="content",
        type="Edm.String",
        analyzer_name="ja.lucene",
    ),
    SearchField(
        name="content_embedding",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=EMBEDDING_DIMENSION,
        vector_search_configuration="default",
    ),
    SimpleField(
        name="source_path",
        type="Edm.String",
    ),
    SimpleField(
        name="page_num",
        type="Edm.Int32",
    ),
    SimpleField(
        name="num_tokens",
        type="Edm.Int32",
    ),
    SimpleField(
        name="modified_from_source",
        type="Edm.Boolean",
    ),
]


def create_search_index_object(index_name: str, fields: list) -> SearchIndex:
    """Create search index object.

    For vector search, we use Hierarchical Navigable Small World (HNSW)
    approximate nearest neighbors algorithm.
    """
    return SearchIndex(
        name=index_name,
        fields=fields,
        semantic_settings=SemanticSettings(
            configurations=[
                SemanticConfiguration(
                    name="default",
                    prioritized_fields=PrioritizedFields(
                        title_field=None,
                        prioritized_content_fields=[
                            SemanticField(field_name="content")
                        ],
                    ),
                )
            ]
        ),
        vector_search=VectorSearch(
            algorithm_configurations=[
                HnswVectorSearchAlgorithmConfiguration(
                    name="default",
                    kind="hnsw",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 1000,
                        "metric": "cosine"
                    }
                )
            ]
        ),
    )


def create_search_index(
    *,
    index_name: str,
    fields: list,
    verbose: bool = True
) -> None:
    """Create ACS index if it does not exist already.

    For a list of all supported ACS REST API versions see the following Microsoft document:

    https://learn.microsoft.com/en-us/rest/api/searchservice/search-service-api-versions
    """
    index_client = SearchIndexClient(
        endpoint=ACS_ENDPOINT,
        credential=ACS_CREDENTIAL,
        api_version="2023-07-01-Preview",
    )
    if index_name not in index_client.list_index_names():
        index: SearchIndex = create_search_index_object(index_name, fields)
        if verbose:
            print(f"[INFO] Creating search index '{index_name}'")
        index_client.create_index(index)
    else:
        if verbose:
            print(f"[INFO] Search index '{index_name}' already exists")


def populate_index(
    index_name: str,
    chunks: List[dict],
    verbose: bool = True,
) -> None:
    """Index target documents."""
    if verbose:
        print(
            f"[INFO] Populating index '{index_name}'"
        )

    search_client = SearchClient(
        endpoint=ACS_ENDPOINT,
        index_name=index_name,
        credential=ACS_CREDENTIAL,
        api_version="2023-07-01-Preview",
    )

    batch = []
    results = []
    for chunk in track(sequence=chunks, description="Populating index..."):
        time.sleep(SLEEP_TIME_FOR_INDEXING_IN_SEC)
        batch.append(chunk)
        if sys.getsizeof(json.dumps(batch)) >= 1.5e+7:
            results += search_client.upload_documents(documents=batch)
            batch = []

    if len(batch) > 0:
        results += search_client.upload_documents(documents=batch)

    num_success = sum([1 for record in results if record.succeeded])
    if verbose:
        print(f"[INFO] Indexed {len(results)} chunks, {num_success} succeeded")


def main():
    """Orchestrate indexing."""
    # Get input file names
    filenames = get_filenames("interim", "json")

    # Process input files one by one
    # Create a separate index for each
    for filename in filenames:
        index_name = filename.stem.split("_chunkstore_")[0]
        print(f"[INFO] Setting up index '{index_name}'...")

        # [1] Read chunks from disk
        chunks = load_json(filename)
        print(f"[INFO] Loaded {len(chunks)} chunks from disk")

        # [2] Add title and content embeddings to chunks and
        # save the final form of the chunks to disk if needed
        title_embeddings = {chunk["title"]: [] for chunk in chunks}
        for title in track(sequence=title_embeddings.keys(), description="Embedding titles..."):
            title_embeddings[title] = generate_embeddings(title)
        for chunk in track(sequence=chunks, description="Embedding contents..."):
            chunk["title_embedding"] = title_embeddings[chunk["title"]]
            chunk["content_embedding"] = generate_embeddings(chunk["content"])
        num_errors = sum(
            [
                "content_embedding" not in chunk
                or chunk["content_embedding"] is None
                for chunk in chunks
            ]
        )
        print(f"[DEBUG] num_errors_during_embedding: {num_errors}")
        if SAVE_CHUNKS_TO_DISK:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"{index_name}_{timestamp}.pickle", "wb") as fout:
                pickle.dump(chunks, fout)

        # [3] Create ACS index if it does not exist
        create_search_index(
            index_name=index_name,
            fields=ACS_FIELDS_PDF,
            verbose=True
        )

        # [4] Upload updated chunks to ACS
        populate_index(
            index_name=index_name,
            chunks=chunks,
            verbose=True,
        )

    return


if __name__ == "__main__":
    main()
