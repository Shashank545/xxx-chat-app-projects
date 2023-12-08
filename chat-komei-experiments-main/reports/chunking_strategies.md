# Comparing RAG Performances of Different Text Chunking Strategies

## Introduction

- This document compares different text chunking methods, or strategies, used in **Retrieval-Augmented Generation (RAG)** systems.
- **Text chunking** is a crucial preprocessing step in RAG, as it helps to identify and organize meaningful and manageable units of text for retrieval and generation.
- During text chunking, large documents are separated into contiguous spans of the desired length.
- Currently implemented methods are as follows.

    1. Ad-hoc Sentence Grouping
    1. Recursive Character Text Splitter
    1. Auto Merging based Text Splitter
    1. Document (in fact Page) Summary based Text Splitter

## Setup

1. Provision an Azure Open AI Service instance.
1. Deploy (i) `text-embedding-ada-002`, (ii) `GPT-3.5`, and (iii) `GPT-4` models.
1. Provision an Azure Cognitive Search (ACS) instance.
1. Enable semantic search of the instance. You could do this on [Azure Portal](https://portal.azure.com).
1. Create a `.env` file and set the following environment variables:

    1. `OPENAI_API_KEY`
    1. `OPENAI_API_ENDPOINT`
    1. `SEARCH_API_KEY`
    1. `SEARCH_ENDPOINT`

1. Create a local Python environment with `requirements.txt`.
1. Place all PDF files to be chunked into `input/` directory.

## Methods

## Ad-hoc Sentence Grouping

Steps:

1. Read PDF page
1. Split page text into sentences (with an ad-hoc rule)
1. Concatenate each PDF_SENTENCES_PER_SECTION number of sentences into a single block (by adding a whitespace between each sentence within the block)
1. Move on to the next page

```bash
python src/step1_chunk_documents.py -m adhocsentencegrouping -i 'input/'
```

Example output -

```bash
python src/step1_chunk_documents.py -m adhocsentencegrouping -i prod.json

[INFO] chunking_method: 'adhocsentencegrouping'
[INFO] total_num_files: 143
[INFO] total_num_pages: 5143
[INFO] min(num_tokens_per_page): 0
[INFO] mean(num_tokens_per_page): 1346
[INFO] median(num_tokens_per_page): 1228
[INFO] max(num_tokens_per_page): 9899
[INFO] num_parent_chunks: 22020
[INFO] min(num_tokens_per_parent_chunk): 10
[INFO] mean(num_tokens_per_parent_chunk): 290
[INFO] median(num_tokens_per_parent_chunk): 288
[INFO] max(num_tokens_per_parent_chunk): 4886
[INFO] Chunks are stored in 'interim/adhocsentencegrouping_chunkstore_20230910_203503.json'
```

### Recursive Character Text Splitter

- `RecursiveCharacterTextSplitter` from [LangChain](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter) splits input text based on series of separators in order until the chunks are small enough.
- It aims to keep pieces of texts (paragraphs, sentences, and words) together as long as possible.
- This way it optimizes to create largest possible chunks of text which are semantically related.
- The `chunk_size` parameter determines the maximum size of each chunk (in number of tokens), while the `chunk_overlap` parameter specifies the number of characters that should overlap between consecutive chunks.

```bash
python src/step1_chunk_documents.py -m rcsplit -i 'input/'
```

Example output -

```bash
python src/step1_chunk_documents.py -m rcsplit -i prod.json

[INFO] chunking_method: 'rcsplit'
[INFO] total_num_files: 143
[INFO] total_num_pages: 5143
[INFO] min(num_tokens_per_page): 0
[INFO] mean(num_tokens_per_page): 1346
[INFO] median(num_tokens_per_page): 1228
[INFO] max(num_tokens_per_page): 9899
[INFO] num_parent_chunks: 10559
[INFO] min(num_tokens_per_parent_chunk): 1
[INFO] mean(num_tokens_per_parent_chunk): 771
[INFO] median(num_tokens_per_parent_chunk): 955
[INFO] max(num_tokens_per_parent_chunk): 1024
[INFO] Chunks are stored in 'interim/rcsplit_chunkstore_20230910_203540.json'
```

### Auto Merging Text Splitter

- This method establishes hierarchical relationships between chunks of larger sizes (parents) with chunks of smaller sizes (children).
- When querying, it first retrieves children (chunks of smaller sizes) based on embedding similarity.
- If the majority of the subset of these chunks are retrieved for a given parent and its children, then the parent is used downstream instead of individual children.
- This method aims to deliver more coherent contextual sections to the Large Language Model (LLM) used in answer synthesis.

```bash
python src/step1_chunk_documents.py -m automerging -i 'input/'
```

Example output -

```bash
python src/step1_chunk_documents.py -m automerging -i prod.json

[INFO] chunking_method: 'automerging'
[INFO] total_num_files: 143
[INFO] total_num_pages: 5143
[INFO] min(num_tokens_per_page): 0
[INFO] mean(num_tokens_per_page): 1346
[INFO] median(num_tokens_per_page): 1228
[INFO] max(num_tokens_per_page): 9899
[INFO] num_parent_chunks: 6077
[INFO] min(num_tokens_per_parent_chunk): 1
[INFO] mean(num_tokens_per_parent_chunk): 1233
[INFO] median(num_tokens_per_parent_chunk): 1241
[INFO] max(num_tokens_per_parent_chunk): 4228
[INFO] num_child_chunks: 17864
[INFO] min(num_tokens_per_child_chunk): 1
[INFO] mean(num_tokens_per_child_chunk): 527
[INFO] median(num_tokens_per_child_chunk): 654
[INFO] max(num_tokens_per_child_chunk): 1290
[INFO] Chunks are stored in 'interim/automerging_chunkstore_20230910_203605.json'
```

### Document (Page) Summary Text Splitter

Excerpt from [LlamaIndex Blog](https://medium.com/llamaindex-blog/a-new-document-summary-index-for-llm-powered-qa-systems-9a32ece2f9ec):

---

There are a few limitations of embedding retrieval using text chunks.

1. **Text chunks lack global context.** Oftentimes the question requires context beyond what is indexed in a specific chunk.
1. **Careful tuning of top-k / similarity score thresholds.** Make the value too small and you’ll miss context. Make the value too big and cost/latency might increase with more irrelevant context.
1. **Embeddings don’t always select the most relevant context for a question.** Embeddings are inherently determined separately between text and the context.

---

As a solution, the following approach is implemented.

1. During chunking, each PDF page is summarized using an LLM.
1. During chunking, each PDF page is also further split up into chunks of smaller sizes.
1. Both the summaries and the chunks of smaller sizes are stored in ACS with a mapping, effectively establishing summaries as parent entities and smaller chunks as their children.
1. During query-time, we first retrieve relevant pages to the query based on summary embedding similarity. We filter out children in this iteration.
1. Once relevant pages are identified, a second iteration of search is conducted over smaller chunks. We filter out parents (summaries) in this iteration.
1. After identifying relavant chunks, we use LLM to synthesize the answer.

```bash
python src/step1_chunk_documents.py -m summary -i 'input/'
```

Example output -

```bash
python src/step1_chunk_documents.py -m summary -i prod.json

[INFO] chunking_method: 'summary'
[INFO] total_num_files: 143
[INFO] total_num_pages: 5143
[INFO] min(num_tokens_per_page): 1
[INFO] mean(num_tokens_per_page): 771
[INFO] median(num_tokens_per_page): 955
[INFO] max(num_tokens_per_page): 1024
[INFO] num_parent_chunks: 5143
[INFO] min(num_tokens_per_parent_chunk): 0
[INFO] mean(num_tokens_per_parent_chunk): 295
[INFO] median(num_tokens_per_parent_chunk): 275
[INFO] max(num_tokens_per_parent_chunk): 2025
[INFO] num_child_chunks: 10559
[INFO] min(num_tokens_per_child_chunk): 1
[INFO] mean(num_tokens_per_child_chunk): 771
[INFO] median(num_tokens_per_child_chunk): 955
[INFO] max(num_tokens_per_child_chunk): 1024
[INFO] Chunks are stored in 'interim/summary_chunkstore_20230911_225738.json'
```

## Code

Main operations are carried out by these three scripts in the provided order:

1. `src/step1_chunk_documents.py`
1. `src/step2_setup_acs.py`
1. `src/step3_run_rag_pipeline.py`

- At step-1, we create text chunks.
- At step-2, we add embeddings to chunks and create an ACS index with the updated chunks.
- At step-3, we run the RAG pipeline end-to-end for a given set of test questions (`input/questions.csv`) and generate answers.

## References

1. [Recursive Char text splitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter)
1. [Document Summary Splitter](https://gpt-index.readthedocs.io/en/latest/examples/index_structs/doc_summary/DocSummary.html)
1. [Auto Merging Splitter](https://gpt-index.readthedocs.io/en/latest/examples/retrievers/auto_merging_retriever.html)
