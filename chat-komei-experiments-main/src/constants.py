"""Constants used in the project."""
from enum import IntEnum

# OpenAI deployment names
OPENAI_DEPLOYMENT_GPT_35_TURBO = "gpt-35-turbo"
OPENAI_DEPLOYMENT_GPT_4 = "gpt-4"
OPENAI_DEPLOYMENT_TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
MAX_ALLOWED_TOKEN_COUNT_FOR_TEXT_EMBEDDING_ADA_002 = 8191

# Azure OpenAI pricing
# https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
OPENAI_PRICING_PER_TOKEN = {
    OPENAI_DEPLOYMENT_GPT_35_TURBO: {
        "prompt_tokens": 0.281e-3,
        "completion_tokens": 0.281e-3,
    },
    OPENAI_DEPLOYMENT_GPT_4: {
        "prompt_tokens": 4.215e-3,
        "completion_tokens": 8.430e-3,
    },
    OPENAI_DEPLOYMENT_TEXT_EMBEDDING_ADA_002: {
        "prompt_tokens": 0.014455e-3,
    }
}


class SearchOption(IntEnum):
    """Azure Cognitive Search - Search options."""

    BM25 = 0
    Semantic = 1
    Vector = 2
    VectorBM25 = 3
    VectorSemantic = 4


# Prompt templates
SYSTEM_PROMPT_GENERATE_ANSWER = """
あなたは、公認会計士のアシスタントです。以下の情報をもとに回答を出力してください。
提供された情報に基づいて回答し、質問に対する情報が不十分な場合は、「わかりません」と回答してください。
"""

USER_PROMPT_GENERATE_QUESTION = """
{content}

上の情報をもとに、「{question}」という質問に対してより正確な回答が得られる日本語の質問文を１つ出力してください。
"""

USER_PROMPT_GENERATE_ANSWER = """
#Source
{source}
#Question: {question}
#Answer:
"""

USER_PROMPT_CONFIRMING_ANSWER = """
回答文「{answer}」が質問文「{question}」に対して適切であれば、「はい」と出力して下さい。
適切でなければ、回答文をより適切な形に書き換えて、出力して下さい。
"""

CONTEXT_LENGTHS = {
    "gpt-4": 8192,
    "gpt-35-turbo": 4096,
}
