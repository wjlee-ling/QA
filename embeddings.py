import os
from typing import List

from openai import OpenAI


def get_embeddings(input: List[str]) -> dict:
    """
    Returns `response`: `dict_keys(['data', 'model', 'object', 'usage'])`
    """
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=input,
    ).model_dump(mode="python")

    return response


def extract_embedding_vectors(responses: dict) -> list:
    vectors = []
    for response in responses["data"]:
        vectors.append(response["embedding"])
    return vectors
