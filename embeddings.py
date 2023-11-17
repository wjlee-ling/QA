import os
from typing import List

from openai import OpenAI


client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)


def get_embeddings(input: List[str]):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=input,
    )
    return response
