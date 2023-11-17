import os
from typing import List

from openai import OpenAI


def get_embeddings(input: List[str]):
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=input,
    )
    return response
