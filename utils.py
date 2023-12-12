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


def extract 

def create_metadata_from_dataframe(df, columns: List[str]) -> List[dict]:
    """
    DB의 metadata에 들어갈 데이터를 dataframe의 정해진 columns로만 추출하여 생성
    """
    col_metadata = df.apply(lambda row: {col: row[col] for col in columns}, axis=1)
    return col_metadata.to_list()


def update_dataframe(old_df, new_df):
    """
    기존 dataframe에 새로운 dataframe의 변경점을 적용
    """
    old_df.update(new_df)
    return old_df


def save_dataframe(df, path: str, index: bool = True):
    """
    수정할 파일은 index = True, 수정을 적용한 최종 파일은 index = False
    """
    df.to_excel(path, index=index)
