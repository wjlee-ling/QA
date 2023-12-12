from utils import *
from indexer import Indexer

import pandas as pd


indexer = Indexer(persist_directory="", collection_name="")
collection = indexer.collection


def create_spam():
    # import old (crew) dataframe and copy it to new dataframe
    old_df = pd.read_excel("")

    spam = old_df[old_df["2차 검수 요청"] == True]
    ham = old_df[old_df["2차 검수 요청"] == False]

    # get vector embeddings from ham and store them to chroma DB
    embeddings = get_embeddings(ham[""].to_list())
    embedding_vectors = extract_embedding_vectors(embeddings)
    documents = ham[""]  # 질문열
    metadatas = create_metadata_from_dataframe(ham, ["", ""])  # 메타데이터로 저장한 cols
    indices = ham.index.astype("str").to_list()

    indexer.add_embeddings(
        documents=documents,
        metadatas=metadatas,
        embeddings=embedding_vectors,
        ids=indices,
    )

    # 수정할 spam 파일에 유사 질의응답 열 추가하여 저장
    spam = indexer.find_alternatives(spam, query_column="")

    save_dataframe(spam, path="", index=True)  # 인덱스 필수


def update_ham_with_spam(old_path, new_path):
    old_df = pd.read_excel(old_path)
    new_df = pd.read_excel(new_path, index_col=0)

    # update changes
    old_df.update(new_df)
    save_dataframe(old_df, path="", index=True)
