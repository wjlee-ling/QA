from utils import *
from indexer import Indexer

import argparse
import pandas as pd


def create_spam(args):
    old_path = args.path
    save_path = args.save_path

    indexer = Indexer(persist_directory="", collection_name="")
    collection = indexer.collection

    # import old (crew) dataframe and copy it to new dataframe
    old_df = pd.read_excel(old_path)

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

    save_dataframe(spam, path=save_path, index=True)  # 인덱스 필수


def update_ham_with_spam(args):
    old_path, new_path, save_path = args.path, args.new_path, args.save_path
    old_df = pd.read_excel(old_path)
    new_df = pd.read_excel(new_path, index_col=0)
    print(new_df.head())

    # update changes
    old_df.update(new_df)
    save_dataframe(old_df, path=save_path, index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, type=str, help="mode to run")
    parser.add_argument(
        "--path", required=True, type=str, help="path to the excel file"
    )
    parser.add_argument("--new-path", type=str, help="path to the file to update with")
    parser.add_argument("--save-path", type=str, help="path to the file to save")
    args = parser.parse_args()

    if args.mode == "create":
        create_spam(args)
    elif args.mode == "update":
        update_ham_with_spam(args)
