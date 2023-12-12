import os
import chromadb

from typing import List
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


class Indexer:
    """
    list를 받아서 임베딩 후 chromadb에 vector index/search를 수행하는 클래스
    """

    def __init__(self, persist_directory, collection_name):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=OpenAIEmbeddingFunction(
                model_name="text-embedding-ada-002",
                api_key=os.environ["OPENAI_API_KEY"],
            ),
        )
        print(f"@collection updated: {self.collection.count()}")
        self.configs = {
            "n_results": 5,
        }

    def add_embeddings(
        self,
        documents: List,
        metadatas: List,
        embeddings: List,
        ids: List,
    ):
        """
        chromaDB에 레코드 추가
        """
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids,
        )
        print(f"@collection updated: {self.collection.count()}")

    def search_similarity(
        self,
        queries: List[str],
        where_filter: dict = None,
        where_doc_filter: dict = None,
    ):
        results = self.collection.search(
            query_texts=queries,
            n_results=self.configs["n_results"],
            where=where_filter,
            where_document=where_doc_filter,
        )
        return results

    def find_alternatives(
        self,
        spam_df,
        query_column: str,
    ):
        """
        dataframe에서 query_column에 위치한 질의와 유사한 질의와 해당 질의의 답변을 반환하고 dataframe에 추가
        """
        for i, row in spam_df.iterrows():
            query = row[query_column]
            results = self.search_similarity([query])
            retrieved_queries = results["documents"][0]  # n_results만큼 결과
            retrieved_metadatas = results["metadatas"][0]

            for j in range(self.configs["n_results"]):
                try:
                    qa_alternative = (
                        retrieved_queries[j] + retrieved_metadatas[j]["수정 답변"]
                    )
                except:
                    qa_alternative = retrieved_queries[j] + retrieved_metadatas[j]["답변"]

                spam_df.loc[spam_df.index[i], f"유사QA{j+1}"] = qa_alternative

        return spam_df
