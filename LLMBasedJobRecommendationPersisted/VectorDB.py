import chromadb
import CustomEmbedding
from logging import Logger

class VectorDBForJobs:
    __COLLECTION_NAME = 'JOBS_COLLECTION'
    __n_Top_results = 3

    def __init__(self, logger: Logger):
        self.__logger = logger
        # TODO: switch to persistent client
        self.__client = chromadb.Client()
        self.__collection = self.__client.get_or_create_collection(
            self.__COLLECTION_NAME, embedding_function=CustomEmbedding(self.__logger))
        
    # TODO: Add metadata of job title
    def upsertToCollection(self, documents: chromadb.Documents, ids: chromadb.IDs):
        self.__collection.upsert(ids, documents)

    def queryFromCollection(self, 
                            documents: chromadb.Documents, 
                            n_results=__n_Top_results) -> tuple[chromadb.Documents, list[str]]:
        results = self.__collection.query(query_texts=documents, n_results=n_results)

        return (results['documents'], results['ids'])
