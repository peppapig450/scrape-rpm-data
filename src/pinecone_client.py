import logging

import numpy as np
from pinecone.grpc import PineconeGRPC as Pinecone

logger = logging.getLogger(__name__)


class PineconeClient:
    def __init__(self, api_key: str, index_name: str):
        self.pc = Pinecone(api_key)
        self.index = self.pc.Index(host="https://test-index-gz59xn8.svc.aped-4627-b74a.pinecone.io")
        
    def test_connection(self):
        """
        Test the connection to Pinecone index.
        """
        try:
            # Attempt to list index metadata to verify connection
            index_info = self.index.describe_index_stats()
            logger.info(f"Connection to Pinecone index successful. Index info: {index_info}")
            return True
        except Exception as e:
            logger.error(f"Error testing connection to Pinecone index: {str(e)}", exc_info=True)
            return False

    def upsert_professor_embeddings(
        self, professor_id: str, embedding: np.ndarray, metadata: dict
    ):
        """
        Upsert professor embeddings and metadata into Pinecone.
        """
        logger.info(f"Upserting embeddings for professor: {professor_id}")
        logger.debug(f"Embedding shape: {embedding.shape}")
        logger.debug(f"Metadata: {metadata}")

        # Ensure the embedding is a 1D list
        embedding_list = embedding.flatten().tolist()
        
        try:
            self.index.upsert(vectors=[(professor_id, embedding_list, metadata)])
            logger.info(
                f"Successfully upserted embeddings for professor: {professor_id}"
            )
        except Exception as e:
            logger.error(
                f"Error upserting embeddings for professor {professor_id}: {str(e)}",
                exc_info=True,
            )
            raise 

    def upsert_batch(
        self, ids: list[str], embeddings: list[np.ndarray], metadatas: list[dict]
    ):
        vectors = [
            (id, embedding.flatten().tolist(), metadata)
            for id, embedding, metadata in zip(ids, embeddings, metadatas)
        ]
        
        try:
            self.index.upsert(vectors=vectors)
            logger.info(f"Successfully upserted batch of {len(ids)} professors")
        except Exception as e:
            logger.error(
                f"Error upserting batch of professors: {str(e)}",
                exc_info=True,
            )
            raise
