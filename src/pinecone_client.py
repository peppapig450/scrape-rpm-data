import logging

import numpy as np
from pinecone import Pinecone

logger = logging.getLogger(__name__)


class PineconeClient:
    def __init__(self, api_key: str, index_name: str):
        self.pc = Pinecone(api_key)
        self.index = self.pc.Index(index_name)

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
