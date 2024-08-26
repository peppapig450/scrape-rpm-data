import logging
from typing import Any

import numpy as np
from pinecone.grpc import PineconeGRPC as Pinecone

logger = logging.getLogger(__name__)


class PineconeClient:
    def __init__(self, api_key: str, index_name: str):
        self.pc = Pinecone(api_key)
        self.index = self.pc.Index(
            host="https://rpm-professor-reviews-prod-gz59xn8.svc.aped-4627-b74a.pinecone.io"
        )
        self.existing_ids: set[str] = set()  # Set to keep track of existing IDs
        self.refresh_existing_ids()

    def test_connection(self):
        """
        Test the connection to Pinecone index.
        """
        try:
            # Attempt to list index metadata to verify connection
            index_info = self.index.describe_index_stats()
            logger.info(
                f"Connection to Pinecone index successful. Index info: {index_info}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Error testing connection to Pinecone index: {str(e)}", exc_info=True
            )
            return False

    def refresh_existing_ids(self) -> None:
        """
        Refresh the set of existing IDs from Pinecone.
        """
        try:
            for id in self.index.list():
                self.existing_ids.update(id)
        except Exception as e:
            logger.error(f"Error refreshing existing IDs: {str(e)}", exc_info=True)
            raise

    def upsert_professor_embeddings(
        self, professor_id: str, embedding: np.ndarray, metadata: dict[str, Any]
    ) -> None:
        """
        Upsert professor embeddings and metadata into Pinecone.
        """
        logger.info(f"Upserting embeddings for professor: {professor_id}")
        logger.debug(f"Embedding shape: {embedding.shape}")
        logger.debug(f"Metadata: {metadata}")

        # Ensure the embedding is a 1D list
        embedding_list = embedding.flatten().tolist()

        if professor_id in self.existing_ids:
            logger.info(f"Professor ID {professor_id} already exists. Skipping upsert.")
            return

        try:
            self.index.upsert(vectors=[(professor_id, embedding_list, metadata)])
            self.existing_ids.add(professor_id)  # Add ID to the set
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
        self,
        ids: list[str],
        embeddings: list[np.ndarray],
        metadatas: list[dict[str, Any]],
    ) -> None:
        vectors = [
            (id, embedding.flatten().tolist(), metadata)
            for id, embedding, metadata in zip(ids, embeddings, metadatas)
        ]

        try:
            # Filter out existing IDs
            vectors_to_upsert = [v for v in vectors if v[0] not in self.existing_ids]
            if not vectors_to_upsert:
                logger.info("No new IDs to upsert.")
                return

            self.index.upsert(vectors=vectors_to_upsert)
            self.existing_ids.update(id for id, _, _ in vectors_to_upsert)
            logger.info(
                f"Successfully upserted batch of {len(vectors_to_upsert)} professors"
            )
        except Exception as e:
            logger.error(
                f"Error upserting batch of professors: {str(e)}", exc_info=True
            )
            raise
