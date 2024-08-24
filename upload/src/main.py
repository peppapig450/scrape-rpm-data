import json
import logging
from os import getenv, path
import sys

from .embeddings import EmbeddingService
from .models import Professor
from .pinecone_client import PineconeClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


embedding_service = EmbeddingService()
pinecone_client = PineconeClient(
    api_key=getenv("PINECONE_API_KEY", ""),
    index_name=getenv("PINECONE_INDEX_NAME", ""),
)


def read_and_upsert_professors(file_path: str):
    logger.info(f"Reading professors from file: {file_path}")
    try:
        with open(file_path, "r") as file:
            professors_data = json.load(file)

        for data in professors_data:
            try:
                # Validate and parse the professor data
                professor = Professor(**data)
                logger.info(f"Processing professor: {professor.name}")

                # Generate professor embedding
                review_texts = [review.review for review in professor.reviews]
                logger.debug(f"Generating embeddings for {len(review_texts)} reviews")
                combined_embedding = embedding_service.generate_embeddings(
                    name=professor.name,
                    department=professor.department,
                    university=professor.university,
                    tags=professor.tags,
                    review_texts=review_texts,
                )
                logger.info(
                    f"Generated combined embedding of shape {combined_embedding.shape}"
                )

                # Prepare professor metadata
                professor_metadata = {
                    "name": professor.name,
                    "department": professor.department,
                    "university": professor.university,
                    "averageRating": professor.averageRating,
                }
                logger.debug(f"Prepared metadata: {professor_metadata}")

                # Upsert professor-level data
                professor_id = professor.name.replace(" ", "_")
                logger.info(f"Upserting professor data with ID: {professor_id}")
                pinecone_client.upsert_professor_embeddings(
                    professor_id, combined_embedding, professor_metadata
                )
                logger.info(f"Successfully upserted professor {professor.name}")
            except Exception as e:
                logger.error(
                    f"Error processing professor {data.get('name', 'Unknown')}: {str(e)}",
                    exc_info=True,
                )

    except Exception as e:
        logger.error(f"Error reading or processing file: {str(e)}", exc_info=True)


if __name__ == "__main__":
    json_file_path = getenv("PROFESSORS_JSON_PATH")
    if not json_file_path:
        print("Error: PROFESSORS_JSON_PATH environment variable is not set.")
        sys.exit(1)
    
    if not path.exists(json_file_path):
        print(f"Error: File not found: {json_file_path}")
        sys.exit(1)
    
    read_and_upsert_professors(json_file_path)
