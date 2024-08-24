import argparse
import asyncio
import logging
from os import getenv

from tqdm import tqdm

from src.scrape_professor import scrape_professor
from src.search_school import search_school_for_professor_links
from src.embeddings import EmbeddingService
from src.models import Professor
from src.pinecone_client import PineconeClient

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
)

type ProfessorDictList = list[Professor]

# Initialize services
embedding_service = EmbeddingService()
pinecone_client = PineconeClient(
    api_key=getenv("PINECONE_API_KEY", ""),
    index_name=getenv("PINECONE_INDEX_NAME", ""),
)


async def scrape_school_professors(school_name: str, max_professor_pages: int):
    professor_links = await search_school_for_professor_links(
        school_name, max_professor_pages
    )
    return professor_links


def scrape_professor_links(professor_links: set[str]) -> ProfessorDictList:
    all_professors: ProfessorDictList = []

    for link in tqdm(professor_links, desc="Scraping professors"):
        professor = scrape_professor(link)
        if professor:
            all_professors.append(professor)

    return all_professors


def upsert_professors(professors: ProfessorDictList):
    for professor in professors:
        try:
            # Validate and parse the professor data
            logging.info(f"Processing professor: {professor.name}")

            # Generate professor embedding
            review_texts = [review.review for review in professor.reviews]
            logging.debug(f"Generating embeddings for {len(review_texts)} reviews")
            combined_embedding = embedding_service.generate_embeddings(
                name=professor.name,
                department=professor.department,
                university=professor.university,
                tags=professor.tags,
                review_texts=review_texts,
            )
            logging.info(
                f"Generated combined embedding of shape {combined_embedding.shape}"
            )

            # Prepare professor metadata
            professor_metadata = {
                "name": professor.name,
                "department": professor.department,
                "university": professor.university,
                "averageRating": professor.averageRating,
            }
            logging.debug(f"Prepared metadata: {professor_metadata}")

            # Upsert professor-level data
            professor_id = professor.name.replace(" ", "_")
            logging.info(f"Upserting professor data with ID: {professor_id}")
            pinecone_client.upsert_professor_embeddings(
                professor_id, combined_embedding, professor_metadata
            )
            logging.info(f"Successfully upserted professor {professor.name}")
        except Exception as e:
            logging.error(
                f"Error processing professor {professor.name}: {str(e)}",
                exc_info=True,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Scrape and upload professor reviews from a school."
    )
    parser.add_argument(
        "school_name", type=str, help="The name of the school to search."
    )
    parser.add_argument(
        "--max_professor_pages",
        type=int,
        default=50,
        help="The maximum number of professor pages to scrape per school.",
    )

    args = parser.parse_args()

    # Scrape professor links
    professor_links = asyncio.run(
        scrape_school_professors(
            school_name=args.school_name,
            max_professor_pages=args.max_professor_pages,
        )
    )

    # Scrape professor details and parse into Pydantic models
    all_professors = scrape_professor_links(professor_links)

    # Upsert professor data to Pinecone
    upsert_professors(all_professors)


if __name__ == "__main__":
    main()
