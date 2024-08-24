import argparse
import asyncio
import logging
from os import getenv

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from src.embeddings import EmbeddingService
from src.models import Professor
from src.pinecone_client import PineconeClient
from src.scrape_professor import scrape_professor
from src.search_school import search_school_for_professor_links

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


async def scrape_professor_links(professor_links: set[str]) -> ProfessorDictList:
    all_professors: ProfessorDictList = []

    async with aiohttp.ClientSession() as session:

        tasks = [
            scrape_professor(session, link)
            for link in tqdm(professor_links, desc="Scraping professors")
        ]
        all_professors = await tqdm.gather(*tasks, desc="Scraping professors")

    return [professor for professor in all_professors if professor is not None]


def generate_embeddings_batch(
    embedding_service: EmbeddingService, professors: ProfessorDictList
) -> list[tuple[str, np.ndarray, dict]]:
    batch_data = []
    for professor in professors:
        try:
            logging.info(f"Generating embedding for professor: {professor.name}")
            review_texts = [review.review for review in professor.reviews]
            combined_embedding = embedding_service.generate_embeddings(
                name=professor.name,
                department=professor.department,
                university=professor.university,
                tags=professor.tags,
                review_texts=review_texts,
            )

            professor_metadata = {
                "name": professor.name,
                "department": professor.department,
                "university": professor.university,
                "averageRating": professor.averageRating,
            }

            professor_id = professor.name.replace(" ", "_")
            batch_data.append((professor_id, combined_embedding, professor_metadata))
        except Exception as e:
            logging.error(
                f"Error processing professor {professor.name}: {str(e)}", exc_info=True
            )

    return batch_data


def upsert_professors_batch(professors: ProfessorDictList, batch_size: int = 100):
    all_batch_data = generate_embeddings_batch(embedding_service, professors)

    for i in range(0, len(all_batch_data), batch_size):
        batch = all_batch_data[i : i + batch_size]
        ids, embeddings, metadatas = zip(*batch)

        try:
            logging.info(f"Upserting batch of {len(batch)} professors")
            pinecone_client.upsert_batch(ids, embeddings, metadatas)
            logging.info(f"Successfully upserted batch of {len(batch)} professors")
        except Exception as e:
            logging.error(f"Error upserting batch: {str(e)}", exc_info=True)


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
    all_professors = asyncio.run(scrape_professor_links(professor_links))

    # Upsert professor data to Pinecone
    upsert_professors_batch(all_professors)


if __name__ == "__main__":
    main()
