import argparse
import asyncio
import logging
from os import getenv
from statistics import mean

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from src.embeddings import EmbeddingService
from src.models import Professor, YesNo
from src.pinecone_client import PineconeClient
from src.review_filter import ProfessorFilter
from src.scrape_professor import scrape_professor
from src.search_school import search_school_for_professor_links

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
)

type ProfessorDictList = list[Professor]


class ProfessorEmbeddingsProcessor:
    def __init__(self, pinecone_api_key: str, pinecone_index_name: str) -> None:
        self.embedding_service = EmbeddingService()
        self.pinecone_client = PineconeClient(pinecone_api_key, pinecone_index_name)
        self.filter = ProfessorFilter()

    async def scrape_school_professors(
        self, school_name: str, max_professor_pages: int
    ):
        professor_links = await search_school_for_professor_links(
            school_name, max_professor_pages
        )
        return professor_links

    async def scrape_professor_links(
        self, professor_links: set[str]
    ) -> ProfessorDictList:
        all_professors: ProfessorDictList = []

        async with aiohttp.ClientSession() as session:

            tasks = [
                scrape_professor(session, link)
                for link in tqdm(professor_links, desc="Scraping professors")
            ]
            all_professors = await tqdm.gather(*tasks, desc="Scraping professors")

        return [professor for professor in all_professors if professor is not None]

    def generate_embeddings_batch(
        self, professors: ProfessorDictList
    ) -> list[tuple[str, np.ndarray, dict]]:
        batch_data = []
        for professor in professors:
            try:
                logging.info(f"Generating embedding for professor: {professor.name}")

                # Filter and clean the professor's reviews
                filtered_reviews = self.filter.filter_reviews(professor.reviews)
                print(filtered_reviews)
                tags = professor.tags

                # Generate embeddings for the tags and reviews
                review_texts = [review.review for review in filtered_reviews]
                combined_embedding = (
                    self.embedding_service.generate_professor_embedding(
                        tags=tags, review_texts=review_texts
                    )
                )

                # Calculate metadata
                top_reviews_avg_rating = mean(
                    [review.quality for review in filtered_reviews]
                )
                would_take_again_percentage = (
                    (
                        sum(
                            1
                            for review in professor.reviews
                            if review.wouldTakeAgain == YesNo.YES
                        )
                        / len(professor.reviews)
                    )
                    * 100
                    if professor.reviews
                    else 0
                )

                professor_metadata = {
                    "name": professor.name,
                    "university": professor.university,
                    "averageRating": professor.averageRating,
                    "topReviewsAvgRating": round(top_reviews_avg_rating, 2),
                    "numRatings": professor.numRatings,
                    "wouldTakeAgainPercentage": round(would_take_again_percentage, 2),
                    "tags": ", ".join(tags),
                    "reviews_summary": "; ".join(
                        [
                            f"{review.review} (Helpful: {review.helpfulVotes})"
                            for review in filtered_reviews
                        ]
                    ),
                }

                professor_id = professor.name.replace(" ", "_")
                batch_data.append(
                    (professor_id, combined_embedding, professor_metadata)
                )
            except Exception as e:
                logging.error(
                    f"Error processing professor {professor.name}: {str(e)}",
                    exc_info=True,
                )

        return batch_data

    def upsert_professors_batch(
        self, professors: ProfessorDictList, batch_size: int = 100
    ):
        all_batch_data = self.generate_embeddings_batch(professors)

        for i in range(0, len(all_batch_data), batch_size):
            batch = all_batch_data[i : i + batch_size]
            ids, embeddings, metadatas = zip(*batch)

            try:
                logging.info(f"Upserting batch of {len(batch)} professors")
                self.pinecone_client.upsert_batch(ids, embeddings, metadatas)
                logging.info(f"Successfully upserted batch of {len(batch)} professors")
            except Exception as e:
                logging.error(f"Error upserting batch: {str(e)}", exc_info=True)

    def run(self, school_name: str, max_professor_pages: int):
        # Scrape professor links
        professor_links = asyncio.run(
            self.scrape_school_professors(
                school_name=school_name,
                max_professor_pages=max_professor_pages,
            )
        )

        # Scrape professor details and parse into Pydantic models
        all_professors = asyncio.run(self.scrape_professor_links(professor_links))

        # Upsert professor data to Pinecone
        self.upsert_professors_batch(all_professors)


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

    if not (api_key := getenv("PINECONE_API_KEY")):
        raise ValueError("Pinecone api key not defined.")

    if not (index_name := getenv("PINECONE_INDEX_NAME")):
        raise ValueError("Pinecone index name not defined.")

    runner = ProfessorEmbeddingsProcessor(api_key, index_name)
    runner.run(
        school_name=args.school_name, max_professor_pages=args.max_professor_pages
    )


if __name__ == "__main__":
    main()
