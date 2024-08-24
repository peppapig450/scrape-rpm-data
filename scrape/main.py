import argparse
import asyncio
import json
import logging
from typing import Any

from tqdm import tqdm

from src.scrape_professor import scrape_professor
from src.search_school import search_school_for_professor_links

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

type ProfessorDictList = list[dict[str, Any]]

async def scrape_school_professors(
    school_name: str, max_professor_pages: int
):
    professor_links = await search_school_for_professor_links(
        school_name, max_professor_pages
    )
    
    return professor_links

def scrape_professor_links(professor_links: set[str], output_filename: str):
    all_professors: ProfessorDictList = []
    
    for link in tqdm(professor_links, desc="Scraping professors"):
        professor = scrape_professor(link)
        if professor:
            all_professors.append(professor.model_dump(mode='json'))
            
    with open(output_filename, "w") as file:
        json.dump(all_professors, file, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape professor reviews from a school."
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
    parser.add_argument(
        "--output_filename",
        type=str,
        default="school_professors.json",
        help="The file to save the output JSON.",
    )

    args = parser.parse_args()

    professor_links = asyncio.run(scrape_school_professors(
        school_name=args.school_name,
        max_professor_pages=args.max_professor_pages,
    ))
    
    scrape_professor_links(professor_links, args.output_filename)


if __name__ == "__main__":
    main()
