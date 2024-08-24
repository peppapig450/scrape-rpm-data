from playwright.async_api import async_playwright
import re
import logging
import urllib.parse
import asyncio

logger = logging.getLogger(__name__)


def create_href_url(url: str) -> str:
    base_url = "https://www.ratemyprofessors.com"
    return urllib.parse.urljoin(base_url, url)


async def search_school_for_professor_links(
    school_name: str, max_professor_links: int = 100
) -> set[str]:
    search_url = "https://www.ratemyprofessors.com"
    professor_links: set[str] = set()

    logging.info(f"Starting search for professors at {school_name}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(search_url, wait_until="domcontentloaded")
        logging.info(f"Searching for school: {school_name}")
        # Search for the school
        search_input = page.locator("input")
        await search_input.fill(school_name)
        await search_input.press("Enter")
        logging.info("Selecting the first school from search results")
        school_link = await page.locator("a", has_text="ratings").first.get_attribute(
            "href"
        )
        if school_link:
            school_link = create_href_url(school_link)
            await page.goto(school_link, wait_until="domcontentloaded")
        logging.info("Clicking 'View all Professors' link")
        professor_list_link = await page.locator(
            "a", has_text="View all Professors"
        ).first.get_attribute("href")
        if professor_list_link:
            professor_list_link = create_href_url(professor_list_link)
            await page.goto(professor_list_link)
        while len(professor_links) <= max_professor_links:
            logging.info(
                f"Scraping professor links, current count: {len(professor_links)}"
            )
            # Get all professor cards on the current page
            cards = await page.query_selector_all(
                "a.TeacherCard__StyledTeacherCard-syjs0d-0"
            )
            for card in cards:
                # Extract the review count
                review_count_elem = await card.query_selector(
                    ".CardNumRating__CardNumRatingCount-sc-17t4b9u-3"
                )
                if review_count_elem:
                    review_count_text = await review_count_elem.inner_text()
                    if match := re.search(r"\d+", review_count_text):
                        review_count = int(match.group())
                        if review_count > 0:
                            href = await card.get_attribute("href")
                            full_url = create_href_url(href) if href else None
                            if full_url and full_url not in professor_links:
                                professor_links.add(full_url)
                                logging.info(f"Found professor with reviews: {full_url}")
            next_button = page.locator(
                "button:has-text('Show More')"
            )  # Ensure correct selector
            if await next_button.is_visible():
                logging.info("Clicking 'Show More' button to load more professors")

                async with page.expect_response(
                    lambda response: response.status == 200,
                ):
                    await page.locator("button:has-text('Show More')").dispatch_event('click')

            else:
                logging.info("No more 'Show More' button found. Ending search.")
                break
    logging.info(
        f"Search completed. Total professor links found: {len(professor_links)}"
    )
    return professor_links
