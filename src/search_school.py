import logging
import re
import urllib.parse

from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
    Page,
)

logger = logging.getLogger(__name__)


def create_href_url(url: str) -> str:
    base_url = "https://www.ratemyprofessors.com"
    return urllib.parse.urljoin(base_url, url)


async def retry_click_show_more(page: Page, max_retries: int = 3):
    retries = 0
    while retries < max_retries:
        try:
            next_button = page.locator("button:has-text('Show More')")
            if await next_button.is_visible(timeout=15):
                await next_button.scroll_into_view_if_needed()
                logging.info("Clicking 'Show More' button to load more professors")
                await next_button.click()

                # Wait for the next page to load
                async with page.expect_response(
                    lambda response: response.status == 200,
                ):
                    await next_button.dispatch_event("event")

                # If we successfully clicked and loaded more professors, break out of the loop
                return True
            logging.info("No more 'Show More' button found. Ending search.")
            return False
        except PlaywrightTimeoutError:
            logging.warning(
                f"Timeout occurred when clicking 'Show More'. Retry {retries + 1}/{max_retries}"
            )
            retries += 1

            # If retries exceed max_retries, return False indicating the failure
            if retries >= max_retries:
                logging.error("Max retries exceeded. Unable to click 'Show More'.")
                return False

            # Open a new page and retry
            new_page = await page.context.new_page()
            await new_page.goto(page.url, wait_until="domcontentloaded")
            page = new_page


async def search_school_for_professor_links(
    school_name: str | None = None,
    school_url: str | None = None,
    max_professor_links: int = 100,
) -> set[str]:
    search_url = "https://www.ratemyprofessors.com"
    professor_links: set[str] = set()

    logging.info(f"Starting search for professors at {school_name}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        if school_url:
            # If a school URL is provided, use it directly
            logging.info(f"Using provided school URL: {school_url}")
            await page.goto(school_url, wait_until="domcontentloaded")
        elif school_name:
            # Otherwise, perform a search for the school
            logging.info(f"Starting search for professors at {school_name}")

            await page.goto(search_url, wait_until="domcontentloaded")

            logging.info(f"Searching for school: {school_name}")

            search_input = page.locator("input")
            await search_input.fill(school_name)
            await search_input.press("Enter")

            logging.info("Selecting the first school from search results")

            school_link = await page.locator(
                "a", has_text="ratings"
            ).first.get_attribute("href")
            if school_link:
                school_link = create_href_url(school_link)
                await page.goto(school_link, wait_until="domcontentloaded")
            else:
                logging.error("No school link found in search results.")
                return professor_links

        logging.info("Clicking 'View all Professors' link")

        professor_list_link = await page.locator(
            "a", has_text="View all Professors"
        ).first.get_attribute("href")

        if professor_list_link:
            professor_list_link = create_href_url(professor_list_link)
            await page.goto(professor_list_link, wait_until="domcontentloaded")

        await page.locator("button:has-text('Close')").click()

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
                                logging.info(
                                    f"Found professor with reviews: {full_url}"
                                )

            # Retry clicking 'Show More' if it times out
            if not await retry_click_show_more(page):
                break
    logging.info(
        f"Search completed. Total professor links found: {len(professor_links)}"
    )
    return professor_links
