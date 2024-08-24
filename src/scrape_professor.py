import logging
import re
from datetime import datetime

import requests
from bs4 import BeautifulSoup, Tag

from .models import Attendance, Grade, Professor, ProfessorReview, YesNo

logger = logging.getLogger(__name__)


def parse_date(date_string: str):
    """Parses a date string in the format "Month Dayth, Year" and returns a datetime object.

    Args:
      date_str: The date string to parse.

    Returns:
      A datetime object representing the parsed date.
    """
    # Remove the ordinal suffix (st, nd, rd, th) from the day
    date_string = re.sub(r"(st|nd|rd|th),", "", date_string)

    # Parse the modified string
    return datetime.strptime(date_string, "%b %d %Y")


def get_text_content(soup: BeautifulSoup | Tag, selector: str, parent=None) -> str:
    element = (parent or soup).select_one(selector)
    if element:
        return element.text.strip()
    logging.info(f"Selector: {selector} failed to find element")
    return ""


def get_meta_item_value(review_element, keyword: str) -> str | None:
    meta_item = review_element.find(
        "div", text=keyword, class_="MetaItem__StyledMetaItem-y0ixml-0"
    )
    return meta_item.find_next("span").text.strip() if meta_item else None


def get_optional_meta_value(review_element, keyword: str, parse_func):
    value = get_meta_item_value(review_element, keyword)
    return parse_func(value) if value else None


def scrape_professor(url: str) -> Professor | None:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml")

        name = get_text_content(soup, ".NameTitle__Name-dowf0z-0")
        department = get_text_content(
            soup, ".TeacherDepartment__StyledDepartmentLink-fl79e8-0 > b"
        )[: -len("department")]
        university = get_text_content(
            soup, "div.NameTitle__Title-dowf0z-1 > a:nth-child(2)"
        )
        average_rating = float(
            get_text_content(soup, ".RatingValue__Numerator-qw8sqy-2.liyUjw")
        )
        num_ratings = int(
            get_text_content(soup, ".RatingValue__NumRatings-qw8sqy-0 a").split()[0]
        )

        tags = [
            tag.text.strip()
            for tag in soup.select(".TeacherTags__TagsContainer-sc-16vmh1y-0 span")
        ]
        reviews = []

        for review_element in soup.select(".Rating__StyledRating-sc-1rhvpxz-1"):
            votes_list = review_element.select(".Thumbs__HelpTotalNumber-sc-19shlav-2")
            helpfulVotes = ""
            unhelpfulVotes = ""

            if votes_list:
                helpfulVotes = votes_list[0].text.strip()
                unhelpfulVotes = votes_list[1].text.strip()

            review = ProfessorReview(
                quality=float(
                    get_text_content(
                        review_element,
                        ".CardNumRating__CardNumRatingNumber-sc-17t4b9u-2",
                    )
                ),
                difficulty=float(
                    get_text_content(
                        review_element,
                        ".CardNumRating__CardNumRatingNumber-sc-17t4b9u-2.cDKJcc",
                    )
                ),
                course=get_text_content(
                    review_element, ".RatingHeader__StyledClass-sc-1dlkqw1-3.eXfReS"
                ),
                date=parse_date(
                    get_text_content(
                        review_element, ".TimeStamp__StyledTimeStamp-sc-9q2r30-0"
                    )
                ),
                review=get_text_content(
                    review_element, ".Comments__StyledComments-dzzyvm-0"
                ),
                helpfulVotes=int(helpfulVotes),
                unhelpfulVotes=int(unhelpfulVotes),
                textbook=get_optional_meta_value(
                    review_element,
                    "Textbook",
                    lambda v: YesNo.YES if v == "Yes" else YesNo.NO,
                ),
                forCredit=get_optional_meta_value(
                    review_element,
                    "For Credit",
                    lambda v: YesNo.YES if v == "Yes" else YesNo.NO,
                ),
                attendence=get_optional_meta_value(
                    review_element,
                    "Attendance",
                    lambda v: (
                        Attendance.MANDATORY
                        if v == "Mandatory"
                        else Attendance.NOT_MANDATORY
                    ),
                ),
                grade=get_optional_meta_value(review_element, "Grade", Grade),
                wouldTakeAgain=get_optional_meta_value(
                    review_element,
                    "Would Take Again",
                    lambda v: YesNo.YES if v == "Yes" else YesNo.NO,
                ),
                tags=[
                    tag.text.strip() for tag in review_element.select(".Tag-bs9vf4-0")
                ],
            )
            reviews.append(review)

        professor = Professor(
            name=name,
            department=department,
            university=university,
            averageRating=average_rating,
            numRatings=num_ratings,
            tags=tags,
            reviews=reviews,
        )

        return professor

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None
    except ValueError as e:
        logger.error(f"Parsing error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None
