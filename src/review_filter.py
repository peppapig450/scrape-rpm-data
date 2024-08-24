from datetime import datetime, timedelta
from .models import ProfessorReview


class ProfessorFilter:
    def __init__(
        self, review_length_treshold: int = 50, years_limit: int = 6, top_n: int = 10
    ) -> None:
        self.review_length_threshold = review_length_treshold
        self.years_limit = years_limit
        self.top_n = top_n
        self.current_year = datetime.now().year

    def filter_reviews(self, reviews: list[ProfessorReview]) -> list[ProfessorReview]:
        """Applies all filtering criteria and returns top reviews."""

        # Filter by recency and review length
        recent_reviews = self._filter_by_recency(reviews)
        filtered_reviews = self._filter_by_review_length(recent_reviews)

        # Sort by helpful votes and take the top_n reviews
        return self._filter_top_reviews(filtered_reviews)

    def _filter_by_recency(
        self, reviews: list[ProfessorReview]
    ) -> list[ProfessorReview]:
        return list(
            filter(
                lambda review: self.current_year - review.date.year <= self.years_limit,
                reviews,
            )
        )

    def _filter_by_review_length(
        self, reviews: list[ProfessorReview]
    ) -> list[ProfessorReview]:
        return list(
            filter(
                lambda review: len(review.review.split())
                >= self.review_length_threshold,
                reviews,
            )
        )

    def _filter_top_reviews(
        self, reviews: list[ProfessorReview]
    ) -> list[ProfessorReview]:
        # Adjust the top_n based on the number of reviews
        num_reviews = len(reviews)
        top_n = min(self.top_n, num_reviews)  # Cap top_n to available reviews

        return sorted(reviews, key=lambda x: x.helpfulVotes, reverse=True)[:top_n]
