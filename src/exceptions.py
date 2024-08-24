class ScrapeException(Exception):
    """Base class for all scraping-related exceptions."""

    pass

class EmbeddingException(Exception):
    """Base class for all embedding generation related exceptions."""

class NetworkException(ScrapeException):
    """Raised for network-related errors."""

    pass


class ParsingException(ScrapeException):
    """Raised for errors in parsing or data extraction."""

    pass


class DataException(ScrapeException):
    """Raised for errors in data processing or validation."""

    pass

class EmbeddingsDimensionException(Exception):
    """Raised when the embedding dimension isn't found"""