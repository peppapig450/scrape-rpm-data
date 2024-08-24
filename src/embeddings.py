from typing import Any

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize



class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2"
        )

    def generate_embeddings(self, texts: list[str]):
        """
        Generate sentence embeddings for a list of texts.
        """
        return self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )

    def generate_professor_embedding(
        self,
        tags: list[str],
        review_texts: list[str],
        weight_tags: float = 0.3,
        weight_reviews: float = 0.7
    ):
        
        embedding_dim = self.model.get_sentence_embedding_dimension() 
        if not embedding_dim:
            embedding_dim = 768

        # Generate embeddings for tags
        tags_embedding: NDArray[Any]
        if not tags:
            tags_embedding = np.zeros(embedding_dim)
            weight_reviews = 1.0 # Increase the weight of reviews when no tags are available
            weight_tags = 0.0 # And set the weight of tags to 0
        else:
            tags_embeddings = self.generate_embeddings(tags)
            tags_embedding = np.mean(tags_embeddings, axis=0)
        
        # Normalize tag embeddings
        tags_embedding = normalize(tags_embedding.reshape(1, -1))[0] #type:ignore


        # Sentence embeddings
        review_embeddings = self.generate_embeddings(review_texts)
        review_embedding: NDArray[Any] = np.mean(review_embeddings, axis=0)
        
        # Normalize review embedding
        reviews_embedding: NDArray[Any] = normalize(review_embedding.reshape(1, -1))[0] #type: ignore
        
        # Apply weights and combine
        weighted_tags = tags_embedding * weight_tags
        weighted_reviews = reviews_embedding * weight_reviews
        combined_embedding = np.concatenate([weighted_tags, weighted_reviews])
        
        # Normalize the final combined embedding
        normalized_embedding = normalize(combined_embedding.reshape(1, -1))[0]

        return normalized_embedding
