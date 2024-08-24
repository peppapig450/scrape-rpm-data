from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer


class EmbeddingService:
    def __init__(self):
        # Initialize SentenceTransformer for sentence embeddings
        self.sentence_model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2"
        )

        # Initialize BERT model for token embeddings
        self.token_model = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def generate_sentence_embeddings(self, texts: list[str]):
        """
        Generate sentence embeddings for a list of texts.
        """
        return self.sentence_model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )

    def generate_token_embeddings(self, texts: list[str]):
        """
        Generate token embeddings for a list of texts.
        """
        embeddings: list[Any] = []
        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self.token_model(**inputs)
            # Use the mean of the last hidden states as the token embedding
            embedding = outputs.last_hidden_state.mean(1).squeeze().numpy()
            embeddings.append(embedding)
        return np.array(embeddings)

    def generate_embeddings(
        self,
        name: str,
        department: str,
        university: str,
        tags: list[str],
        review_texts: list[str],
    ):
        """
        Generate token embeddings for name, department, university, and tags.
        Generate sentence embeddings for reviews.
        """
        # Token embeddings
        name_embedding = self.generate_token_embeddings([name])[0]
        department_embedding = self.generate_token_embeddings([department])[0]
        university_embedding = self.generate_token_embeddings([university])[0]
        tags_embeddings = self.generate_token_embeddings(tags)
        tags_embedding = np.mean(tags_embeddings, axis=0)  # Average tags embedding

        # Sentence embeddings
        review_embeddings = self.generate_sentence_embeddings(review_texts)
        avg_review_embedding = np.mean(review_embeddings, axis=0)

        # Combine all embeddings
        combine_embeddings = np.concatenate(
            [
                name_embedding,
                department_embedding,
                university_embedding,
                tags_embedding,
                avg_review_embedding,
            ]
        )

        return combine_embeddings
