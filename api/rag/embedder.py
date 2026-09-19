from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(model_name: str, texts: list[str]) -> list[list[float]]:
    model = get_embedding_model(model_name)
    vectors = model.encode(texts, show_progress_bar=False)
    return vectors.tolist()
