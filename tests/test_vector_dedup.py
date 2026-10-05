import numpy as np
from app.services.vector_dedup import get_embedding, cosine_similarity

def test_embedding_generation():
    vec1 = get_embedding("What is the primary function of photosynthesis?")
    assert isinstance(vec1, np.ndarray)
    assert len(vec1) > 0

def test_cosine_similarity_identical():
    v1 = get_embedding("What is photosynthesis?")
    v2 = get_embedding("What is photosynthesis?")
    sim = cosine_similarity(v1, v2)
    assert round(sim, 2) == 1.0

def test_semantic_similarity_paraphrased():
    v1 = get_embedding("How do plants convert sunlight into energy?")
    v2 = get_embedding("What process do plants use to turn sunlight into food?")
    sim = cosine_similarity(v1, v2)
    # Paraphrased questions should have high cosine similarity
    assert sim > 0.50

if __name__ == "__main__":
    test_embedding_generation()
    test_cosine_similarity_identical()
    test_semantic_similarity_paraphrased()
    print("ALL VECTOR DEDUP TESTS PASSED SUCCESSFULLY!")
