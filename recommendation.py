from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np


def get_recommendations(query, df, embeddings, top_n=5):
    """
    Kullanıcının sorgusuna göre en benzer önerileri döndürür.
    """
    # Kullanıcının sorgusunu vektöre dönüştür
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vector = model.encode([query])

    # Cosine benzerliğini hesapla
    similarities = cosine_similarity(query_vector, embeddings)

    # En yüksek benzerliklere sahip 'top_n' öğelerini seç
    similar_indices = np.argsort(similarities[0])[::-1][:top_n]

    # Sonuçları döndür
    recommendations = df.iloc[similar_indices]
    return recommendations[['title', 'rating', 'description']]
