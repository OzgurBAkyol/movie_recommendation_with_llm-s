from sentence_transformers import SentenceTransformer

def vectorize_text(text_list):
    """
    Verilen metinleri vektöre dönüştürür.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Llama3-mini veya başka bir küçük model de kullanılabilir
    embeddings = model.encode(text_list)
    return embeddings
