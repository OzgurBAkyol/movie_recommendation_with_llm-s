from sentence_transformers import SentenceTransformer

def vectorize_text(text_list):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_list)
    return embeddings
