import pickle

def save_embeddings(embeddings, file_path):
    """
    Vektörleri bir dosyaya kaydeder.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(file_path):
    """
    Vektörleri bir dosyadan yükler.
    """
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings
