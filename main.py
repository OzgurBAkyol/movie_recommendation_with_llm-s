from data_preprocessing import load_data, preprocess_data
from vectorization import vectorize_text
from recommendation import get_recommendations
from utils import save_embeddings, load_embeddings
import os

EMBEDDING_PATH = "embeddings.pkl"
DATA_PATH = "netflix_titles.csv"

def load_or_generate_embeddings(df, path=EMBEDDING_PATH):
    if os.path.exists(path):
        print("🔁 Kayıtlı embedding dosyası bulundu, yükleniyor...")
        return load_embeddings(path)
    print("🧠 Embedding'ler oluşturuluyor...")
    embeddings = vectorize_text(df['text'].tolist())
    save_embeddings(embeddings, path)
    return embeddings

def run_pipeline():
    print("📄 Veri yükleniyor...")
    df = load_data(DATA_PATH)

    print("🧼 Veri ön işleniyor...")
    df = preprocess_data(df)

    embeddings = load_or_generate_embeddings(df)

    user_query = input("\n🎯 Nasıl bir içerik arıyorsunuz? (Örnek: bilim kurgu dizisi, İspanyol yapımı romantik film...)\n\n> ")

    print("\n🎬 Tavsiye Asistanı:\n")
    print(get_recommendations(user_query, df, embeddings))

if __name__ == "__main__":
    run_pipeline()
