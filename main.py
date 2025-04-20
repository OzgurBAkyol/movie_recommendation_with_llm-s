from data_preprocessing import load_data, preprocess_data
from vectorization import vectorize_text
from recommendation import get_recommendations
from utils import save_embeddings, load_embeddings
from generate_response import generate_response  # Yeni modeli ve fonksiyonu ekledik
import os

def main():
    # Veri yükleme ve işleme
    df = load_data('netflix_titles.csv')
    df = preprocess_data(df)

    # Eğer embedding dosyası varsa yükle, yoksa hesapla ve kaydet
    embedding_path = 'embeddings.pkl'
    if os.path.exists(embedding_path):
        embeddings = load_embeddings(embedding_path)
    else:
        embeddings = vectorize_text(df['text'].tolist())
        save_embeddings(embeddings, embedding_path)

    # Kullanıcıdan gelen sorgu
    user_query = input("Hangi türde bir film izlemek istersiniz? ")

    # LLM ile sohbet havasında öneri al
    conversational_response = get_recommendations(user_query, df, embeddings)

    # Sohbet yanıtını göster
    print("\n🎬 Tavsiye Asistanı:\n")
    print(conversational_response)

if __name__ == "__main__":
    main()



#from transformers import pipeline
#model = pipeline(task = "text-generation",model="facebook/bart-large-cnn")
#response = model("Netflix'te izlemek için bir film önerisi ver.", max_length=50)
#print(response)