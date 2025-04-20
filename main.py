from data_preprocessing import load_data, preprocess_data
from vectorization import vectorize_text
from recommendation import get_recommendations
from utils import save_embeddings, load_embeddings


def main():
    # Veri yükleme ve işleme
    df = load_data('netflix_data.csv')
    df = preprocess_data(df)

    # Vektörleştirme işlemi (ilk çalıştırma için)
    embeddings = vectorize_text(df['text'].tolist())

    # Vektörleri kaydet
    save_embeddings(embeddings, 'embeddings.pkl')

    # Kullanıcıdan gelen sorgu
    user_query = input("Hangi türde bir film izlemek istersiniz? ")

    # Kaydedilmiş vektörleri yükle (ileride çalıştırıldığında)
    embeddings = load_embeddings('embeddings.pkl')

    # Öneri alma
    recommendations = get_recommendations(user_query, df, embeddings)

    # Sonuçları gösterme
    print("Önerilen Filmler/Diziler:")
    print(recommendations)


if __name__ == "__main__":
    main()
