import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df['text'] = (
            "Film Bilgisi: " + df['title'] + " " +
            "Tür: " + df['type'] + " " +
            "Yönetmen: " + df['director'].fillna('Unknown') + " " +  # Eksik değerler için 'Unknown' kullanıyoruz
            "Oyuncular: " + df['cast'].fillna('Unknown') + " " +  # Eksik değerler için 'Unknown' kullanıyoruz
            "Ülke: " + df['country'].fillna('Unknown') + " " +  # Eksik değerler için 'Unknown' kullanıyoruz
            "Yayınlanma Tarihi: " + df['date_added'].fillna('Unknown') + " " +
            "Yıl: " + df['release_year'].astype(str) + " " +
            "Rating: " + df['rating'].fillna('Unknown') + " " +  # Eksik değerler için 'Unknown' kullanıyoruz
            "Süre: " + df['duration'].fillna('Unknown') + " " +  # Eksik değerler için 'Unknown' kullanıyoruz
            "Kategori: " + df['listed_in'].fillna('Unknown') + " " +  # Eksik değerler için 'Unknown' kullanıyoruz
            "Açıklama: " + df['description'].fillna('No description available')
    )
    return df

file_path = 'netflix_titles.csv'
df = load_data(file_path)
df = preprocess_data(df)


