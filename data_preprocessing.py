import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df['text'] = (
            "Film Bilgisi: " + df['title'] + " " +
            "Tür: " + df['type'] + " " +
            "Yönetmen: " + df['director'].fillna('Unknown') + " " +  
            "Oyuncular: " + df['cast'].fillna('Unknown') + " " + 
            "Ülke: " + df['country'].fillna('Unknown') + " " +  
            "Yayınlanma Tarihi: " + df['date_added'].fillna('Unknown') + " " +
            "Yıl: " + df['release_year'].astype(str) + " " +
            "Rating: " + df['rating'].fillna('Unknown') + " " +  
            "Süre: " + df['duration'].fillna('Unknown') + " " + 
            "Kategori: " + df['listed_in'].fillna('Unknown') + " " + 
            "Açıklama: " + df['description'].fillna('No description available')
    )
    return df

file_path = 'netflix_titles.csv'
df = load_data(file_path)
df = preprocess_data(df)
print(df.head())
