import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    features = {
        "Film Bilgisi": "title",
        "Tür": "type",
        "Yönetmen": "director",
        "Oyuncular": "cast",
        "Ülke": "country",
        "Yayınlanma Tarihi": "date_added",
        "Yıl": "release_year",
        "Rating": "rating",
        "Süre": "duration",
        "Kategori": "listed_in",
        "Açıklama": "description"
    }

    df.fillna("Unknown", inplace=True)
    df['text'] = df.apply(lambda row: " ".join([
        f"{k}: {str(row[v])}" for k, v in features.items()
    ]), axis=1)

    return df

file_path = 'netflix_titles.csv'
df = load_data(file_path)
df = preprocess_data(df)
print(df.head())



