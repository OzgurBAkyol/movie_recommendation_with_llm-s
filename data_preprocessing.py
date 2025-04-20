import pandas as pd

def load_data(file_path):
    """
    CSV dosyasını yükler ve DataFrame'e dönüştürür.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """
    Veriyi işlemek ve metinleri birleştirmek.
    """
    # Metin sütunlarını birleştirme (title, description, type vb.)
    df['text'] = df['title'] + " " + df['description'] + " " + df['type'] + " " + df['rating']
    return df
