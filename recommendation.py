from sentence_transformers import SentenceTransformer, util
import numpy as np
from generate_response import generate_response  # LLM cevap üretimi için

def get_recommendations(user_query, df, embeddings, top_n=5):
    """
    Kullanıcının sorgusuna göre embedding benzerliğini kullanarak önerilen filmleri alır.
    Daha sonra bu önerileri bir dil modeline vererek kişisel ve sohbet havasında yanıt döndürür.
    """
    # Sorguyu embed et
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(user_query, convert_to_tensor=True)

    # Benzerlikleri hesapla
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = cos_scores.argsort(descending=True)[:top_n]

    # En benzer filmleri al
    recommended = df.iloc[top_results]

    # Prompt oluştur
    prompt = f"Kullanıcının film isteği: \"{user_query}\"\n"
    prompt += "Aşağıdaki filmleri öneriyoruz:\n\n"
    for idx, row in recommended.iterrows():
        prompt += f"- {row['title']}: {row['description']}\n"

    prompt += "\nLütfen bu filmleri sohbet havasında, kişisel bir dille kullanıcıya öner.\n"

    # LLM ile cevap oluştur
    conversational_output = generate_response(prompt)

    return conversational_output

