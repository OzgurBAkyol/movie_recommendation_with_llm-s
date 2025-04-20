# recommendation.py

from transformers import AutoTokenizer, AutoModel
import torch

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


def get_recommendations(user_query, df, embeddings):
    inputs = tokenizer(user_query, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1)  # Cümle embedding'ini almak için mean pooling yapıyoruz

    embeddings = torch.tensor(embeddings).to(device)
    cos_scores = torch.cosine_similarity(query_embedding, embeddings, dim=1)
    top_results = torch.topk(cos_scores, k=5)

    recommendations = []
    for score, idx in zip(top_results[0], top_results[1]):
        idx = idx.item()  # Tensor'ü integer'a çevir
        movie_title = df.iloc[idx]["title"]
        movie_description = df.iloc[idx]["description"]  # Film açıklaması
        recommendations.append({
            "title": movie_title,
            "score": score.item(),
            "description": movie_description
        })

    response = "🎬 Tavsiye Asistanı:\n\n"
    for rec in recommendations:
        response += f"• {rec['title']}: {rec['description']} (Benzerlik skoru: {rec['score']:.2f})\n\n"

    return response
