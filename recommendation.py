from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Llama modelini yükleyin
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Cihaz seçimi (GPU var mı kontrol et)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def get_recommendations(user_query, df, embeddings):
    # Kullanıcı sorgusunu token'lara ayır
    inputs = tokenizer(user_query, return_tensors="pt").to(device)

    # Modelin çıktısını al
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        logits = outputs.logits

    # Burada embedding'leri almak için modelin son katmanlarından çıkan logits'i kullanıyoruz
    # Eğer daha derin bir embedding isterseniz, bir ara katmandan alabilirsiniz.
    query_embedding = logits.mean(dim=1)  # Output'un ortalamasını alarak bir embedding elde ediyoruz.

    # Cosine benzerliğini hesapla (query ve embeddings tensor'lerinin aynı cihazda olması gerekiyor)
    cos_scores = torch.cosine_similarity(query_embedding, embeddings, dim=1)

    # En yüksek benzerlik skorlarına sahip filmleri seç
    top_results = torch.topk(cos_scores, k=5)

    # En yüksek skora sahip 5 öneriyi döndür
    recommendations = []
    for score, idx in zip(top_results[0], top_results[1]):
        recommendations.append({
            "title": df.iloc[idx]["title"],
            "score": score.item()
        })

    return recommendations
