from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import json
from datetime import datetime
import os

# Embedding modeli ve tokenizer
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

# Cevap üretim modeli ve tokenizer
generation_model_name = "microsoft/DialoGPT-medium"
generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
generation_model = AutoModelForCausalLM.from_pretrained(generation_model_name)

# Pad token kontrolü
if generation_tokenizer.pad_token is None:
    generation_tokenizer.pad_token = generation_tokenizer.eos_token

# Cihaz seçimi
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model.to(device)
generation_model.to(device)

# Log dosyası
LOG_FILE = "user_query_logs.json"

# Kullanıcı sorgularını loglama
def log_user_query(query, recommendations, response_text):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "recommendations": [r[0] for r in recommendations],
        "response": response_text
    }
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(log_entry)

    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)

# Sohbet geçmişini yükleme
def load_chat_history(n=3):
    if not os.path.exists(LOG_FILE):
        return ""
    with open(LOG_FILE, "r") as f:
        data = json.load(f)[-n:]  # Son n kaydı al
    history = ""
    for entry in data:
        history += f"Kullanıcı: {entry['query']}\nSistem: {entry['response']}\n"
    return history.strip()

# Few-shot örnekleri
FEW_SHOT_EXAMPLES = """
Kullanıcı: Aksiyon ve bilim kurgu karışımı bir film arıyorum.
Sistem: İşte tam sana göre içerikler:
1. The Matrix: Sanal gerçeklik dünyasında geçen, felsefi ve aksiyon dolu bir film. 
Bu filmler zihin açıcı ve sürükleyici, keyifli seyirler!

Kullanıcı: Romantik ama komik bir film önerir misin?
Sistem: Elbette! Aşağıdaki içerikler tam senlik:
1. Crazy Rich Asians: Lüks ve aşk dolu bir yolculuk. 
2. The Proposal: Patron ve çalışan arasında sürprizlerle dolu bir evlilik planı. 
Gülümseten bir aşk hikayesi arıyorsan, bu filmler birebir.
"""

# Öneri sistemi
def get_recommendations(user_query, df, embeddings, top_k=5):
    # Kullanıcı sorgusunu embedding'e dönüştürme
    inputs = embedding_tokenizer(user_query, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1)

    # Embedding'ler ile benzerlik hesaplama
    embeddings_tensor = torch.tensor(embeddings).to(device)
    cos_scores = torch.cosine_similarity(query_embedding, embeddings_tensor, dim=1)
    top_results = torch.topk(cos_scores, k=top_k)

    # En iyi sonuçları toplama
    recommendations = []
    for score, idx in zip(top_results.values, top_results.indices):
        idx = idx.item()
        title = df.iloc[idx]["title"]
        desc = df.iloc[idx]["description"]
        fake_link = f"https://netflix.com/watch/{idx}"
        recommendations.append((title, desc, score.item(), fake_link))

    # Önerileri birleştirme
    joined_recommendations = "\n".join([
        f"{i+1}. {title}: {desc} (📺 {link})"
        for i, (title, desc, _, link) in enumerate(recommendations)
    ])

    # Prompt oluşturma (sadece mevcut sorgu için)
    prompt = f"Kullanıcı: {user_query}\nSistem: İşte önerilerim:\n{joined_recommendations}\n"

    # Cevap üretimi
    gen_inputs = generation_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    gen_inputs = {key: value.to(device) for key, value in gen_inputs.items()}
    with torch.no_grad():
        generated_ids = generation_model.generate(
            gen_inputs["input_ids"],
            max_new_tokens=50,
            num_return_sequences=1,
            pad_token_id=generation_tokenizer.pad_token_id,
            attention_mask=gen_inputs["attention_mask"]
        )

    response_text = generation_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Kullanıcı sorgusunu loglama
    log_user_query(user_query, recommendations, response_text)

    # Sadece kullanıcı sorgusu ve model cevabını döndürme
    return f"Kullanıcı: {user_query}\nSistem: {response_text}"