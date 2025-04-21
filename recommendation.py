from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

generation_model_name = "microsoft/DialoGPT-medium"
generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
generation_model = AutoModelForCausalLM.from_pretrained(generation_model_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model.to(device)
generation_model.to(device)

def get_recommendations(user_query, df, embeddings, top_k=5):
    inputs = embedding_tokenizer(user_query, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        query_embedding = outputs.last_hidden_state.mean(dim=1)  # mean pooling

    embeddings_tensor = torch.tensor(embeddings).to(device)
    cos_scores = torch.cosine_similarity(query_embedding, embeddings_tensor, dim=1)
    top_results = torch.topk(cos_scores, k=top_k)

    recommendations = []
    for score, idx in zip(top_results.values, top_results.indices):
        idx = idx.item()
        title = df.iloc[idx]["title"]
        desc = df.iloc[idx]["description"]
        recommendations.append((title, desc, score.item()))

    joined_recommendations = "\n".join([f"{i+1}. {title}: {desc}" for i, (title, desc, _) in enumerate(recommendations)])
    prompt = f"Kullanıcı şunu sordu: '{user_query}'. Buna karşılık aşağıdaki içerikleri öneriyorsun:\n{joined_recommendations}\n\nBu önerileri doğal ve sempatik bir dille açıkla."

    gen_inputs = generation_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        generated_ids = generation_model.generate(
            gen_inputs["input_ids"],
            max_length=300,
            num_return_sequences=1,
            pad_token_id=generation_tokenizer.eos_token_id
        )

    generated_text = generation_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text
