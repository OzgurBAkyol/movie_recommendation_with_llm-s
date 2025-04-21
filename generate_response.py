import os
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

def generate_response(prompt, model_name="sentence-transformers/all-MiniLM-L6-v2", max_tokens=300, device=None):
    if not hf_token:
        raise ValueError("HF_TOKEN çevre değişkeni tanımlı değil. Lütfen .env dosyasını kontrol edin.")

    device = device if device is not None else (0 if torch.cuda.is_available() else -1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
    except Exception as e:
        raise RuntimeError(f"Model veya tokenizer yüklenirken bir hata oluştu: {e}")

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    response = generator(prompt, max_length=max_tokens, num_return_sequences=1)
    return response[0]['generated_text'].strip()