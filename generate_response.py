import os
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# .env dosyasını yükle
load_dotenv()

# Hugging Face token'ı .env dosyasından al
hf_token = os.getenv("HF_TOKEN")

def generate_response(prompt, max_tokens=300):
    """
    Meta-Llama 3.1-8B-Instruct modelinden bir yanıt alır.
    """
    # Hugging Face modelini yükle
    model_name = "facebook/bart-large-cnn"

    # Tokenizer ve modeli yükle
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)

    # Pipeline oluştur
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1  # GPU var ise GPU'yu kullan, yoksa CPU'yu kullan
    )

    # Prompt'a göre metin üret
    response = generator(prompt, max_length=max_tokens, num_return_sequences=1)

    # Üretilen metni döndür
    return response[0]['generated_text'].strip()

