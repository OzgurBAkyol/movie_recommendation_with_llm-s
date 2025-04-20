from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# Token'ı .env dosyasından al
token = os.getenv('HF_TOKEN')

# Model ismi
MODEL_NAME = "microsoft/phi-2"

# Model ve tokenizer yükleniyor
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",  # otomatik olarak CPU/GPU seçimi yapar
    use_auth_token=token  # token parametresi ile yetkilendirme yapılır
)

# Pipeline tanımı
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

def generate_response(prompt, max_tokens=300):
    """
    Prompt'a göre dil modeliyle sohbet havasında cevap üretir.
    """
    response = generator(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        num_return_sequences=1
    )

    # Çıktıdan prompt'u ayırıp sadece cevabı döndürüyoruz
    generated_text = response[0]['generated_text']
    answer = generated_text[len(prompt):].strip()

    return answer
