import os
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

def generate_response(prompt, max_tokens=300):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1  
    )

    response = generator(prompt, max_length=max_tokens, num_return_sequences=1)
    return response[0]['generated_text'].strip()

