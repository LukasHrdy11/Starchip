import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

# --- 1. Konfigurace ---
BASE_MODEL_PATH = "/home/luk-hrd/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/2dcff66eac0c01dc50e4c41eea959968232187fe"
# Použijte správnou cestu k vašim finálním adaptérům
ADAPTER_PATH = "./mistral-7b-quantum-instruct-v1/final_adapter" # Ujistěte se, že cesta je správná

# --- 2. Načtení modelu a tokenizeru ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Načítám základní model z: {BASE_MODEL_PATH}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Načítám LoRA adaptéry z: {ADAPTER_PATH}")
# Načtení checkpointu, pokud existuje (nahraďte cestou k nejlepšímu checkpointu, pokud chcete)
# Např. ADAPTER_PATH = "./mistral-7b-quantum-instruct-v1/checkpoint-100"
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

model.config.use_cache = True
model = model.eval()

# --- 3. Generování odpovědi ---
question = "What is a qubit and how does it differ from a classical bit?"
prompt = f"""
[INST] You are an expert in 'quantum computing', specifically focusing on 'a specific topic'. Provide a detailed, step-by-step solution to the following problem:

### Problem:
{question} [/INST]
"""

inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cuda")

# <<< --- ZAČÁTEK DEBUG SEKCE --- >>>
print("\n--- DEBUG INFORMACE ---")
input_token_count = inputs['input_ids'].shape[1]
print(f"Počet tokenů v promptu: {input_token_count}")
print(f"ID 'end-of-sequence' tokenu (EOS): {tokenizer.eos_token_id}")
# <<< --- KONEC DEBUG SEKCE --- >>>

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

# <<< --- ZAČÁTEK DEBUG SEKCE 2 --- >>>
print("\n--- DEBUG INFORMACE 2 ---")
output_token_count = outputs[0].shape[0]
print(f"Celkový počet tokenů ve výstupu: {output_token_count}")
print(f"Počet nově vygenerovaných tokenů: {output_token_count - input_token_count}")

# Dekódujeme CELÝ výstup, abychom viděli, co model vrátil
full_decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
print("\n--- KOMPLETNÍ DEKÓDOVANÝ VÝSTUP (včetně speciálních tokenů) ---")
print(repr(full_decoded_output)) # Používáme repr() pro zobrazení speciálních znaků jako \n nebo </s>
# <<< --- KONEC DEBUG SEKCE 2 --- >>>

# Původní kód pro získání odpovědi
response_ids = outputs[0][inputs['input_ids'].shape[1]:]
response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

print("\n\n--- Otázka ---")
print(question)
print("\n--- Finální odpověď modelu ---")
print(response_text if response_text else "[ODPOVĚĎ JE PRÁZDNÁ]")
