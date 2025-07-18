# molecule_generator.py
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_cancer_drug_candidates(prompt: str, model='microsoft/biogpt', num=5):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=64, 
        num_return_sequences=num, 
        do_sample=True, temperature=0.8
    )
    candidates = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    smiles = [s.strip() for c in candidates for s in c.split(',') if s.strip()]
    return list(set(smiles))
