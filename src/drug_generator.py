from transformers import AutoTokenizer, AutoModelForCausalLM

class DrugGenerator:
    def __init__(self, model_name="microsoft/biogpt"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_candidates(self, prompt: str, num_return_sequences=5) -> list[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs, max_new_tokens=64, 
            num_return_sequences=num_return_sequences,
            do_sample=True, temperature=0.8
        )
        sequences = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        smiles = []
        for seq in sequences:
            smiles += [s.strip() for s in seq.split(",") if s.strip()]
        return list(set(smiles))
