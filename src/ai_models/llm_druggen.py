from transformers import AutoTokenizer, AutoModelForCausalLM

class DrugGeneratorLLM:
    def __init__(self, model_name="microsoft/biogpt"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt, n=5):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs, max_new_tokens=64,
            num_return_sequences=n, do_sample=True, temperature=0.8
        )
        decoded = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        smiles = []
        for seq in decoded:
            smiles += [s for s in seq.split(",") if s.strip()]
        return list(set(smiles))
