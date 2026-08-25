import re
content = open('rag_engine.py', 'r', encoding='utf-8').read()

new_load_llm = """    def load_llm(self):
        if self.generator:
            return

        print(f"[*] Loading EXAONE LLM: {self.llm_model_name}...")
        from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
        import torch
        from transformers import pipeline

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        print("[*] Using 4-bit quantization (BitsAndBytes NF4) for VRAM optimization.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.llm_model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
        self.model.eval()

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1
        )
        print("[+] EXAONE LLM loaded successfully.")
"""
content = re.sub(r'    def load_llm\(self\):.*?(?=\n    def retrieve)', new_load_llm, content, flags=re.DOTALL)

with open('rag_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)
