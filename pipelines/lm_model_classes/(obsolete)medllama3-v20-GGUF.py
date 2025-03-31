# huggingface-cli download tensorblock/medllama3-v20-GGUF \
#   --include "medllama3-v20-Q2_K.gguf" \
#   --local-dir ./model_weights \
#   --local-dir-use-symlinks False

# above cell downloads the test run model weights. smallest, significant quality loss - not recommended for most purposes

# reference: https://huggingface.co/tensorblock/medllama3-v20-GGUF
# next step: medllama3-v20-Q5_K_S.gguf

from llama_cpp import Llama
import os

class MedLLaMA3GGUF:
    """
    A wrapper for tensorblock/medllama3-v20 GGUF model running via llama-cpp-python.
    """

    def __init__(self, model_path="./model_weights/medllama3-v20-Q2_K.gguf", n_ctx=2048):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from {model_path}...")
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx)

    def _generate(self, prompt: str, max_tokens=256) -> str:
        full_prompt = (
            f"Human: {prompt}<|eot_id|>\n"
            f"Assistant:"
        )
        response = self.llm(full_prompt, max_tokens=max_tokens, stop=["<|eot_id|>"])
        return response["choices"][0]["text"].strip()

    def lexical_simplification(self, term: str) -> str:
        prompt = f"Explain the medical term '{term}' in simple language for a patient."
        return self._generate(prompt)

    def syntactic_simplification(self, text: str) -> str:
        prompt = f"Rewrite the following explanation using shorter, simpler sentences:\n{text}"
        return self._generate(prompt)

    def format_summarization(self, text: str) -> str:
        prompt = f"Summarize and organize the explanation clearly and concisely:\n{text}"
        return self._generate(prompt)

    def simplify(self, term: str) -> dict:
        print("[Stage 1] Lexical Simplification")
        lex = self.lexical_simplification(term)

        print("[Stage 2] Syntactic Simplification")
        synt = self.syntactic_simplification(lex)

        print("[Stage 3] Format Summarization")
        formatted = self.format_summarization(synt)

        return {
            "lexical": lex,
            "syntactic": synt,
            "formatted": formatted,
        }

# Example usage
if __name__ == "__main__":
    model = MedLLaMA3GGUF()
    term = "acute myocardial infarction"
    result = model.lexical_simplification(term)
    print(result)
