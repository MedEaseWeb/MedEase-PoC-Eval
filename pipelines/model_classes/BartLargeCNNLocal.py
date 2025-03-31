from transformers import BartTokenizer, BartForConditionalGeneration

class BartLargeCNNLocal:
    """
    A local BART-large (CNN-trained) pipeline for three-stage simplification:
    1. Lexical simplification
    2. Syntactic simplification
    3. Format summarization
    """

    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.model_name = model_name
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)

    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def lexical_simplification(self, text: str) -> str:
        prompt = (
            f"Simplify the medical jargon in the following text using layman-friendly terms:\n\n{text}"
        )
        return self._generate(prompt)

    def syntactic_simplification(self, text: str) -> str:
        prompt = (
            f"Break the following medical text into shorter, clearer, and simpler sentences:\n\n{text}"
        )
        return self._generate(prompt)

    def format_summarization(self, text: str) -> str:
        prompt = (
            f"Restructure and summarize the following medical text into clean, readable paragraphs:\n\n{text}"
        )
        return self._generate(prompt)

    def simplify(self, text: str) -> dict:
        print("[Stage 1] Lexical Simplification")
        lex = self.lexical_simplification(text)

        print("[Stage 2] Syntactic Simplification")
        synt = self.syntactic_simplification(lex)

        print("[Stage 3] Format Summarization")
        formatted = self.format_summarization(synt)

        return {
            "lexical": lex,
            "syntactic": synt,
            "formatted": formatted,
        }

if __name__ == "__main__":
    model = BartLargeCNNLocal()
    input_text = "The patient experienced a myocardial infarction and was prescribed acetylsalicylic acid."
    result = model.simplify(input_text)
    print(result["formatted"])

'''
Simplify the medical jargon in the following text using layman-friendly terms. The patient experienced a myocardial infarction and was prescribed acetylsalicylic acid. Restructure and summarize the following medical text into clean, readable paragraphs: 
The patient was also given an anti-depressant, and was admitted to the hospital.
'''