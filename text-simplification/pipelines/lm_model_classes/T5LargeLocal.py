from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class T5LargeLocal:
    """
    A local T5-small-based pipeline with three simplification stages:
    1. Lexical simplification
    2. Syntactic simplification
    3. Format summarization
    """

    def __init__(self, model_name="t5-large", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _generate(self, prompt, max_new_tokens=256):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def lexical_simplification(self, text: str) -> str:
        prompt = f"simplify the vocabulary in this medical text: {text}"
        return self._generate(prompt)

    def syntactic_simplification(self, text: str) -> str:
        prompt = f"simplify the sentence structure of this text: {text}"
        return self._generate(prompt)

    def format_summarization(self, text: str) -> str:
        prompt = f"improve formatting and clarity of the following: {text}"
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
    model = T5LargeLocal()
    raw_text = (
       "summarize: Two trials met the inclusion criteria. One compared 2% ketanserin ointment in polyethylene glycol (PEG) with PEG alone, used twice a day by 40 participants with arterial leg ulcers, for eight weeks or until healing, whichever was sooner. One compared topical application of blood-derived concentrated growth factor (CGF) with standard dressing (polyurethane film or foam); both applied weekly for six weeks by 61 participants with non-healing ulcers (venous, diabetic arterial, neuropathic, traumatic, or vasculitic). Both trials were small, reported results inadequately, and were of low methodological quality. Short follow-up times (six and eight weeks) meant it would be difficult to capture sufficient healing events to allow us to make comparisons between treatments. One trial demonstrated accelerated wound healing in the ketanserin group compared with the control group. In the trial that compared CGF with standard dressings, the number of participants with diabetic arterial ulcers were only reported in the CGF group (9/31), and the number of participants with diabetic arterial ulcers and their data were not reported separately for the standard dressing group. In the CGF group, 66.6% (6/9) of diabetic arterial ulcers showed more than a 50% decrease in ulcer size compared to 6.7% (2/30) of non-healing ulcers treated with standard dressing. We assessed this as very-low certainty evidence due to the small number of studies and arterial ulcer participants, inadequate reporting of methodology and data, and short follow-up period. Only one trial reported side effects (complications), stating that no participant experienced these during follow-up (six weeks, low-certainty evidence). It should also be noted that ketanserin is not licensed in all countries for use in humans. Neither study reported time to ulcer healing, patient satisfaction or quality of life. There is insufficient evidence to determine whether the choice of topical agent or dressing affects the healing of arterial leg ulcers."
    )
    # result = model.simplify(raw_text)
    # print(result)
    
    # dummy_output = {'lexical': 'ketanserin ointment in polyethylene glycol (PEG) used twice a day by 40 participants . one compared topical application of blood-derived concentrated growth factor (CGF) . in the CGF group, 66.6% (6/9) of diabetic arterial ulcers showed more than a 50% decrease in ulcer size .', 'syntactic': '. To . ketanserin ointment in polyethylene glycol (PEG) used twice a day by 40 participants .. ketanserin ointment in. and the sentence structure of this text: the sentence structure of this text:: . ...::.. . .  . . to     ', 'formatted': '. ketanserin ointment in polyethylene glycol (PEG) used twice a day by 40 participants .. ketanserin ointment in. and the sentence structure of this text: . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . '}
    print(model.syntactic_simplification("hello world this is yuxuan shi, your favorite friend"))