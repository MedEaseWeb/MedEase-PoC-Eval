import os
from dotenv import load_dotenv
from dotenv import find_dotenv
import openai
from pathlib import Path
import pprint


load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


def gpt_4o_simplify(prompt, model="gpt-4o", temperature=0.7, max_tokens=800):
    """
    Calls the OpenAI GPT API with a user-defined prompt.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[GPT ERROR] {e}")
        return ""

class GPT4oAPI:
    """
    A GPT-based baseline pipeline with four simplification stages:
    1. Lexical simplification
    2. Syntactic simplification
    3. Format summarization
    """

    def __init__(self, model="gpt-4o"):
        self.model = model

    def lexical_simplification(self, text: str) -> str:
        prompt = (
            f"You are a medical language simplification assistant. Your task is to replace all complex medical jargon "
            f"in the following text with plain, layman-friendly language, without changing the meaning.\n\n"
            f"Text:\n{text}"
        )
        return gpt_4o_simplify(prompt, model=self.model)

    def syntactic_simplification(self, text: str) -> str:
        prompt = (
            f"You are a text simplifier. Break the following medical text into shorter, simpler, and more readable sentences. "
            f"Avoid unnecessary repetition, and keep the meaning intact.\n\n"
            f"Text:\n{text}"
        )
        return gpt_4o_simplify(prompt, model=self.model)

    def format_summarization(self, text: str) -> str:
        prompt = (
            f"You are a summarization assistant. Improve the paragraph structure and logical flow of the following simplified medical text. "
            f"Group related ideas together and ensure the output is clean and easy to read.\n\n"
            f"Text:\n{text}"
        )
        return gpt_4o_simplify(prompt, model=self.model)

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
    pipeline = GPT4oAPI()

    raw_text = (
       "Two trials met the inclusion criteria. One compared 2% ketanserin ointment in polyethylene glycol (PEG) with PEG alone, used twice a day by 40 participants with arterial leg ulcers, for eight weeks or until healing, whichever was sooner. One compared topical application of blood-derived concentrated growth factor (CGF) with standard dressing (polyurethane film or foam); both applied weekly for six weeks by 61 participants with non-healing ulcers (venous, diabetic arterial, neuropathic, traumatic, or vasculitic). Both trials were small, reported results inadequately, and were of low methodological quality. Short follow-up times (six and eight weeks) meant it would be difficult to capture sufficient healing events to allow us to make comparisons between treatments. One trial demonstrated accelerated wound healing in the ketanserin group compared with the control group. In the trial that compared CGF with standard dressings, the number of participants with diabetic arterial ulcers were only reported in the CGF group (9/31), and the number of participants with diabetic arterial ulcers and their data were not reported separately for the standard dressing group. In the CGF group, 66.6% (6/9) of diabetic arterial ulcers showed more than a 50% decrease in ulcer size compared to 6.7% (2/30) of non-healing ulcers treated with standard dressing. We assessed this as very-low certainty evidence due to the small number of studies and arterial ulcer participants, inadequate reporting of methodology and data, and short follow-up period. Only one trial reported side effects (complications), stating that no participant experienced these during follow-up (six weeks, low-certainty evidence). It should also be noted that ketanserin is not licensed in all countries for use in humans. Neither study reported time to ulcer healing, patient satisfaction or quality of life. There is insufficient evidence to determine whether the choice of topical agent or dressing affects the healing of arterial leg ulcers."
    )

    # result = pipeline.simplify(raw_text)


    # print(result["formatted"])
    
    dummyoutput = '''This review includes two studies that examined different treatments for wound healing. The first study focused on an ointment containing 2% ketanserin mixed with polyethylene glycol (PEG) compared to the PEG base alone. In this study, forty participants with artery-related leg wounds applied these treatments twice daily for up to eight weeks or until their wounds healed. Results indicated that wounds healed faster with the ketanserin ointment than with the PEG base alone.

The second study explored the effects of a special blood-based growth factor in comparison to regular wound coverings, such as thin plastic or foam. This study involved sixty-one participants with non-healing wounds caused by various issues, including vein problems, diabetes, nerve disorders, injuries, or blood vessel inflammation. Treatments were applied once a week for six weeks. Within the growth factor group, 9 out of 31 participants had diabetic artery-related wounds, and 6 of these 9 wounds shrank by more than half. In contrast, only 2 out of 30 wounds in the regular dressing group showed similar improvement.

Both studies shared limitations, including small sample sizes, poor design, and insufficient reporting. The follow-up period of six to eight weeks was relatively short, complicating efforts to accurately compare treatment outcomes. Additionally, only one study mentioned side effects, reporting none during the six-week period; however, this evidence is not considered reliable.

Importantly, ketanserin is not approved for human use in all countries, and neither study provided information on healing duration, patient satisfaction, or quality of life. Overall, the evidence is highly uncertain, making it difficult to determine whether the type of ointment or dressing significantly impacts the healing of artery-related leg wounds.'''
        
    pprint.pp(dummyoutput)