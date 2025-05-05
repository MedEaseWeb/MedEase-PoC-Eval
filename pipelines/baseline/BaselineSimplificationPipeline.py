import os
from dotenv import load_dotenv
from dotenv import find_dotenv
from openai import OpenAI
import openai
from pathlib import Path
import pprint


load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def gpt_simplify(prompt, model="gpt-4o", temperature=0.7, max_tokens=800):
    """
    Calls the OpenAI GPT API with a user-defined prompt using the new >=1.0.0 API.
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[GPT ERROR] {e}")
        return ""


class BaselineSimplificationPipeline:
    """
    A GPT-based baseline pipeline with four simplification stages:
    1. Lexical simplification
    2. Syntactic simplification
    3. Format summarization
    4. Dynamic summarization
    """

    def __init__(self, model="gpt-4o"):
        self.model = model

    def lexical_simplification(self, text: str) -> str:
        prompt = (
            f"You are a medical language simplification assistant. Your task is to replace all complex medical jargon "
            f"in the following text with plain, layman-friendly language, without changing the meaning.\n\n"
            f"Text:\n{text}"
        )
        return gpt_simplify(prompt, model=self.model)

    def syntactic_simplification(self, text: str) -> str:
        prompt = (
            f"You are a text simplifier. Break the following medical text into shorter, simpler, and more readable sentences. "
            f"Avoid unnecessary repetition, and keep the meaning intact.\n\n"
            f"Text:\n{text}"
        )
        return gpt_simplify(prompt, model=self.model)

    def format_summarization(self, text: str) -> str:
        prompt = (
            f"You are a summarization assistant. Improve the paragraph structure and logical flow of the following simplified medical text. "
            f"Group related ideas together and ensure the output is clean and easy to read.\n\n"
            f"Text:\n{text}"
        )
        return gpt_simplify(prompt, model=self.model)

    def dynamic_summarization(self, text: str) -> dict:
        prompt = (
            f"You are an intelligent assistant for patients. Read the medical summary below and extract key sections such as "
            f"'Diagnosis', 'Treatment', 'Next Steps', and 'Other Information'. Return the result as a JSON object with clear labels.\n\n"
            f"Text:\n{text}"
        )
        response = gpt_simplify(prompt, model=self.model)
        try:
            import json
            return json.loads(response)
        except json.JSONDecodeError:
            return {"raw_output": response, "error": "Could not parse JSON"}

    def simplify(self, text: str) -> dict:
        print("[Stage 1] Lexical Simplification")
        lex = self.lexical_simplification(text)

        print("[Stage 2] Syntactic Simplification")
        synt = self.syntactic_simplification(lex)

        print("[Stage 3] Format Summarization")
        formatted = self.format_summarization(synt)

        print("[Stage 4] Dynamic Summarization")
        dynamic = self.dynamic_summarization(formatted)

        return {
            "lexical": lex,
            "syntactic": synt,
            "formatted": formatted,
            "final_output": dynamic
        }
        
if __name__ == "__main__":
    pipeline = BaselineSimplificationPipeline()

    raw_text = (
       "Two trials met the inclusion criteria. One compared 2% ketanserin ointment in polyethylene glycol (PEG) with PEG alone, used twice a day by 40 participants with arterial leg ulcers, for eight weeks or until healing, whichever was sooner. One compared topical application of blood-derived concentrated growth factor (CGF) with standard dressing (polyurethane film or foam); both applied weekly for six weeks by 61 participants with non-healing ulcers (venous, diabetic arterial, neuropathic, traumatic, or vasculitic). Both trials were small, reported results inadequately, and were of low methodological quality. Short follow-up times (six and eight weeks) meant it would be difficult to capture sufficient healing events to allow us to make comparisons between treatments. One trial demonstrated accelerated wound healing in the ketanserin group compared with the control group. In the trial that compared CGF with standard dressings, the number of participants with diabetic arterial ulcers were only reported in the CGF group (9/31), and the number of participants with diabetic arterial ulcers and their data were not reported separately for the standard dressing group. In the CGF group, 66.6% (6/9) of diabetic arterial ulcers showed more than a 50% decrease in ulcer size compared to 6.7% (2/30) of non-healing ulcers treated with standard dressing. We assessed this as very-low certainty evidence due to the small number of studies and arterial ulcer participants, inadequate reporting of methodology and data, and short follow-up period. Only one trial reported side effects (complications), stating that no participant experienced these during follow-up (six weeks, low-certainty evidence). It should also be noted that ketanserin is not licensed in all countries for use in humans. Neither study reported time to ulcer healing, patient satisfaction or quality of life. There is insufficient evidence to determine whether the choice of topical agent or dressing affects the healing of arterial leg ulcers."
    )

    result = pipeline.simplify(raw_text)

    print("\nFinal Dynamic Output:")
    print(result["final_output"])
    
    dummyoutput = {'Diagnosis': 'Leg ulcers', 'Treatment': 'Special ointment used twice daily for eight weeks in one study, growth factor used once a week for six weeks in another study', 'Next Steps': 'Due to limitations in the studies, including short follow-up periods and lack of detailed results, further research is needed to determine the effectiveness of the treatments', 'Other Information': {'Key Points': ['One study reported faster wound healing with the ointment, while the other study showed shrinkage of ulcers with the growth factor', 'Ketanserin, a medication used in one of the studies, is not approved in all countries', 'Important factors such as healing time, patient satisfaction, and quality of life were not discussed in either study', 'Reliability of results is questionable due to poor quality of studies and lack of detailed information'], 'Complications': 'Only one study mentioned the absence of complications during the follow-up period'}}
    
    pprint.pp(dummyoutput)