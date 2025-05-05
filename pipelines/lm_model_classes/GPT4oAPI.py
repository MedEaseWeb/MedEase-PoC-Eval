import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import pprint
from openai import OpenAI  # ✅ THIS is new in SDK 1.0+
load_dotenv(find_dotenv())

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def gpt_4o_simplify(messages: list, model="gpt-4o", temperature=0.0, max_tokens=500):
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
        """
        Lexical simplification with full prompt engineering.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical language simplification assistant. Your task is to rewrite complex medical sentences "
                    "using simpler vocabulary without changing the original meaning. Do not explain or remove information—only "
                    "replace terms with simpler equivalents. Break up long sentences where needed. Return plain text only."
                )
            },
            {
                "role": "user",
                "content": (
                    "Original: The patient presented with dyspnea and required supplemental oxygen.\n"
                    "Simplified: The patient had trouble breathing and needed extra oxygen.\n\n"
                    "Original: Administer acetaminophen PRN for febrile episodes exceeding 38°C.\n"
                    "Simplified: Give acetaminophen when needed if the fever goes above 38°C.\n\n"
                    "Original: Hypertension was managed conservatively without pharmacological intervention.\n"
                    "Simplified: High blood pressure was controlled without using medicine.\n\n"
                    "Original: A colonoscopy was recommended to rule out neoplastic changes.\n"
                    "Simplified: A colonoscopy was suggested to check for signs of cancer.\n\n"
                    f"Original: {text}\nSimplified:"
                )
            }
        ]
        return gpt_4o_simplify(messages, model=self.model)

    def syntactic_simplification(self, text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a syntactic simplifier specializing in medical text. Your task is to break down long or complex sentences "
                    "into multiple shorter, simpler sentences while keeping the meaning exactly the same. "
                    "Do not remove or add any information. Do not summarize or simplify vocabulary. Only modify sentence structure for clarity.\n\n"
                    "Output plain text only. No bullet points, no markdown, no extra formatting."
                )
            },
            {
                "role": "user",
                "content": (
                    "Original: The patient was admitted for chest pain, which started three hours prior and was accompanied by shortness of breath and sweating.\n"
                    "Simplified: The patient was admitted for chest pain. The pain started three hours earlier. It was accompanied by shortness of breath and sweating.\n\n"
                    
                    "Original: Follow-up imaging was performed to evaluate the effectiveness of the prescribed antibiotics in resolving the patient's pneumonia.\n"
                    "Simplified: Follow-up imaging was performed. It was done to evaluate whether the antibiotics were helping to resolve the patient’s pneumonia.\n\n"
                    
                    "Original: The patient denied experiencing any nausea, vomiting, or changes in bowel habits but reported increased fatigue and occasional dizziness.\n"
                    "Simplified: The patient did not experience nausea, vomiting, or changes in bowel habits. However, they reported feeling more tired. They also experienced occasional dizziness.\n\n"

                    f"Original: {text}\nSimplified:"
                )
            }
        ]
        return gpt_4o_simplify(messages, model=self.model)
    
   



    def format_summarization(self, text: str) -> str:
        def gpt_4o_format(messages, model="gpt-4o"):
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()


        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional medical editor specializing in patient education materials. "
                    "Your task is to take simplified clinical text and polish it into clear, coherent, and fluent English suitable for patients. "
                    "Keep all medical facts accurate — do not add or remove any factual content. "
                    "Make the text smooth, grammatically correct, and supportive in tone. "
                    "Return plain text only. No titles, markdown, or extra commentary."
                )
            },
            {
                "role": "user",
                "content": (
                    "Input: The meta-analysis included 894 men. No studies reported live birth. The combined fixed-effect odds ratio (OR) of the 10 studies for the outcome of pregnancy was 1.47 (95% CI 1.05 to 2.05), favouring the intervention. [...] \n"
                    "Polished: This review analysed 10 studies (894 participants) and found evidence (combined odds ratio was 1.47 (95% CI 1.05 to 2.05)) to suggest an increase in pregnancy rates after varicocele treatment compared to no treatment in subfertile couples. [...]\n\n"

                    "Input: Six studies comprising nearly 450 patients were included. In general the quality of the studies was good. [...] \n"
                    "Polished: The review authors included five randomised and one controlled clinical trial involving a total of nearly 450 patients. In general the quality of the studies was good. [...]\n\n"

                    "Input: Only two eligible trials were included (593 patients), both of reasonable quality although one was unblinded. [...] \n"
                    "Polished: We reviewed the trials that compared giving MAO-B inhibitors with other types of medication in people with early Parkinson's disease. However, only two trials were found (593 patients), so the evidence is limited. [...]\n\n"

                    f"Input: {text}\nPolished:"
                )
            }
        ]
        return gpt_4o_format(messages)




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
    # pipeline = GPT4oAPI()

    raw_text = (
       "Two trials met the inclusion criteria. One compared 2% ketanserin ointment in polyethylene glycol (PEG) with PEG alone, used twice a day by 40 participants with arterial leg ulcers, for eight weeks or until healing, whichever was sooner. One compared topical application of blood-derived concentrated growth factor (CGF) with standard dressing (polyurethane film or foam); both applied weekly for six weeks by 61 participants with non-healing ulcers (venous, diabetic arterial, neuropathic, traumatic, or vasculitic). Both trials were small, reported results inadequately, and were of low methodological quality. Short follow-up times (six and eight weeks) meant it would be difficult to capture sufficient healing events to allow us to make comparisons between treatments. One trial demonstrated accelerated wound healing in the ketanserin group compared with the control group. In the trial that compared CGF with standard dressings, the number of participants with diabetic arterial ulcers were only reported in the CGF group (9/31), and the number of participants with diabetic arterial ulcers and their data were not reported separately for the standard dressing group. In the CGF group, 66.6% (6/9) of diabetic arterial ulcers showed more than a 50% decrease in ulcer size compared to 6.7% (2/30) of non-healing ulcers treated with standard dressing. We assessed this as very-low certainty evidence due to the small number of studies and arterial ulcer participants, inadequate reporting of methodology and data, and short follow-up period. Only one trial reported side effects (complications), stating that no participant experienced these during follow-up (six weeks, low-certainty evidence). It should also be noted that ketanserin is not licensed in all countries for use in humans. Neither study reported time to ulcer healing, patient satisfaction or quality of life. There is insufficient evidence to determine whether the choice of topical agent or dressing affects the healing of arterial leg ulcers."
    )

    # result = pipeline.simplify(raw_text)


    # print(result["formatted"])
    
#     dummyoutput = '''This review includes two studies that examined different treatments for wound healing. The first study focused on an ointment containing 2% ketanserin mixed with polyethylene glycol (PEG) compared to the PEG base alone. In this study, forty participants with artery-related leg wounds applied these treatments twice daily for up to eight weeks or until their wounds healed. Results indicated that wounds healed faster with the ketanserin ointment than with the PEG base alone.

# The second study explored the effects of a special blood-based growth factor in comparison to regular wound coverings, such as thin plastic or foam. This study involved sixty-one participants with non-healing wounds caused by various issues, including vein problems, diabetes, nerve disorders, injuries, or blood vessel inflammation. Treatments were applied once a week for six weeks. Within the growth factor group, 9 out of 31 participants had diabetic artery-related wounds, and 6 of these 9 wounds shrank by more than half. In contrast, only 2 out of 30 wounds in the regular dressing group showed similar improvement.

# Both studies shared limitations, including small sample sizes, poor design, and insufficient reporting. The follow-up period of six to eight weeks was relatively short, complicating efforts to accurately compare treatment outcomes. Additionally, only one study mentioned side effects, reporting none during the six-week period; however, this evidence is not considered reliable.

# Importantly, ketanserin is not approved for human use in all countries, and neither study provided information on healing duration, patient satisfaction, or quality of life. Overall, the evidence is highly uncertain, making it difficult to determine whether the type of ointment or dressing significantly impacts the healing of artery-related leg wounds.'''
        
#     pprint.pp(dummyoutput)
    simplifier = GPT4oAPI()
    result = simplifier.syntactic_simplification(raw_text)
    print(result)
