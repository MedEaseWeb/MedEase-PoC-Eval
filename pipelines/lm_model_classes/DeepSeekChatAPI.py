import os
from dotenv import load_dotenv
from dotenv import find_dotenv
from pathlib import Path
import pprint
import requests
import json


load_dotenv(find_dotenv())
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def deepseek_simplify(prompt, model="deepseek-chat", temperature=0.7, max_tokens=800):
    """
    Calls the DeepSeek API with a user-defined prompt.
    """
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[DeepSeek ERROR] {e}")
        return ""

class DeepSeekChatAPI:
    """
    A DeepSeek-chat based pipeline for three simplification stages.
    """

    def __init__(self, model="deepseek-chat"):
        self.model = model

    def lexical_simplification(self, text: str) -> str:
        prompt = (
            f"You are a medical language simplification assistant. Your task is to replace all complex medical jargon "
            f"in the following text with plain, layman-friendly language, without changing the meaning.\n\n"
            f"Text:\n{text}"
        )
        return deepseek_simplify(prompt, model=self.model)

    def syntactic_simplification(self, text: str) -> str:
        prompt = (
            f"You are a text simplifier. Break the following medical text into shorter, simpler, and more readable sentences. "
            f"Avoid unnecessary repetition, and keep the meaning intact.\n\n"
            f"Text:\n{text}"
        )
        return deepseek_simplify(prompt, model=self.model)

    def format_summarization(self, text: str) -> str:
        prompt = (
            f"You are a summarization assistant. Improve the paragraph structure and logical flow of the following simplified medical text. "
            f"Group related ideas together and ensure the output is clean and easy to read.\n\n"
            f"Text:\n{text}"
        )
        return deepseek_simplify(prompt, model=self.model)

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
    pipeline = DeepSeekChatAPI()

    raw_text = (
       "Two trials met the inclusion criteria. One compared 2% ketanserin ointment in polyethylene glycol (PEG) with PEG alone, used twice a day by 40 participants with arterial leg ulcers, for eight weeks or until healing, whichever was sooner. One compared topical application of blood-derived concentrated growth factor (CGF) with standard dressing (polyurethane film or foam); both applied weekly for six weeks by 61 participants with non-healing ulcers (venous, diabetic arterial, neuropathic, traumatic, or vasculitic). Both trials were small, reported results inadequately, and were of low methodological quality. Short follow-up times (six and eight weeks) meant it would be difficult to capture sufficient healing events to allow us to make comparisons between treatments. One trial demonstrated accelerated wound healing in the ketanserin group compared with the control group. In the trial that compared CGF with standard dressings, the number of participants with diabetic arterial ulcers were only reported in the CGF group (9/31), and the number of participants with diabetic arterial ulcers and their data were not reported separately for the standard dressing group. In the CGF group, 66.6% (6/9) of diabetic arterial ulcers showed more than a 50% decrease in ulcer size compared to 6.7% (2/30) of non-healing ulcers treated with standard dressing. We assessed this as very-low certainty evidence due to the small number of studies and arterial ulcer participants, inadequate reporting of methodology and data, and short follow-up period. Only one trial reported side effects (complications), stating that no participant experienced these during follow-up (six weeks, low-certainty evidence). It should also be noted that ketanserin is not licensed in all countries for use in humans. Neither study reported time to ulcer healing, patient satisfaction or quality of life. There is insufficient evidence to determine whether the choice of topical agent or dressing affects the healing of arterial leg ulcers."
    )

    # result = pipeline.simplify(raw_text)


    # print(result["formatted"])
    
    dummyoutput = '''### **Study Overview**  
Two small studies were reviewed, each investigating different treatments for leg ulcers.  

#### **Study 1: Ketanserin Ointment**  
- **Design:** Compared a 2% ketanserin ointment mixed with a moisturizer (PEG) against the moisturizer alone.  
- **Participants:** 40 people with leg ulcers caused by poor blood flow.  
- **Treatment:** Applied twice daily for 8 weeks or until healing (whichever came first).  

#### **Study 2: Concentrated Growth Factors**  
- **Design:** Compared a wound treatment derived from concentrated blood growth factors to standard dressings (e.g., foam or film).  
- **Participants:** 61 people with slow-healing ulcers due to diabetes, poor circulation, nerve damage, injury, or blood vessel issues.  
- **Treatment:** Administered weekly for 6 weeks.  

---  

### **Study Limitations**  
- Both studies were small, poorly reported, and had weak designs.  
- Follow-up was short (6–8 weeks), leaving long-term effects uncertain.  

---  

### **Results**  

#### **Ketanserin Study**  
- Ulcers healed faster with ketanserin than with the moisturizer alone.  

#### **Growth Factor Study**  
- Only 9 of 31 treated patients had diabetes-related ulcers.  
- **Treatment Group:** 6 of 9 (66.6%) saw ulcers shrink by >50%.  
- **Control Group:** Only 2 of 30 (6.7%) improved similarly.  
- **Issue:** The study did not specify how many control patients had diabetes-related ulcers, making comparisons unreliable.  

---  

### **Key Limitations**  
- **Weak Evidence:** Few studies, poor reporting, and short follow-up.  
- **Side Effects:** Only one study mentioned them (none reported after 6 weeks, but data quality was low).  

---  

### **Additional Notes**  
- Ketanserin is not universally approved for human use.  
- Neither study measured full healing time, patient satisfaction, or quality of life.  

### **Conclusion**  
There is insufficient evidence to support the effectiveness of these treatments for leg ulcers caused by poor blood flow.  

---  

This version improves readability by grouping related information, using clear headings, and maintaining a logical progression from study details to results and conclusions. Let me know if you'd like any further refinements!'''
    pprint.pp(dummyoutput)