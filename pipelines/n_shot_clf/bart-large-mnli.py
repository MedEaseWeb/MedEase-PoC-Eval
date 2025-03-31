from transformers import pipeline

class NShotMedicalClassifier:
    """
    Wrapper around a zero-shot classification model to distinguish medical reports from other text.
    """

    def __init__(self, model_name="facebook/bart-large-mnli", candidate_labels=None):
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        self.labels = candidate_labels or ["medical report",
                                            "technical documentation",
                                            "news article",
                                            "personal message",
                                            "social media post",
                                            "casual conversation",
                                            "legal text",
                                            "educational content",
                                            "code or programming request",
                                            "AI prompt engineering"]

    def classify(self, text: str, labels=None) -> dict:
        """
        Classify a given text into one of the predefined or user-provided labels.
        Returns a dict with label scores.
        """
        used_labels = labels if labels else self.labels
        result = self.classifier(text, used_labels)
        return dict(zip(result["labels"], result["scores"]))

    def is_medical(self, text: str) -> bool:
        """
        Check if 'medical report' appears in the top 3 predicted labels.
        """
        used_labels = self.labels
        result = self.classifier(text, used_labels)
        top_labels = result["labels"][:3]
        return "medical report" in top_labels


if __name__ == "__main__":
    clf = NShotMedicalClassifier()
    sample_text = "This class is actually torturing and it is bad for my mental health. It is causing my head ache and my blood pressure is actually going up."
    
    print("Classification Results:\n", clf.classify(sample_text))
    print("\nIs medical report:", clf.is_medical(sample_text))
