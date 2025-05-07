import joblib
import os
import yaml

class TextProcessor:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def clean_text(self, text):
        import re
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

