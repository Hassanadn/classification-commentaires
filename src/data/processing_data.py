import pandas as pd
import re
from pathlib import Path

class DataPreprocessor:
    def __init__(self, input_path: str, output_path: str, chunk_size: int = 10000):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.chunk_size = chunk_size

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)
        return text.strip()

    def _data_generator(self):
        """Générateur qui lit le fichier en chunks, nettoie et retourne les données progressivement."""
        for chunk in pd.read_csv(self.input_path, chunksize=self.chunk_size):
            chunk.dropna(subset=['text', 'label'], inplace=True)
            chunk['text'] = chunk['text'].astype(str).apply(self._clean_text)
            yield chunk

    def process_and_save(self):
        """Traite chaque chunk et l'enregistre progressivement dans un fichier."""
        first = True
        for cleaned_chunk in self._data_generator():
            cleaned_chunk.to_csv(self.output_path, mode='a', index=False, header=first)
            first = False

    def preview_first_chunk(self) -> pd.DataFrame:
        """Renvoie uniquement le premier chunk nettoyé pour inspection."""
        generator = self._data_generator()
        return next(generator)

# Exemple d'utilisation
if __name__ == "__main__":
    input_path = "../data/raw/train.csv"
    output_path = "../data/processed/train_clean.csv"

    preprocessor = DataPreprocessor(input_path, output_path)
    preprocessor.process_and_save()

    preview = preprocessor.preview_first_chunk()
    print(preview.head())
