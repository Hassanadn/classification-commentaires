import os
import re
import yaml
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Iterator, Union
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Configure global logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config_path: Union[str, Path] = None):
        # Determine project root and default config path
        project_root = Path(__file__).resolve().parents[2]
        default_cfg = project_root / "config" / "config.yaml"
        self.config_path = Path(config_path) if config_path else default_cfg
        
        # Load config
        self.config = self._load_config(self.config_path)
        
        # Paths & params
        self.input_file = project_root / self.config.get("input_file", "data/raw/train.csv")
        self.output_file = project_root / self.config.get("output_file", "data/processed/x_train_clean.csv")
        self.chunk_size = self.config.get("chunk_size", 10000)
        
        # Preprocessing settings
        pp = self.config.get("preprocessing", {})
        self.lowercase        = pp.get("lowercase", True)
        self.remove_stopwords = pp.get("remove_stopwords", True)
        self.min_word_length  = pp.get("min_word_length", 1)

    def _load_config(self, path: Path) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                logger.info(f"Configuration chargée depuis {path}")
                return cfg
        except Exception as e:
            logger.warning(f"Impossible de charger {path}: {e}\nUtilisation des valeurs par défaut.")
            return {}

    def _preprocess_text(self, text: str) -> str:
        if self.lowercase:
            text = text.lower()
        text = re.sub(r"\W+", " ", text)
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
        tokens = [t for t in tokens if len(t) >= self.min_word_length]
        return " ".join(tokens)

    def _data_generator(self) -> Iterator[pd.DataFrame]:
        for chunk in pd.read_csv(self.input_file, chunksize=self.chunk_size, encoding="utf-8"):
            logger.info(f"Chunk chargé: {len(chunk)} lignes")
            chunk.dropna(subset=["text", "label"], inplace=True)
            chunk["text"] = chunk["text"].astype(str).apply(self._preprocess_text)
            yield chunk

    def run(self):
        """Lance le prétraitement et écrit les résultats dans output_file."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        first = True
        for chunk in self._data_generator():
            chunk.to_csv(
                self.output_file,
                mode="a",
                index=False,
                header=first,
                encoding="utf-8"
            )
            logger.info(f"Chunk sauvegardé: {len(chunk)} lignes")
            first = False

if __name__ == "__main__":
    processor = DataPreprocessor()
    logger.info("Démarrage du prétraitement...")
    try:
        processor.run()
        logger.info("Prétraitement terminé avec succès!")
    except Exception as e:
        logger.error(f"Erreur durant le prétraitement: {e}")