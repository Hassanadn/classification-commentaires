import yaml
import logging
from typing import List, Dict
from transformers import BertTokenizer, BatchEncoding

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineer:
    def __init__(self, config_path: str) -> None:
        """
        Initialise le tokenizer BERT à partir du fichier de configuration YAML.
        """
        with open(config_path, 'r') as file:
            self.config: Dict = yaml.safe_load(file)

        model_name: str = self.config['model']['bert']['model_name']
        self.bert_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
        logging.info("FeatureEngineer (BERT) initialisé avec succès")

    def transform_bert(self, texts: List[str], max_length: int = 128) -> BatchEncoding:
        """
        Transforme une liste de textes en encodages compatibles BERT (input_ids, attention_mask).
        
        Args:
            texts (List[str]): La liste de textes bruts à transformer.
            max_length (int): La longueur maximale des séquences (défaut 128).
        
        Returns:
            BatchEncoding: Les encodages BERT prêts pour l'entraînement avec PyTorch.
        """
        encodings: BatchEncoding = self.bert_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        logging.info(f"{len(texts)} textes transformés en encodages BERT.")
        return encodings

# just for testing commit 