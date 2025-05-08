import os
import sys
import pandas as pd
import logging

# Ajoute le dossier racine au sys.path pour les imports relatifs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.bert_model import BertTextClassifier
from src.data.load_data import DataLoader


class Main:
    def __init__(self, config_path: str):
        """
        Initialise DataLoader et l'entraîneur BERT.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config introuvable : {config_path}")
        self.config_path = config_path

        # 1. Charger les données en chunks
        self.data_loader = DataLoader(config_path)

        # 2. Initialiser l'entraîneur BERT
        self.bert_trainer = BertTextClassifier(config_path)

        # Dossier de sauvegarde
        cfg = self.bert_trainer.config
        self.bert_output_dir = cfg['model']['bert']['output_dir']
        os.makedirs(self.bert_output_dir, exist_ok=True)

    def run(self):
        """
        Pour chaque chunk :
         - On entraîne BERT dessus
        Enfin on sauvegarde le modèle.
        """
        for i, df_chunk in enumerate(self.data_loader.data_generator()):
            logging.info(f"=== Chunk {i} ===")
            logging.info("🔹 Entraînement BERT sur chunk %d", i)
            self.bert_trainer.train(df_chunk, chunk_num=i)
            # self.bert_trainer.train((texts, labels), chunk_num=i)

        # Sauvegarde finale BERT
        logging.info("📦 Sauvegarde finale BERT dans %s", self.bert_output_dir)
        self.bert_trainer.save_model(self.bert_trainer.model, self.bert_output_dir)
        self.bert_trainer.feature_engineer.tokenizer.save_pretrained(self.bert_output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main = Main(config_path="config/config.yaml")
    main.run()
