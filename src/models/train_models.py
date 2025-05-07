# import logging
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# # from models.random_forest_model import RandomForest
# from src.data.load_data import DataLoader
# from BertTextClassifier import BertTextClassifier

# if __name__ == "__main__":

#     # try:
#     #     # Détermine le chemin vers le fichier de configuration
#     #     #script_dir = os.path.dirname(os.path.abspath(__file__))
#     #     #config_path = os.path.join(script_dir, "..", "..", "..", "config", "config.yaml")
#     #     config_path = "config/config.yaml"
#     #     # Vérifie si le fichier de configuration existe
#     #     if not os.path.exists(config_path):
#     #         raise FileNotFoundError(f"Le fichier de configuration '{config_path}' est introuvable.")

#     #     logging.info("Chargement du modèle avec le fichier de configuration : %s", config_path)

#     #     # Crée l'objet RandomForest et lance l'entraînement et l'évaluation
#     #     classifier = RandomForest(config_path)
#     #     classifier.run()

#     # except FileNotFoundError as e:
#     #     logging.error("Erreur: %s", e)
#     # except Exception as e:
#     #     logging.error("Une erreur inattendue est survenue : %s", e)


#     config_path = './config/config.yaml'
#     data_loader = DataLoader(config_path)
#     trainer = BertTextClassifier(config_path)

#     for i, chunk_df in enumerate(data_loader.data_generator()):
#         print(f"=== Entraînement sur le chunk {i+1} ===")
#         trainer.train_on_chunk(chunk_df, chunk_num=i+1)

#     # Sauvegarde finale du modèle et du tokenizer
#     final_model_path = "/content/models/final_bert"
#     trainer.save_model(trainer.model, final_model_path)
#     trainer.feature_engineer.tokenizer.save_pretrained(final_model_path)



import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import logging
from src.models.bert_model import BertTextClassifier
from src.models.random_forest_model import RandomForestTextClassifier
from src.data.load_data import DataLoader

class Main:
    def __init__(self, config_path: str):
        """
        Initialise DataLoader et les deux entraîneurs RF et BERT.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config introuvable : {config_path}")
        self.config_path = config_path

        # 1. Charger les données en chunks
        self.data_loader = DataLoader(config_path)

        # 2. Initialiser l'entraîneur Random Forest
        self.rf_trainer = RandomForestTextClassifier(config_path)

        # 3. Initialiser l'entraîneur BERT
        self.bert_trainer = BertTextClassifier(config_path)

        # Dossiers de sauvegarde
        cfg = self.bert_trainer.config  # on suppose que BERT et RF partagent la même config
        self.rf_output_dir = cfg['model']['random_forest']['output_dir']
        self.bert_output_dir = cfg['model']['bert']['output_dir']

        os.makedirs(self.rf_output_dir, exist_ok=True)
        os.makedirs(self.bert_output_dir, exist_ok=True)

    def run(self):
        """
        Pour chaque chunk :
         - On entraîne RF dessus
         - On entraîne BERT dessus
        Enfin on sauvegarde les deux modèles.
        """
        for i, chunk_df in enumerate(self.data_loader.data_generator(), start=1):
            logging.info(f"=== Chunk {i} ===")
            # RF
            logging.info("🔹 Entraînement RandomForest sur chunk %d", i)
            self.rf_trainer.train(chunk_df)

            # BERT
            logging.info("🔹 Entraînement BERT sur chunk %d", i)
            self.bert_trainer.train(chunk_df, chunk_num=i)

        # Sauvegarde finale RF
        logging.info("📦 Sauvegarde finale RandomForest dans %s", self.rf_output_dir)
        self.rf_trainer.save_model(self.rf_trainer.model, self.rf_output_dir)

        # Sauvegarde finale BERT
        logging.info("📦 Sauvegarde finale BERT dans %s", self.bert_output_dir)
        self.bert_trainer.save_model(self.bert_trainer.model, self.bert_output_dir)
        self.bert_trainer.feature_engineer.tokenizer.save_pretrained(self.bert_output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main = Main(config_path="config/config.yaml")
    main.run()
