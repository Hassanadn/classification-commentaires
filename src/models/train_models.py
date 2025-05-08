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
import pandas as pd
import logging

# Ajoute le dossier racine au sys.path pour les imports relatifs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import logging
from src.models.bert_model import BertTextClassifier
from src.data.load_data import DataLoader

class Main:
    def __init__(self, config_path: str, wandb_config_path: str = "config/wandb_config.yaml"):
        """
        Initialise DataLoader et les deux entraîneurs RF et BERT.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config introuvable : {config_path}")
            
        if not os.path.exists(wandb_config_path):
            logging.warning(f"Config wandb introuvable : {wandb_config_path}. Fonctionnement sans wandb.")
            self.use_wandb = False
        else:
            self.use_wandb = True
            
        self.config_path = config_path
        self.wandb_config_path = wandb_config_path
        
        # Initialiser les loggers de métriques
        if self.use_wandb:
            self.wandb_logger = WandbLogger(wandb_config_path)
            self.wandb_run = self.wandb_logger.setup()
            
        self.metrics_logger = MetricsLogger()
        
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
        
        # Log de configuration dans wandb
        if self.use_wandb:
            self.wandb_logger.log_metrics({
                'config': self._flatten_config(cfg)
            })
            
    def _flatten_config(self, config: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """
        Aplatit un dictionnaire de configuration imbriqué pour faciliter le logging.
        
        Args:
            config: Configuration sous forme de dictionnaire
            parent_key: Clé parente pour la récursion
            
        Returns:
            Dictionnaire aplati avec des clés de la forme 'parent.enfant'
        """
        items = {}
        for k, v in config.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.update(self._flatten_config(v, new_key))
            else:
                items[new_key] = v
                
        return items

    def run(self):
        """
        Pour chaque chunk :
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
        bert_model_path = os.path.join(self.bert_output_dir, "model.pt")
        self.bert_trainer.save_model(self.bert_trainer.model, self.bert_output_dir)
        self.bert_trainer.feature_engineer.tokenizer.save_pretrained(self.bert_output_dir)
        
        # Log des modèles dans wandb
        if self.use_wandb:
            # Random Forest
            rf_metadata = {
                'model_type': 'random_forest',
                'final_metrics': rf_metrics if 'rf_metrics' in locals() else {}
            }
            self.wandb_logger.log_model(rf_model_path, 'random_forest_model', rf_metadata)
            
            # BERT
            bert_metadata = {
                'model_type': 'bert',
                'final_metrics': bert_metrics if 'bert_metrics' in locals() else {}
            }
            self.wandb_logger.log_model(bert_model_path, 'bert_model', bert_metadata)
            
            # Terminer la session wandb
            self.wandb_logger.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entraînement des modèles avec intégration wandb')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Chemin vers le fichier de configuration')
    parser.add_argument('--wandb-config', type=str, default='config/wandb_config.yaml',
                        help='Chemin vers le fichier de configuration wandb')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Désactiver l\'intégration wandb')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    if args.no_wandb:
        main = Main(config_path=args.config)
    else:
        main = Main(config_path=args.config, wandb_config_path=args.wandb_config)
    
    main.run()
