import os
import logging
from random_forest_model import RandomForest

# Configurer le logging pour afficher les messages d'erreur et d'information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    try:
        # Détermine le chemin vers le fichier de configuration
        #script_dir = os.path.dirname(os.path.abspath(__file__))
        #config_path = os.path.join(script_dir, "..", "..", "..", "config", "config.yaml")
        config_path = "C:/Users/pc/Desktop/classification-commentaires/config/config.yaml"
        # Vérifie si le fichier de configuration existe
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Le fichier de configuration '{config_path}' est introuvable.")

        logging.info("Chargement du modèle avec le fichier de configuration : %s", config_path)

        # Crée l'objet RandomForest et lance l'entraînement et l'évaluation
        classifier = RandomForest(config_path)
        classifier.run()

    except FileNotFoundError as e:
        logging.error("Erreur: %s", e)
    except Exception as e:
        logging.error("Une erreur inattendue est survenue : %s", e)
