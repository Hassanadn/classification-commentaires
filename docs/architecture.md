├── config/                            # Fichiers de configuration
│   ├── config.yaml                    # Fichier principal de configuration des modèles et pipeline
│   ├── wandb_config.yaml              # Paramètres pour wandb (API key, nom de projet, etc.)
│   └── logging.conf                   # Configuration du format et niveau de logging

├── data/                              # Contient les jeux de données
│   ├── processed/                     # Données nettoyées et transformées
│   │   └── processed_data.csv         # Exemple de données prêtes à être utilisées
│   ├── raw/                           # Données brutes originales
│   │   └── raw_data.csv               # Exemple de données brutes

├── docs/                              # Documentation technique du projet
│   └── architecture.md                # Description de l'architecture logicielle et dataflow

├── models/                            # Répertoire de sauvegarde des modèles entraînés
│   ├── bert_model.pkl                 # Modèle BERT entraîné et sérialisé
│   └── random_forest_model.pkl        # Modèle Random Forest entraîné et sérialisé

├── notebooks/                         # Jupyter Notebooks pour exploration et tests
│   ├── teset_random_forest.ipynb      # Notebook d'exploration  avec Random Forest (GPU)
│   └── test_bert.ipynb                # Tests et visualisation avec BERT (GPU)

├── src/                               # Code source principal
│   ├── api/                           # Serveur d’API pour exposer les modèles
│   │   ├── index.html                 # Interface simple
│   │   └── main.py                    # Point d’entrée de l’API

│   ├── data/                          # Chargement et découpage des données
│   │   └── load_data.py               # Script de chargement

│   ├── features/                      # Création de nouvelles variables/features
│   │   ├── feature_engineering/       # Feature engineering spécifique à BERT
│   │   └── text_processor.py          # Préparation des features pour Random Forest

│   ├── models/                        # Entraînement et abstraction des modèles
│   │   ├── abstract_text_classification_model.py  # Classe abstraite de base
│   │   ├── bert_model.py              # Modèle BERT
│   │   ├── random_forest_model.py     # Modèle Random Forest
│   │   ├── sentiment_dataset.py       # Préparation du dataset
│   │   └── train_models.py            # Entraînement des modèles

│   ├── monitoring/                    # Suivi et observabilité
│   │   ├── log_metrics.py             # Envoi vers InfluxDB / Prometheus
│   │   └── wandb_logger.py            # Intégration Weights & Biases

│   └── utils/                         # Fonctions réutilisables
│       ├── helper_functions.py        # Fonctions utiles génériques
│       └── wandb_utils.py             # Utilitaires wandb



├── .dvc/                              # Métadonnées internes DVC (si utilisé)
├── .gitignore                         # Fichiers à ignorer dans le repo
├── dvc.lock                           # Verrouillage des versions DVC
├── dvc.yaml                           # Pipeline DVC (étapes d’entraînement, prétraitement)
├── docker-compose.yml                 # Configuration multi-services Docker
├── dockerfile                         # Image Docker du projet
├── README.md                          # Description et instructions du projet
├── requirements.txt                   # Dépendances Python


