├── .dvc/                              # Métadonnées internes DVC (si utilisé)
├── config/                            # Fichiers de configuration
│   ├── config.yaml                    # Fichier principal de configuration des modèles et pipeline
│   ├── wandb_config.yaml              # Paramètres pour wandb (API key, nom de projet, etc.)
│   ├── logging.conf                   # Configuration du format et niveau de logging
│   └── prometheus.yml                 # Configuration de Prometheus (si utilisé)
│   └── .env                           # Fichier de configuration pour les variables d'environnement wandb 
│   └── log_config.yaml                # Configuration des logs
├── data/                              # Contient les jeux de données
│   ├── processed/                     # Données nettoyées et transformées
│   │   └── processed_data.csv
│   ├── raw/                           # Données brutes originales
│   │   └── raw_data.csv
│   └── logs/                          # Fichier de logs
│       └── logs.txt
├── docs/                              # Documentation technique du projet
│   └── architecture.md
├── grafana/                           # Tableau de bord Grafana (ex. fichiers JSON si configurés)
├── models/                            # Répertoire de sauvegarde des modèles entraînés
│   ├── bert_model.pkl
│   └── random_forest_model.pkl
├── notebooks/                         # Jupyter Notebooks pour exploration et tests
│   ├── ex.ipynb
│   └── test_bert.ipynb
├── src/                               # Code source principal
│   ├── api/
│   │   ├── index.html
│   │   └── main.py
│   ├── data/
│   │   └── load_data.py
│   ├── features/
│   │   ├── feature_engineering/
│   │   └── text_processor.py
│   ├── models/
│   │   ├── abstract_text_classification_model.py
│   │   ├── bert_model.py
│   │   ├── random_forest_model.py
│   │   ├── sentiment_dataset.py
│   │   └── train_models.py
│   ├── monitoring/
│   │   ├── log_metrics.py
│   │   └── wandb_logger.py
│   └── utils/
│       ├── helper_functions.py
│       └── wandb_utils.py
│       └── log_utils.py
├── .gitignore
├── dvc.lock
├── dvc.yaml
├── docker-compose.yml
├── dockerfile
├── README.md
├── requirements.txt