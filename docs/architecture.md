├── README.md              # Fichier principal avec la description du projet   
├── data/
│   ├── raw/               # Données brutes, non modifiées
│   │   └── raw_data.csv   # Exemple de données brutes
│   ├── processed/         # Données prétraitées
│   │   └── processed_data.csv  # Exemple de données prétraitées
│   └── external/          # Données provenant de sources externes
│       └── external_data.csv  # Exemple de données externes
├── models/                # Stockage des modèles entraînés
│   ├── model_v1.pkl       # Modèle version 1
│   └── model_v2.pkl       # Modèle version 2
├── src/                   # Code source
│   ├── data/              # Scripts pour charger et traiter les données
│   │   └── load_data.py   # Script pour charger les données
│   ├── features/          # Scripts pour créer des features
│   │   └── text_processor.py  # Script pour l'ingénierie des features
│   ├── models/            # Scripts pour entraîner et évaluer les modèles
│   │   ├── train_model.py  # Script pour entraîner le modèle
│   │   └── evaluate_model.py  # Script pour évaluer le modèle
│   ├── api/               # Code pour l'API FastAPI
│   │   ├── main.py        # Point d'entrée de l'API
│   │   └── predict.py     # Script pour prédire avec le modèle
│   ├── visualization/     # Code pour visualiser les résultats
│   │   └── plot_results.py  # Script pour visualiser les résultats
│   └── utils/             # Fonctions utilitaires
│       └── helper_functions.py  # Fonctions utiles réutilisables
├── notebooks/             # Jupyter Notebooks pour exploration et prototypage
│   └── exploration.ipynb  # Exemple de notebook pour l'exploration des données
├── config/                # Fichiers de configuration
│   ├── config.yaml        # Exemple de fichier de configuration (par exemple pour les paramètres du modèle)
│   └── logging.conf       # Fichier de configuration pour la gestion des logs
├── tests/                 # Tests unitaires et d'intégration
│   ├── test_data.py       # Test pour les scripts de données
│   ├── test_model.py      # Test pour les scripts de modèle
│   ├── test_api.py        # Test pour l'API
│   └── test_visualization.py  # Test pour les scripts de visualisation
├── docs/
│    └── architecture.md
└── requirements.txt       # Liste des dépendances du projet

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


