📦Classification-Commentaire
│
├── 📂.dvc/                               # Dossier de configuration Data Version Control
├── 📂config/                             # Configuration du projet
│   ├── 📜config.yaml                     # Configuration générale
│   ├── 📜wandb_config.yaml               # Configuration Weights & Biases
│   ├── 📜logging.conf                    # Configuration des logs
│   └── 📜prometheus.yml                  # Configuration Prometheus pour monitoring
│
├── 📂data/                               # Données du projet
│   ├── 📂processed/                      # Données prétraitées
│   │   └── 📜processed_data.csv          # Données après traitement
│   └── 📂raw/                            # Données brutes
│       ├── 📜test.csv                    # Données de test
│       ├── 📜test.csv.dvc                # Fichier DVC pour données de test
│       ├── 📜train.csv                   # Données d'entraînement
│       └── 📜train.csv.dvc               # Fichier DVC pour données d'entraînement
│
├── 📂docs/                               # Documentation
│   └── 📜architecture.md                 # Description de l'architecture
│
<<<<<<< HEAD
├── 📂grafana/                            # Configuration Grafana pour visualisation
│   └── 📂provisioning/
│       └── 📜datasource.yml              # Configuration sources de données Grafana
=======
├── 📂models/
│   └── 📜bert_model/
>>>>>>> origin/abde/mlops_v2
│
├── 📂models/                             # Modèles entraînés sauvegardés
│   └── 📜bert_model                      # Modèle BERT sauvegardé
│
├── 📂notebooks/                          # Notebooks pour exploration et tests
│   └── 📜test_bert.ipynb                 # Test du modèle BERT
│
├── 📂src/                                # Code source
│   ├── 📂api/                            # API FastAPI
│   │   ├── 📂routes/                     # Routes FastAPI
│   │   │   ├── 📜auth_routes.py          # Routes d'authentification (login/logout)
│   │   │   ├── 📜dashboard.py            # Routes pour le dashboard
│   │   │   ├── 📜home.py                 # Routes pour la page d'accueil
│   │   │   └── 📜comments.py             # Routes pour la gestion des commentaires
│   │   │
│   │   ├── 📂services/                   # Services pour l'API
│   │   │   └── 📜auth.py                 # Service d'authentification (fonctions login, verify, token)
│   │   │
│   │   ├── 📂schemas/                    # Schémas de validation
│   │   │   ├── 📜comments.py             # Validation Pydantic pour commentaires
│   │   │   └── 📜user.py                 # Validation Pydantic pour utilisateurs
│   │   │
│   │   ├── 📂templates/                  # Templates HTML Jinja2
│   │   │   ├── 📜dashboard.html          # Template pour le dashboard
│   │   │   ├── 📜filtered_dash.html      # Template pour dashboard filtré
│   │   │   ├── 📜index.html              # Template pour page d'accueil
│   │   │   └── 📜login.html              # Template pour page de connexion
│   │   │
│   │   ├── 📜database.py                 # Gestion de la base de données
│   │   ├── 📜models.py                   # Modèles FastAPI
│   │   ├── 📜main.py                     # Point d'entrée FastAPI
│   │   ├── 📜default_user.py             # Création d'un utilisateur par défaut
│   │   └── 📜db.py                       # Connexion à SQLite
│   │
│   ├── 📂data/                           # Traitement des données
│   │   └── 📜load_data.py                # Chargement des données
│   │
│   ├── 📂features/                       # Ingénierie des caractéristiques
│   │   ├── 📂feature_engineering/        # Dossier pour extraction de caractéristiques
│   │   └── 📜text_processor.py           # Traitement de texte
│   │
│   ├── 📂models/                         # Modèles d'apprentissage automatique
│   │   ├── 📜abstract_text_classification_model.py  # Classe abstraite pour modèles
│   │   ├── 📜bert_model.py               # Implémentation du modèle BERT
│   │   ├── 📜random_forest_model.py      # Implémentation Random Forest
│   │   ├── 📜sentiment_dataset.py        # Dataset pour analyse de sentiment
│   │   └── 📜train_models.py             # Script d'entraînement des modèles
│   │
│   ├── 📂monitoring/                     # Monitoring des modèles
│   │   └── 📜wandb_logger.py             # Logger pour Weights & Biases
│   │
│   └── 📂utils/                          # Utilitaires
│       ├── 📜helper_functions.py         # Fonctions auxiliaires
│       ├── 📜wandb_utils.py              # Utilitaires pour Weights & Biases
│       └── 📜sentiment-analysis.ico      # Icône pour l'application
│
├── 📜.dockerignore                       # Fichiers à ignorer pour Docker
├── 📜.dvcignore                          # Fichiers à ignorer pour DVC
├── 📜.env                                # Variables d'environnement
├── 📜comments.db                         # Base de données SQLite
├── 📜docker-compose.yml                  # Configuration des services Docker
├── 📜Dockerfile                          # Instructions de build Docker
├── 📜README.md                           # Documentation principale
└── 📜requirements.txt                    # Dépendances Python