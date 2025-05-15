🧠 Classification de Commentaires avec BERT & FastAPI
Ce projet vise à développer une application de classification automatique de commentaires textuels (analyse de sentiments ou modération de contenu) en combinant des techniques de NLP avec un modèle BERT, le tout intégré dans une API web via FastAPI.Il intègre également un système de monitoring en temps réel avec Prometheus, Grafana et Weights & Biases (wandb) pour le suivi des performances.

📌 Objectifs

Prétraitement et nettoyage des données textuelles.
Entraînement d’un modèle NLP basé sur BERT pour la classification.
Déploiement d’une API RESTful avec FastAPI.
Monitoring système et applicatif (temps de réponse, usage CPU/RAM, nombre de prédictions, etc.).
Reproductibilité et portabilité grâce à Docker et DVC.


🛠️ Technologies Utilisées



Technologie
Rôle



🤖 BERT / Transformers
Modèle NLP de classification basé sur le langage


⚡ FastAPI
Backend de l'API RESTful


🐍 Python
Langage principal


🐳 Docker / Compose
Conteneurisation et orchestration


📈 Prometheus
Collecte et exposition des métriques applicatives et système


📊 Grafana
Visualisation dynamique des métriques


🧪 wandb
Suivi des expériences et visualisation des performances du modèle


🧬 DVC
Gestion des versions de données et modèles



📁 Structure du Projet
<<<<<<< HEAD
![Interface de l'application](/docs/STructure.png)

🏗️ Schéma de l'Architecture
![Interface de l'application](/docs/Project%20Architecture.jpg)
=======
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
├── 📂grafana/                            # Configuration Grafana pour visualisation
│   └── 📂provisioning/
│       └── 📜datasource.yml              # Configuration sources de données Grafana
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


🏗️ Schéma de l'Architecture

>>>>>>> origin/master

⚙️ Installation & Lancement
1. Cloner le dépôt
git clone https://github.com/Hassanadn/classification-commentaires.git
cd classification-commentaires

2. Lancer les services avec Docker
docker-compose up --build

✅ L'API sera accessible à : http://localhost:8000📄 
Documentation Swagger : http://localhost:8000/docs

3. Arrêter les services
docker-compose down


🌐 Accès aux Interfaces


🧠 API FastAPI
http://localhost:8000
API de classification des commentaires


📄 Swagger
http://localhost:8000/docs
Documentation interactive de l'API


📡 Prometheus
http://localhost:9090
Visualisation des métriques brutes


📊 Grafana
http://localhost:3000
Dashboards personnalisés


🎛️ Grafana Login
admin / admin
Identifiants par défaut (à modifier)



📊 Supervision avec Prometheus & Grafana
L’application expose des métriques via l’endpoint /metrics pour être collectées par Prometheus.
Métriques système :

💻 Utilisation CPU
📈 Consommation mémoire
🌐 Activité réseau

Métriques applicatives :

📦 Nombre total de requêtes
⏱️ Temps moyen de prédiction
🧠 Nombre de prédictions par classe (positif, négatif, neutre)


📈 Suivi des Expériences avec wandb
Chaque entraînement de modèle est suivi avec Weights & Biases :📉 Courbes de perte, 🎯 Précision, ⚖️ F1-score, 🔀 Matrice de confusion, etc.
Connecte-toi avec ton compte wandb :
import wandb
wandb.login()


📦 Versionnage avec DVC
Utilise DVC pour versionner les datasets et modèles :
dvc init
dvc add data/train.csv
dvc push


📮 Exemple d’Utilisation de l’API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Ce produit est incroyable, je recommande !"}'

Réponse attendue :
{
  "label": "positif",
  "score": 0.974
}


👨‍💻 Auteurs

Collaboration 1: ADNAN Hassan
Collaboration 2: EL ATRACH Abdellah
Collaboration 3: EDDREG Khadija
Collaboration 4: OUHMAD Hadda

Projet réalisé dans le cadre du Master Data Science – 2025
