📦Classification-Commentaire
├── 📂.dvc/
├── 📂config/
│   ├── 📜config.yaml
│   ├── 📜wandb_config.yaml
│   ├── 📜logging.conf
│   └── 📜prometheus.yml
│
├── 📂data/
│   ├── 📂processed/
│   │   └── 📜processed_data.csv
│   └── 📂raw/
│       ├── 📜test.csv
│       ├── 📜test.csv.dvc
│       ├── 📜train.csv
│       └── 📜train.csv.dvc
│
├── 📂docs/
│   └── 📜architecture.md
│
├── 📂grafana/provisioning/
│   └── 📜datasource.yml
│
├── 📂models/
│   └── 📜bert_model/
│
├── 📂notebooks/
│   └── 📜test_bert.ipynb
│
├── 📂src/
│   ├── 📂api/
│   │   ├── 📂templates/            # Nouveau dossier pour les templates HTML
│   │   │   ├── 📜dashboard.html
│   │   │   ├── 📜filtered_dash.html
│   │   │   ├── 📜index.html
│   │   │   └── 📜loging.html
│   │   ├── 📜database.py
│   │   ├── 📜models.py
│   │   └── 📜main.py
│   │
│   ├── 📂data/
│   │   └── 📜load_data.py
│   │
│   ├── 📂features/
│   │   ├── 📂feature_engineering/
│   │   └── 📜text_processor.py
│   │
│   ├── 📂models/
│   │   ├── 📜abstract_text_classification_model.py
│   │   ├── 📜bert_model.py
│   │   ├── 📜random_forest_model.py
│   │   ├── 📜sentiment_dataset.py
│   │   └── 📜train_models.py
│   │
│   ├── 📂monitoring/
│   │   └── 📜wandb_logger.py
│   │
│   └── 📂utils/
│       ├── 📜helper_functions.py
│       ├── 📜wandb_utils.py
│       └── 📜sentiment-analysis.ico
│
├── 📜.dockerignore
├── 📜.dvcignore
├── 📜.env
├── 📜.gitignore
├── 📜comments.db
├── 📜docker-compose.yml
├── 📜Dockerfile
├── 📜README.md
└── 📜requirements.txt