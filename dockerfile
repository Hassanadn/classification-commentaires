FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copie du fichier requirements et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .

# Variables d'environnement pour WANDB et Python
ENV PYTHONPATH=/app
ENV WANDB_DIR=/app/models/wandb

# Point d'entrée
CMD ["python", "src/models/train_models.py"]
