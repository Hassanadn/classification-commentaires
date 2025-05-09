FROM python:3.10-slim

WORKDIR /app

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
