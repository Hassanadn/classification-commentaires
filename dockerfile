# Utiliser l'image officielle Python 3.10 slim
FROM python:3.10-slim

# Mettre à jour les paquets système et installer les dépendances
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        libssl-dev \
        curl \
        gcc \
        git \
        && rm -rf /var/lib/apt/lists/*

# Installer les dernières versions de pip, setuptools et wheel
RUN pip install --upgrade pip setuptools wheel

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt
COPY requirements.txt requirements.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    -r requirements.txt

# Copier le reste du code source
COPY src/ src/
COPY models/ models/
COPY config/ config/
COPY . .

# Exposer le port utilisé par FastAPI
EXPOSE 8000

# Commande pour démarrer l'application FastAPI
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
