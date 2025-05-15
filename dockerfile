FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
COPY .env ./
COPY dvc.json ./  

RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install dvc[gdrive] uvicorn wandb

COPY . .

CMD ["bash", "-c", "dvc pull && uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"]
