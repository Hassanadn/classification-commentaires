from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import yaml
from src.api.predict import TextClassifier

app = FastAPI()
config_path = "config/config.yaml"

# Charger la configuration
with open(config_path) as f:
    config = yaml.safe_load(f)

# Charger le modèle
classifier = TextClassifier(config_path)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
async def predict(input: TextInput):
    prediction, probabilities = classifier.predict(input.text)
    return {
        "prediction": int(prediction),
        "probabilities": probabilities.tolist()
    }

@app.get("/")
async def root():
    return {"message": "API de classification de texte"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)