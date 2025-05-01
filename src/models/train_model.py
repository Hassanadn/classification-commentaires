import os
import logging
import pandas as pd
import tensorflow as tf
from pathlib import Path
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BERTTrainer:
    def __init__(self, data_path, model_output_path, model_name="bert-base-uncased", max_length=128, batch_size=16, epochs=3):
        self.data_path = data_path
        self.model_output_path = Path(model_output_path)
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs

        # Charger les données
        self.df = pd.read_csv(self.data_path)
        self.texts = self.df["text"].tolist()
        self.labels = self.df["label"].tolist()

        # Encoder les labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

        # Charger le tokenizer BERT
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        # Charger le modèle BERT pour la classification
        self.model = TFBertForSequenceClassification.from_pretrained(self.model_name, num_labels=len(set(self.encoded_labels)))

    def _tokenize(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="tf"
        )

    def _build_dataset(self):
        encoded_inputs = self._tokenize(self.texts)
        dataset = tf.data.Dataset.from_tensor_slices((dict(encoded_inputs), self.encoded_labels))
        return dataset.shuffle(1000).batch(self.batch_size)

    def train(self):
        logger.info("Préparation du dataset...")
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            self.texts, self.encoded_labels, test_size=0.2, random_state=42
        )

        logger.info("Tokenisation des données...")
        train_encodings = self._tokenize(train_texts)
        val_encodings = self._tokenize(val_texts)

        train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), train_labels)).shuffle(1000).batch(self.batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings), val_labels)).batch(self.batch_size)

        logger.info("Compilation du modèle BERT...")
        optimizer = tf.keras.optimizers.AdamW(learning_rate=2e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[metric]
        )

        logger.info("Entraînement du modèle...")
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs
        )

        logger.info("Sauvegarde du modèle au format .h5...")
        self.model_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(self.model_output_path.with_suffix(".h5")))
        self.tokenizer.save_pretrained(str(self.model_output_path.with_suffix(".h5")))

        logger.info(f"Modèle sauvegardé dans : {self.model_output_path.with_suffix('.h5')}")
        return history

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed" / "x_train_clean.csv"
    model_output_path = project_root / "models" / "bert_model"

    trainer = BERTTrainer(data_path=data_path, model_output_path=model_output_path)
    try:
        trainer.train()
        logger.info("Modèle BERT entraîné et sauvegardé avec succès !")
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du modèle : {e}")