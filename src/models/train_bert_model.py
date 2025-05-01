# src/models/train_bert_model.py
import os
from pathlib import Path
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


class BertSentimentClassifier:
    def __init__(self, model_name='bert-base-uncased', max_length=128, batch_size=16, num_epochs=3):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.label_encoder = LabelEncoder()
        self.model = None

        # Définir les chemins
        self.project_dir = Path(__file__).resolve().parents[2]
        self.data_path = os.path.join(self.project_dir, 'data', 'processed')
        self.models_path = os.path.join(self.project_dir, 'models')
        os.makedirs(self.models_path, exist_ok=True)

    def load_data(self):
        df = pd.read_csv(os.path.join(self.data_path, 'train_clean.csv'))
        texts = df['text_clean'].tolist()
        labels = self.label_encoder.fit_transform(df['sentiment'].tolist())
        return train_test_split(texts, labels, test_size=0.2, random_state=42)

    def tokenize(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='tf'
        )

    def prepare_datasets(self, X_train, X_val, y_train, y_val):
        train_encodings = self.tokenize(X_train)
        val_encodings = self.tokenize(X_val)

        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            y_train
        )).batch(self.batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((
            dict(val_encodings),
            y_val
        )).batch(self.batch_size)

        return train_dataset, val_dataset

    def build_model(self, num_labels):
        self.model = TFBertForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)
        optimizer = Adam(learning_rate=2e-5)
        self.model.compile(optimizer=optimizer, loss=self.model.compute_loss, metrics=['accuracy'])

    def train(self):
        X_train, X_val, y_train, y_val = self.load_data()
        num_labels = len(set(y_train))

        self.build_model(num_labels)
        train_dataset, val_dataset = self.prepare_datasets(X_train, X_val, y_train, y_val)

        checkpoint_path = os.path.join(self.models_path, 'bert_sentiment_model.h5')
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        self.model.fit(train_dataset, validation_data=val_dataset, epochs=self.num_epochs, callbacks=[checkpoint])
        print(f"✅ Modèle sauvegardé à : {checkpoint_path}")


if __name__ == '__main__':
    classifier = BertSentimentClassifier()
    classifier.train()
