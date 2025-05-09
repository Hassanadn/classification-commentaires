import os
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.features.feature_engineering import FeatureEngineer
from src.models.abstract_text_classification_model import TextClassificationModel
from src.models.sentiment_dataset import SentimentDataset


global_metrics = []

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1
    }

class BertTextClassifier(TextClassificationModel):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.model_name = self.config['model']['bert']['model_name']
        self.num_labels = self.config['model']['bert']['num_labels']
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        self.feature_engineer = FeatureEngineer(config_path)
        self.output_dir = self.config['model']['bert']['output_dir']

    def save_model(self, model, path: str):
        model.save_pretrained(path)
        print(f" Modèle sauvegardé à l’emplacement : {path}")

    def train(self, df, chunk_num):
        print(f"\n Entraînement sur chunk {chunk_num}")
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'].tolist(),
            df['label'].tolist(),
            test_size=self.config['model']['bert']['test_size'],
            random_state=self.config['model']['bert']['random_state']
        )

        train_encodings = self.feature_engineer.transform_bert(train_texts)
        val_encodings = self.feature_engineer.transform_bert(val_texts)

        train_dataset = SentimentDataset(train_encodings, train_labels)
        val_dataset = SentimentDataset(val_encodings, val_labels)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config['model']['bert']['epochs'],
            per_device_train_batch_size=self.config['model']['bert']['batch_size'],
            per_device_eval_batch_size=self.config['model']['bert']['batch_size'],
            warmup_steps=self.config['model']['bert']['warmup_steps'],
            weight_decay=self.config['model']['bert']['weight_decay'],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir='./logs',
            logging_steps=100,
            save_total_limit=self.config['model']['bert']['save_total_limit'],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        checkpoint_path = os.path.join(self.output_dir, "checkpoint-last")
        resume = os.path.exists(checkpoint_path)

        #Continuer l’entraînement à partir du dernier checkpoint si disponible
        if resume:
            print(f"Reprise depuis le checkpoint : {checkpoint_path}")
            trainer.train(resume_from_checkpoint=checkpoint_path)
        else:
            print("Entraînement initial")
            trainer.train()

        eval_results = trainer.evaluate()
        eval_results['chunk'] = chunk_num
        global_metrics.append(eval_results)

        # Sauvegarde manuelle du modèle pour la prochaine itération
        self.model.save_pretrained(self.output_dir)
