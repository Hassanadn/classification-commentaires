# Importation des bibliothèques nécessaires
import os
import numpy as np
import pandas as pd

# Importation des outils de Hugging Face pour le modèle BERT et l'entraînement
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers.integrations import WandbCallback

# Importation des fonctions d’évaluation
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Importation des modules internes du projet
from src.monitoring.wandb_logger import WandbLogger
from src.features.feature_engineering import FeatureEngineer
from src.models.abstract_text_classification_model import TextClassificationModel
from src.models.sentiment_dataset import SentimentDataset

# Intégration de Weights & Biases pour le suivi de l'entraînement
import wandb

# Liste globale pour stocker les métriques des différents chunks
global_metrics = []

# Fonction pour calculer les métriques d'évaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    rcall = f1_score(labels, predictions, average='macro')
    precision = f1_score(labels, predictions, average='micro')

    return {
        'accuracy': acc,
        'f1': f1,
        
        
    }



# Définition du classifieur BERT personnalisé
class BertTextClassifier(TextClassificationModel):
    def __init__(self, config_path):
        super().__init__(config_path)
        # Chargement des paramètres depuis le fichier de configuration
        self.model_name = self.config['model']['bert']['model_name']
        self.num_labels = self.config['model']['bert']['num_labels']

        # Chargement du modèle BERT pré-entraîné avec un nombre de classes défini
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )

        # Préparation de l’ingénierie des features et initialisation de Wandb
        self.feature_engineer = FeatureEngineer(config_path)
        self.output_dir = self.config['model']['bert']['output_dir']
        self.wandb_logger = WandbLogger("config/wandb_config.yaml")

    # Méthode d’entraînement sur un chunk de données
    def train(self, df, chunk_num):
        print(f"\n Entraînement sur chunk {chunk_num}")

        # Division des données en jeu d’entraînement et de validation
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'].tolist(),
            df['label'].tolist(),
            test_size=self.config['model']['bert']['test_size'],
            random_state=self.config['model']['bert']['random_state']
        )

        # Encodage des textes via BERT tokenizer
        train_encodings = self.feature_engineer.transform_bert(train_texts)
        val_encodings = self.feature_engineer.transform_bert(val_texts)

        # Création des datasets PyTorch
        train_dataset = SentimentDataset(train_encodings, train_labels)
        val_dataset = SentimentDataset(val_encodings, val_labels)

        # Configuration des paramètres d'entraînement
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config['model']['bert']['epochs'],
            per_device_train_batch_size=self.config['model']['bert']['batch_size'],
            per_device_eval_batch_size=self.config['model']['bert']['batch_size'],
            warmup_steps=self.config['model']['bert']['warmup_steps'],
            weight_decay=self.config['model']['bert']['weight_decay'],
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir='./logs',
            logging_steps=100,
            save_total_limit=self.config['model']['bert']['save_total_limit'],
            report_to="wandb"  # Activation de Weights & Biases
        )

        # Création du trainer Hugging Face
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[WandbCallback()],  # Callback W&B pour le suivi en ligne
        )

        # Vérifie s’il existe un checkpoint pour reprendre l’entraînement
        checkpoint_path = os.path.join(self.output_dir, "checkpoint-last")
        resume = os.path.exists(checkpoint_path)

        if resume:
            print(f"Reprise depuis le checkpoint : {checkpoint_path}")
            trainer.train(resume_from_checkpoint=checkpoint_path)
        else:
            print("Entraînement initial")
            trainer.train()

        # Évaluation après l'entraînement
        eval_results = trainer.evaluate()
        eval_results['chunk'] = chunk_num
        global_metrics.append(eval_results)

        # Log des résultats dans W&B
        self.wandb_logger.log_metrics(eval_results)

        # Sauvegarde du modèle et du tokenizer pour les prochaines itérations
        self.model.save_pretrained(self.output_dir)
        self.feature_engineer.bert_tokenizer.save_pretrained(self.output_dir)

        # Fin de session W&B
        self.wandb_logger.finish()
    
    # Méthode pour sauvegarder le modèle à un chemin spécifique
    def save_model(self, path: str):
        """Save BERT model and tokenizer."""
        self.model.save_pretrained(path)
        self.feature_engineer.bert_tokenizer.save_pretrained(path)
        self.wandb_logger.log_model(path, "bert_model")
        print(f"Modèle sauvegardé à l'emplacement : {path}")
