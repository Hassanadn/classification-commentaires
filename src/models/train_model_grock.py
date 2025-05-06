import sys
import os
import warnings
import yaml
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import pandas as pd
import numpy as np
import logging
import torch
from torch.utils.data import Dataset
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from src.data.load_data import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.utils.helper_functions import timer_decorator

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ignorer les avertissements spécifiques
warnings.filterwarnings("ignore", category=UserWarning, module="_distutils_hack")

# Ajouter le chemin du projet au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Définir les classes
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TextClassificationModel:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.data_loader = DataLoader(config_path)
    
    def save_model(self, model, path: str):
        raise NotImplementedError
    
    def train(self, df: pd.DataFrame):
        raise NotImplementedError

class RandomForestTextClassifier(TextClassificationModel):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)

    @timer_decorator
    def train(self, df: pd.DataFrame):
        logging.info(f"Entraînement sur un lot de {len(df)} lignes")
        texts, labels = df['text'].tolist(), df['label'].tolist()
        
        # Vérifier les classes présentes
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            logging.warning(f"Lot ignoré : contient seulement les labels {unique_labels}")
            return
        
        # Ajuster TF-IDF sur un échantillon si trop grand
        sample_size = min(10000, len(texts))
        self.feature_engineer.fit_tfidf(texts[:sample_size])
        
        # Calculer les poids des classes sur un échantillon
        sample_labels = labels[:sample_size]
        class_weights = compute_class_weight('balanced', classes=np.array([1, 2]), y=sample_labels)
        class_weight_dict = {1: class_weights[0], 2: class_weights[1]}
        
        # Entraînement par lots
        batch_size = 10000
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_labels = np.array(labels[i:i + batch_size])
            X_batch = self.feature_engineer.transform_tfidf(batch_texts)
            sample_weights = np.array([class_weight_dict[label] for label in batch_labels])
            self.model.partial_fit(X_batch, batch_labels, classes=np.array([1, 2]), sample_weight=sample_weights)
            logging.info(f"Lot {i//batch_size + 1} entraîné")
        
        # Évaluation sur un sous-ensemble
        eval_texts = texts[:1000]
        eval_labels = labels[:1000]
        X_eval = self.feature_engineer.transform_tfidf(eval_texts)
        predictions = self.model.predict(X_eval)
        accuracy = accuracy_score(eval_labels, predictions)
        logging.info(f"Précision sur le sous-ensemble d'évaluation : {accuracy:.4f}")
        
        # Créer ou récupérer l'expérience MLflow
        experiment_name = "TextClassificationExperiment"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_param("model_type", "SGDClassifier")
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(self.model, "sgd_model")
            logging.info("Métriques et modèle loggés dans MLflow")

    def save_model(self, path):
        import pickle
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'vectorizer': self.feature_engineer.tfidf}, f)
        logging.info(f"Modèle sauvegardé à {path}")

class BertTextClassifier(TextClassificationModel):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        self.model_name = self.config['model']['bert']['model_name']
        self.num_labels = self.config['model']['bert']['num_labels']
        self.max_length = self.config['model']['bert']['max_length']
        self.feature_engineer = FeatureEngineer(config_path)  # Utilisation de FeatureEngineer
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logging.info(f"Modèle BERT chargé ({self.model_name}) sur {self.device}")

    def train(self, df):
        logging.info(f"Entraînement BERT sur {len(df)} lignes")
        # Ajuster les labels si nécessaire
        labels = df['label'].values
        if labels.min() == 1:
            logging.info("Ajustement des labels de 1,2 à 0,1")
            labels = labels - 1
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'].values, labels, test_size=0.2, random_state=42
        )
        # Utiliser FeatureEngineer pour la tokenisation
        train_encodings = self.feature_engineer.transform_bert(train_texts.tolist(), max_length=self.max_length)
        val_encodings = self.feature_engineer.transform_bert(val_texts.tolist(), max_length=self.max_length)
        train_dataset = SentimentDataset(train_texts, train_labels, self.feature_engineer.bert_tokenizer, self.max_length)
        val_dataset = SentimentDataset(val_texts, val_labels, self.feature_engineer.bert_tokenizer, self.max_length)

        training_args = TrainingArguments(
            output_dir='/results',
            num_train_epochs=self.config['model']['bert']['epochs'],
            per_device_train_batch_size=self.config['model']['bert']['batch_size'],
            per_device_eval_batch_size=self.config['model']['bert']['batch_size'],
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='/logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            dataloader_pin_memory=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=lambda p: {'accuracy': (p.predictions.argmax(-1) == p.label_ids).mean()}
        )

        # MLflow
        experiment_name = "TextClassificationExperiment"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_param("model_type", "BERT")
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("epochs", self.config['model']['bert']['epochs'])
            mlflow.log_param("batch_size", self.config['model']['bert']['batch_size'])
            mlflow.log_param("max_length", self.max_length)

            # Entraînement
            trainer.train()

            # Évaluation
            predictions = trainer.predict(val_dataset).predictions.argmax(-1)
            accuracy = accuracy_score(val_labels, predictions)
            mlflow.log_metric("val_accuracy", accuracy)
            mlflow.pytorch.log_model(self.model, "bert_model")
            logging.info(f"Précision sur validation : {accuracy:.4f}")

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.feature_engineer.bert_tokenizer.save_pretrained(output_dir)
        logging.info(f"Modèle BERT sauvegardé à {output_dir}")

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
    data_loader = DataLoader(config_path)
    
    try:
        # logging.info("Début du prétraitement des données")
        # data_loader.process_and_save_chunks()
        
        # logging.info("Début de l'entraînement Random Forest (SGDClassifier)")
        # rf_classifier = RandomForestTextClassifier(config_path)
        # for texts, labels in data_loader.data_generator(batch_size=10000):
        #     df_chunk = pd.DataFrame({'text': texts, 'label': labels})
        #     rf_classifier.train(df_chunk)
        
        # rf_classifier.save_model("models/random_forest_v1.pkl")
        
        logging.info("Début de l'entraînement BERT")
        bert_classifier = BertTextClassifier(config_path)
        for texts, labels in data_loader.data_generator(batch_size=10000):
            df_chunk = pd.DataFrame({'text': texts, 'label': labels})
            bert_classifier.train(df_chunk)
    
        bert_classifier.save_model("models/bert_v1")
        logging.info("Entraînement terminé")
    except Exception as e:
        logging.error(f"Erreur pendant l'exécution : {str(e)}", exc_info=True)
        raise