import os
import time
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime
import requests
import json

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsLogger:
    """
    Classe pour logger les métriques vers InfluxDB et Prometheus
    """
    def __init__(self):
        """Initialise les connexions aux systèmes de métriques"""
        # Configuration InfluxDB
        self.influxdb_host = os.environ.get('INFLUXDB_HOST', 'influxdb')
        self.influxdb_port = os.environ.get('INFLUXDB_PORT', '8086')
        self.influxdb_database = os.environ.get('INFLUXDB_DATABASE', 'metrics')
        self.influxdb_user = os.environ.get('INFLUXDB_USER', 'admin')
        self.influxdb_password = os.environ.get('INFLUXDB_PASSWORD', 'admin123')
        
        # URL pour l'API InfluxDB
        self.influxdb_url = f"http://{self.influxdb_host}:{self.influxdb_port}/write"
        
        # Configuration Prometheus
        self.prometheus_pushgateway = os.environ.get('PROMETHEUS_PUSHGATEWAY', 'prometheus:9091')
        self.prometheus_url = f"http://{self.prometheus_pushgateway}/metrics/job/model_training"
        
    def log_to_influxdb(self, 
                        measurement: str, 
                        tags: Dict[str, str], 
                        fields: Dict[str, Union[float, int, str]],
                        timestamp: Optional[int] = None) -> bool:
        """
        Envoie des métriques à InfluxDB
        
        Args:
            measurement: Nom de la mesure (table)
            tags: Dictionnaire de tags pour les dimensions
            fields: Dictionnaire de champs avec les valeurs à enregistrer
            timestamp: Timestamp en nanosecondes (optionnel)
            
        Returns:
            True si l'envoi a réussi, False sinon
        """
        try:
            # Construire la ligne de protocole InfluxDB
            tag_str = ",".join([f"{k}={v}" for k, v in tags.items()])
            field_str = ",".join([f"{k}={v}" if isinstance(v, (int, float)) else f'{k}="{v}"' 
                                for k, v in fields.items()])
            
            line = f"{measurement},{tag_str} {field_str}"
            if timestamp:
                line += f" {timestamp}"
                
            # Paramètres pour la requête
            params = {
                'db': self.influxdb_database,
                'u': self.influxdb_user,
                'p': self.influxdb_password
            }
            
            # Envoyer à InfluxDB
            response = requests.post(
                self.influxdb_url,
                params=params,
                data=line
            )
            
            if response.status_code == 204:
                logger.info(f"Métriques envoyées à InfluxDB: {measurement}")
                return True
            else:
                logger.error(f"Erreur lors de l'envoi des métriques à InfluxDB: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Exception lors de l'envoi des métriques à InfluxDB: {e}")
            return False
            
    def log_to_prometheus(self, metrics: Dict[str, Dict[str, Any]]) -> bool:
        """
        Envoie des métriques à Prometheus via Pushgateway
        
        Args:
            metrics: Dictionnaire de métriques avec leurs labels et valeurs
                Format: {
                    'metric_name': {
                        'value': 0.95,
                        'labels': {'model': 'bert', 'version': '1.0'}
                    }
                }
                
        Returns:
            True si l'envoi a réussi, False sinon
        """
        try:
            # Construire le format texte pour Prometheus
            lines = []
            
            for metric_name, metric_data in metrics.items():
                value = metric_data.get('value')
                labels = metric_data.get('labels', {})
                
                if labels:
                    label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
                    line = f"{metric_name}{{{label_str}}} {value}"
                else:
                    line = f"{metric_name} {value}"
                    
                lines.append(line)
                
            metric_data = "\n".join(lines)
            
            # Envoyer à Prometheus Pushgateway
            response = requests.post(
                self.prometheus_url,
                data=metric_data,
                headers={'Content-Type': 'text/plain'}
            )
            
            if response.status_code == 200:
                logger.info("Métriques envoyées à Prometheus Pushgateway")
                return True
            else:
                logger.error(f"Erreur lors de l'envoi des métriques à Prometheus: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Exception lors de l'envoi des métriques à Prometheus: {e}")
            return False
            
    def log_model_metrics(self, model_name: str, metrics: Dict[str, float], version: str = "v1") -> None:
        """
        Enregistre les métriques d'un modèle dans InfluxDB et Prometheus
        
        Args:
            model_name: Nom du modèle
            metrics: Dictionnaire de métriques avec leurs valeurs
            version: Version du modèle
        """
        # Préparation pour InfluxDB
        tags = {
            'model': model_name,
            'version': version
        }
        
        self.log_to_influxdb(
            measurement='model_metrics',
            tags=tags,
            fields=metrics
        )
        
        # Préparation pour Prometheus
        prom_metrics = {}
        for metric_name, value in metrics.items():
            prom_metrics[f"model_{metric_name}"] = {
                'value': value,
                'labels': {
                    'model': model_name,
                    'version': version
                }
            }
            
        self.log_to_prometheus(prom_metrics)