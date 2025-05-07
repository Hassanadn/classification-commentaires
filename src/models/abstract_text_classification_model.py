from abc import ABC, abstractmethod
import pandas as pd
import yaml
from  src.data.load_data import DataLoader

class TextClassificationModel:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.data_loader = DataLoader(config_path)
    
    @abstractmethod
    def save_model(self, model, path: str):
        raise NotImplementedError
    
    @abstractmethod
    def train(self, df: pd.DataFrame):
        raise NotImplementedError