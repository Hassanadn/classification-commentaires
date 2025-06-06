# from typing import Iterator, Tuple
# from src.utils.helper_functions import timer_decorator
# import yaml
# import os
# import re
# import pandas as pd
# import emoji

# class DataLoader:
#     """Class to load and preprocess data in chunks."""
    
#     def __init__(self, config_path: str):
#         # """Initialize with config file path."""
#         # if not os.path.exists(config_path):
#         #     raise FileNotFoundError(f"Le fichier de configuration {config_path} n'existe pas.")
#         with open(config_path, 'r') as file:
#             self.config = yaml.safe_load(file)
#         # self.raw_path = self.config['data']['raw_path']
#         # if not os.path.exists(self.raw_path):
#         #     raise FileNotFoundError(f"Le fichier de données {self.raw_path} n'existe pas à l'emplacement : {os.path.abspath(self.raw_path)}")
#         # self.processed_path = self.config['data']['processed_path']
#         # self.chunk_size = self.config.get('data', {}).get('chunk_size', 10000)

#         self.raw_path = self.config['data']['raw_path']
#         self.processed_path = self.config['data']['processed_path']
#         self.chunk_size = self.config['data']['chunk_size']


#     @timer_decorator
#     def load_raw_data(self, chunk_size: int = None) -> Iterator[pd.DataFrame]:
#         """Load raw data from CSV in chunks using a generator."""
#         chunk_size = chunk_size or self.chunk_size
#         for chunk in pd.read_csv(self.raw_path, chunksize=chunk_size):
#             yield chunk
 
#     # def preprocess_data(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
#     #     """Basic preprocessing: remove NaN, lowercase text."""
#     #     print("Prétraitement d'un chunk de données")
#     #     # Supprimer les NaN
#     #     df_chunk = df_chunk.dropna(subset=['text', 'label'])
#     #     df_chunk['text'] = df_chunk['text'].astype(str)
#     #     # Nettoyage avancé
#     #     df_chunk['text'] = df_chunk['text'].apply(lambda x: re.sub(r'http\S+|www\S+|@\w+|#\w+', '', x))  # Supprimer URLs, mentions, hashtags
#     #     df_chunk['text'] = df_chunk['text'].apply(lambda x: emoji.replace_emoji(x, replace=''))  # Supprimer emojis
#     #     df_chunk['text'] = df_chunk['text'].apply(lambda x: re.sub(r'[^\w\s!?]', '', x.lower()))  # Minuscules, garder ! et ?
#     #     # Supprimer aberrants
#     #     df_chunk = df_chunk[df_chunk['text'].str.len() > 10]  # Supprimer textes < 10 caractères
#     #     df_chunk = df_chunk[df_chunk['text'].str.split().str.len() > 2]  # Supprimer < 3 mots
#     #     return df_chunk
    
#     def preprocess_data(self, df_chunk):
#         print("Colonnes du chunk :", df_chunk.columns)
#         df_chunk = df_chunk.dropna(subset=['text', 'label'])
#         df_chunk['text'] = df_chunk['text'].astype(str)
#         df_chunk['text'] = df_chunk['text'].apply(lambda x: re.sub(r'http\S+|www\S+|@\w+|#\w+', '', x))
#         df_chunk['text'] = df_chunk['text'].apply(lambda x: emoji.replace_emoji(x, replace=''))
#         df_chunk['text'] = df_chunk['text'].apply(lambda x: re.sub(r'[^\w\s!?]', '', x.lower()))
#         df_chunk = df_chunk[df_chunk['text'].str.len() > 10]
#         df_chunk = df_chunk[df_chunk['text'].str.split().str.len() > 2]
#         if df_chunk['label'].min() == 1:
#             df_chunk['label'] = df_chunk['label'] - 1
#         return df_chunk


#     @timer_decorator
#     def process_and_save_chunks(self):
#         """Process chunks and save to processed path."""
#         first_chunk = True
#         os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)  # Créer le dossier si nécessaire
#         for chunk in self.load_raw_data():
#             processed_chunk = self.preprocess_data(chunk)
#             mode = 'w' if first_chunk else 'a'
#             processed_chunk.to_csv(self.processed_path, 
#                                 mode=mode, 
#                                 index=False, 
#                                 header=first_chunk)
#             first_chunk = False
    

#     # def data_generator(self, batch_size: int = 32) -> Iterator[Tuple[list, list]]:
#     #     """Generator to yield batches of data from processed CSV for training."""
#     #     if not os.path.exists(self.processed_path):
#     #         raise FileNotFoundError(f"Le fichier traité {self.processed_path} n'existe pas. Exécutez process_and_save_chunks d'abord.")
#     #     for chunk in pd.read_csv(self.processed_path, chunksize=batch_size):
#     #         yield chunk['text'].tolist(), chunk['label'].tolist()

#     def data_generator(self):
#             for chunk in pd.read_csv(self.raw_path, chunksize=self.chunk_size):
#                 yield self.preprocess_data(chunk)

from typing import Iterator, Tuple
from src.utils.helper_functions import timer_decorator
import yaml
import os
import re
import pandas as pd
import emoji

class DataLoader:
    """Class to load and preprocess data in chunks."""
    
    def __init__(self, config_path: str):
        # Chargement de la configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.raw_path = self.config['data']['raw_path']
        self.processed_path = self.config['data']['processed_path']
        self.chunk_size = self.config['data']['chunk_size']

    @timer_decorator
    def load_raw_data(self, chunk_size: int = None) -> Iterator[pd.DataFrame]:
        """Charge les données brutes CSV en chunks avec un générateur."""
        chunk_size = chunk_size or self.chunk_size
        for chunk in pd.read_csv(self.raw_path, chunksize=chunk_size):
            yield chunk
 
    def preprocess_data(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        """Prétraitement de base : nettoyage du texte, suppression des NaN, etc."""
        print("Colonnes du chunk :", df_chunk.columns)

        # Supprimer lignes avec texte ou label manquants
        df_chunk = df_chunk.dropna(subset=['text', 'label'])

        # Convertir en chaîne de caractères
        df_chunk['text'] = df_chunk['text'].astype(str)

        # Nettoyage : supprimer URLs, mentions, hashtags
        df_chunk['text'] = df_chunk['text'].apply(lambda x: re.sub(r'http\S+|www\S+|@\w+|#\w+', '', x))

        # Supprimer les emojis
        df_chunk['text'] = df_chunk['text'].apply(lambda x: emoji.replace_emoji(x, replace=''))

        # Mettre en minuscules et retirer caractères spéciaux sauf ! et ?
        df_chunk['text'] = df_chunk['text'].apply(lambda x: re.sub(r'[^\w\s!?]', '', x.lower()))

        # Supprimer textes trop courts ou aberrants
        df_chunk = df_chunk[df_chunk['text'].str.len() > 10]
        df_chunk = df_chunk[df_chunk['text'].str.split().str.len() > 2]

        # Ajuster les labels (si le minimum est 1)
        if df_chunk['label'].min() == 1:
            df_chunk['label'] = df_chunk['label'] - 1

        return df_chunk

    @timer_decorator
    def process_and_save_chunks(self):
        """Traite les chunks et les enregistre dans un fichier CSV final."""
        first_chunk = True
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)

        for chunk in self.load_raw_data():
            processed_chunk = self.preprocess_data(chunk)

            mode = 'w' if first_chunk else 'a'
            processed_chunk.to_csv(
                self.processed_path,
                mode=mode,
                index=False,
                header=first_chunk
            )
            first_chunk = False

    def data_generator(self) -> Iterator[pd.DataFrame]:
        """Générateur de données prétraitées pour itérer chunk par chunk."""
        for chunk in self.load_raw_data():
            yield self.preprocess_data(chunk)
