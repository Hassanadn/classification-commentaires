import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any

class SentimentDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
