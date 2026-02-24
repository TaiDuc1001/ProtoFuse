from abc import ABC, abstractmethod
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class BaseStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def score(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def select(self, scores: Dict[int, float], budget: int) -> List[int]:
        sorted_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        return sorted_indices[:budget]

    @staticmethod
    def _collect_logits(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> torch.Tensor:
        model.eval()
        all_logits = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(device)
                logits = model(images)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                all_logits.append(logits.cpu())
        return torch.cat(all_logits, dim=0)
