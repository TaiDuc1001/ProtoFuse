from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseStrategy


class EntropyStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "entropy"

    def score(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, Any]:
        logits = self._collect_logits(model, dataloader, device)
        scores = self.score_logits(logits)
        return {
            "scores": scores,
            "logits": logits,
            "strategy": self.name,
        }

    def score_logits(self, logits: torch.Tensor) -> Dict[int, float]:
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropies = -(probs * log_probs).sum(dim=-1)
        return {i: entropies[i].item() for i in range(len(entropies))}
