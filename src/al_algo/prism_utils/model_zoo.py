from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import logger, ConfigNode, load_clip_to_cpu, coerce_to_str
from src.models.apt import ImageEncoder


class ModelWrapper(ABC):
    @property
    @abstractmethod
    def embed_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_logits(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_logits_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings

    def classify_from_embeddings(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)
        return embeddings @ prototypes.T

    def embed_dataset(
        self,
        dataloader: DataLoader,
        device: torch.device,
    ) -> torch.Tensor:
        all_embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(device)
                emb = self.embed(images)
                all_embeddings.append(emb.cpu())
        return torch.cat(all_embeddings, dim=0)


class CLIPModelWrapper(ModelWrapper):
    def __init__(self, backbone: str, weights_dir: str, device: torch.device):
        self._device = device
        self._backbone = backbone
        clip_model = load_clip_to_cpu(backbone)
        clip_model.float()
        self._encoder = ImageEncoder(clip_model).to(device)
        self._encoder.eval()
        for p in self._encoder.parameters():
            p.requires_grad_(False)
        self._logit_scale = clip_model.logit_scale.to(device)
        test_input = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32)
        with torch.no_grad():
            _, feat = self._encoder(test_input)
        self._embed_dim = feat.shape[-1]

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, global_feat = self._encoder(images)
        return global_feat

    def get_logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.embed(images)


class DINOv2ModelWrapper(ModelWrapper):
    def __init__(self, variant: str, device: torch.device):
        self._device = device
        self._model = torch.hub.load(
            "facebookresearch/dinov2",
            variant,
            pretrained=True,
        ).to(device)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        test_input = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32)
        with torch.no_grad():
            out = self._model(test_input)
        self._embed_dim = out.shape[-1]

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._model(images)

    def get_logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.embed(images)


class ResNet50ModelWrapper(ModelWrapper):
    def __init__(self, device: torch.device):
        import torchvision.models as tv_models

        self._device = device
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
        full_model = tv_models.resnet50(weights=weights)
        self._features = nn.Sequential(
            full_model.conv1,
            full_model.bn1,
            full_model.relu,
            full_model.maxpool,
            full_model.layer1,
            full_model.layer2,
            full_model.layer3,
            full_model.layer4,
            full_model.avgpool,
        ).to(device)
        self._features.eval()
        for p in self._features.parameters():
            p.requires_grad_(False)
        self._classifier = full_model.fc.to(device)
        self._classifier.eval()
        for p in self._classifier.parameters():
            p.requires_grad_(False)
        self._embed_dim = full_model.fc.in_features

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self._features(images)
            return feat.flatten(1)

    def get_logits(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.embed(images)
            return self._classifier(feat)

    def get_logits_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._classifier(embeddings)


_MODEL_CACHE: Dict[str, ModelWrapper] = {}


def _cache_key(model_cfg: dict) -> str:
    mtype = model_cfg.get("type", "clip")
    if mtype == "clip":
        return f"clip_{model_cfg.get('backbone', 'ViT-B/16')}"
    elif mtype == "dinov2":
        return f"dinov2_{model_cfg.get('variant', 'dinov2_vitb14')}"
    elif mtype == "resnet50":
        return "resnet50"
    return f"{mtype}_{id(model_cfg)}"


def load_models_from_config(
    config: ConfigNode,
    device: torch.device,
) -> List[ModelWrapper]:
    global _MODEL_CACHE

    prism_cfg = config.get("prism", ConfigNode())
    model_cfgs = prism_cfg.get("models", [])

    if not model_cfgs:
        model_cfgs = [{"type": "clip", "backbone": "ViT-B/16", "weights_dir": "models"}]
        logger.warning("No models configured for PRISM, defaulting to CLIP ViT-B/16")

    models = []
    for mcfg in model_cfgs:
        if isinstance(mcfg, ConfigNode):
            mcfg_dict = mcfg.to_dict() if hasattr(mcfg, "to_dict") else dict(mcfg)
        else:
            mcfg_dict = dict(mcfg)

        key = _cache_key(mcfg_dict)
        if key in _MODEL_CACHE:
            logger.info(f"Reusing cached model: {key}")
            models.append(_MODEL_CACHE[key])
            continue

        mtype = mcfg_dict.get("type", "clip")

        if mtype == "clip":
            backbone = mcfg_dict.get("backbone", "ViT-B/16")
            weights_dir = mcfg_dict.get("weights_dir", "models")
            logger.info(f"Loading CLIP model: {backbone} from {weights_dir}")
            wrapper = CLIPModelWrapper(backbone, weights_dir, device)

        elif mtype == "dinov2":
            variant = mcfg_dict.get("variant", "dinov2_vitb14")
            logger.info(f"Loading DINOv2 model: {variant} (auto-downloads via torch.hub)")
            wrapper = DINOv2ModelWrapper(variant, device)

        elif mtype == "resnet50":
            logger.info("Loading ResNet50 (torchvision IMAGENET1K_V2)")
            wrapper = ResNet50ModelWrapper(device)

        else:
            raise ValueError(f"Unknown model type: {mtype}")

        _MODEL_CACHE[key] = wrapper
        models.append(wrapper)
        logger.info(f"Loaded {mtype} → embed_dim={wrapper.embed_dim}")

    return models


def clear_model_cache():
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
