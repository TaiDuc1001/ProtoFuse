import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import random


class WeakAugment:
    def __init__(self, clip_mean=None, clip_std=None):
        if clip_mean is None:
            clip_mean = [0.48145466, 0.4578275, 0.40821073]
        if clip_std is None:
            clip_std = [0.26862954, 0.26130258, 0.27577711]
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
        ])
    
    def __call__(self, img):
        return self.transform(img)


class StrongAugment:
    def __init__(self, clip_mean=None, clip_std=None):
        if clip_mean is None:
            clip_mean = [0.48145466, 0.4578275, 0.40821073]
        if clip_std is None:
            clip_std = [0.26862954, 0.26130258, 0.27577711]
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=clip_mean, std=clip_std),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ])
    
    def __call__(self, img):
        return self.transform(img)


class FixMatchTransform:
    def __init__(self, clip_mean=None, clip_std=None):
        self.weak = WeakAugment(clip_mean, clip_std)
        self.strong = StrongAugment(clip_mean, clip_std)
    
    def __call__(self, img):
        return self.weak(img), self.strong(img)


class FixMatchDataset(Dataset):
    def __init__(self, base_dataset, indices, clip_mean=None, clip_std=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = FixMatchTransform(clip_mean, clip_std)
        from PIL import Image
        self.Image = Image
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, label = self.base_dataset.samples[real_idx]
        img = self.Image.open(path).convert('RGB')
        weak_img, strong_img = self.transform(img)
        return weak_img, strong_img, label


class FixMatchMixin:
    confidence: float = 0.95
    wu: float = 1.0
    unlabeled_batch_size: int = 56
    
    def _init_fixmatch_config(self):
        fixmatch_cfg = self.config.get('fixmatch', {})
        if hasattr(fixmatch_cfg, 'get'):
            self.confidence = float(fixmatch_cfg.get('confidence', 0.95))
            self.wu = float(fixmatch_cfg.get('wu', 1.0))
            self.unlabeled_batch_size = int(fixmatch_cfg.get('unlabeled_batch_size', self.batch_size * 7))
        else:
            self.confidence = 0.95
            self.wu = 1.0
            self.unlabeled_batch_size = self.batch_size * 7
    
    def _create_fixmatch_loaders(self):
        labeled_ds = Subset(self.dataset, self.labeled_indices)
        labeled_loader = DataLoader(
            labeled_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            drop_last=True
        )
        
        unlabeled_ds = FixMatchDataset(
            self.dataset, 
            self.unlabeled_indices,
            self.clip_mean,
            self.clip_std
        )
        unlabeled_loader = DataLoader(
            unlabeled_ds,
            batch_size=self.unlabeled_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )
        
        return labeled_loader, unlabeled_loader
    
    def _fixmatch_train_step(self, labeled_batch, unlabeled_batch):
        images_l, labels = labeled_batch
        weak_u, strong_u, _ = unlabeled_batch
        
        images_l = images_l.to(self.device)
        labels = labels.to(self.device)
        weak_u = weak_u.to(self.device)
        strong_u = strong_u.to(self.device)
        
        self.trainer.model.train()
        
        logits_l = self.trainer.model(images_l)
        if isinstance(logits_l, tuple):
            logits_l = logits_l[0]
        loss_xe = F.cross_entropy(logits_l, labels)
        
        with torch.no_grad():
            logits_weak = self.trainer.model(weak_u)
            if isinstance(logits_weak, tuple):
                logits_weak = logits_weak[0]
            probs = F.softmax(logits_weak, dim=1)
            max_probs, pseudo_labels = probs.max(dim=1)
            mask = max_probs >= self.confidence
        
        logits_strong = self.trainer.model(strong_u)
        if isinstance(logits_strong, tuple):
            logits_strong = logits_strong[0]
        
        loss_u = F.cross_entropy(logits_strong, pseudo_labels, reduction='none')
        loss_u = (loss_u * mask.float()).mean()
        
        total_loss = loss_xe + self.wu * loss_u
        
        self.trainer.optimizer.zero_grad()
        total_loss.backward()
        self.trainer.optimizer.step()
        
        with torch.no_grad():
            _, predicted = torch.max(logits_l, 1)
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / labels.size(0)
        
        mask_ratio = mask.float().mean().item()
        
        return {
            'loss': total_loss.item(),
            'loss_xe': loss_xe.item(),
            'loss_u': loss_u.item(),
            'accuracy': accuracy,
            'mask_ratio': mask_ratio
        }
