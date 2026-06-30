import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from torchvision.datasets import ImageFolder
from utils import fast_image_folder


class FewShotFGVCDataset:
    def __init__(self, root, split='train', transform=None, kshot=-1, seed=42):
        self.root = root
        self.split = split
        self.seed = seed
        self.kshot = kshot

        split_path = os.path.join(root, split)
        if not os.path.isdir(split_path):
            split_path = root

        self.dataset = fast_image_folder(split_path, transform=transform)
        self.samples = self.dataset.samples
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.transform = transform

        self._samples_by_class = None

    @property
    def classnames(self) -> List[str]:
        return list(self.classes)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def get_samples_by_class(self) -> Dict[int, List[int]]:
        if self._samples_by_class is None:
            self._samples_by_class = defaultdict(list)
            for idx, (_, class_idx) in enumerate(self.samples):
                self._samples_by_class[class_idx].append(idx)
        return dict(self._samples_by_class)

    def get_kshot_indices(self, kshot=None, seed=None) -> Tuple[List[int], List[int]]:
        if kshot is None:
            kshot = self.kshot
        if seed is None:
            seed = self.seed

        samples_by_class = self.get_samples_by_class()
        rng = random.Random(seed)

        labeled = []
        unlabeled = []

        for class_idx in sorted(samples_by_class.keys()):
            indices = list(samples_by_class[class_idx])
            indices.sort()
            rng.shuffle(indices)

            if kshot > 0:
                labeled.extend(indices[:kshot])
                unlabeled.extend(indices[kshot:])
            else:
                labeled.extend(indices)

        return labeled, unlabeled

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.dataset[idx]
