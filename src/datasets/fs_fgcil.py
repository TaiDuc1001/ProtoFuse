import random
from typing import Dict, List, Optional, Tuple

from src.datasets.fs_fgvc import FewShotFGVCDataset


class FGCILTaskSplitter:
    def __init__(self, train_dataset, test_dataset,
                 num_base_classes, classes_per_task, kshot=-1, seed=42):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.num_base_classes = num_base_classes
        self.classes_per_task = classes_per_task
        self.kshot = kshot
        self.seed = seed

        if isinstance(train_dataset, FewShotFGVCDataset):
            all_classes = sorted(train_dataset.class_to_idx.values())
        else:
            all_classes = sorted(set(c for _, c in train_dataset.samples))

        rng = random.Random(seed)
        self._class_order = list(all_classes)
        rng.shuffle(self._class_order)

        self._base_classes = self._class_order[:num_base_classes]
        remaining = self._class_order[num_base_classes:]

        self._incremental_tasks = []
        for i in range(0, len(remaining), classes_per_task):
            task_classes = remaining[i:i + classes_per_task]
            if task_classes:
                self._incremental_tasks.append(task_classes)

        self._train_by_class = self._index_by_class(train_dataset)
        self._test_by_class = self._index_by_class(test_dataset)

    @staticmethod
    def _index_by_class(dataset):
        by_class = {}
        samples = dataset.samples if hasattr(dataset, 'samples') else dataset.dataset.samples
        for idx, (_, class_idx) in enumerate(samples):
            if class_idx not in by_class:
                by_class[class_idx] = []
            by_class[class_idx].append(idx)
        return by_class

    @property
    def num_tasks(self) -> int:
        return 1 + len(self._incremental_tasks)

    @property
    def total_classes(self) -> int:
        return len(self._class_order)

    def class_order(self) -> List[int]:
        return list(self._class_order)

    def _get_task_classes(self, task_id) -> List[int]:
        if task_id == 0:
            return list(self._base_classes)
        inc_idx = task_id - 1
        if inc_idx < len(self._incremental_tasks):
            return list(self._incremental_tasks[inc_idx])
        raise IndexError(f"task_id {task_id} out of range (num_tasks={self.num_tasks})")

    def get_task(self, task_id) -> Dict:
        task_classes = self._get_task_classes(task_id)
        rng = random.Random(self.seed + task_id)

        train_indices = []
        unlabeled_indices = []
        for cls in task_classes:
            indices = list(self._train_by_class.get(cls, []))
            indices.sort()
            rng.shuffle(indices)
            if self.kshot > 0:
                train_indices.extend(indices[:self.kshot])
                unlabeled_indices.extend(indices[self.kshot:])
            else:
                train_indices.extend(indices)

        test_indices = []
        for cls in task_classes:
            test_indices.extend(self._test_by_class.get(cls, []))

        return {
            'task_id': task_id,
            'classes': task_classes,
            'num_classes': len(task_classes),
            'train_indices': train_indices,
            'unlabeled_indices': unlabeled_indices,
            'test_indices': test_indices,
        }

    def get_cumulative_task(self, task_id) -> Dict:
        all_classes = []
        all_train = []
        all_unlabeled = []
        all_test = []

        for t in range(task_id + 1):
            task_data = self.get_task(t)
            all_classes.extend(task_data['classes'])
            all_train.extend(task_data['train_indices'])
            all_unlabeled.extend(task_data['unlabeled_indices'])
            all_test.extend(task_data['test_indices'])

        return {
            'task_id': task_id,
            'classes': all_classes,
            'num_classes': len(all_classes),
            'train_indices': all_train,
            'unlabeled_indices': all_unlabeled,
            'test_indices': all_test,
        }
