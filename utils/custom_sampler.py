import numpy as np

from torch.utils.data import Sampler


class CustomSampler(Sampler):
    def __init__(self, dataset, batch_size=32):
        self.dataset = dataset
        self.labels = self.dataset.labels
        self.classes = np.unique(self.labels)
        self.N = len(self.classes)
        self.m_per_class = batch_size // self.N

        self.S = {c: np.where(self.labels == c)[0].tolist() for c in self.classes}
        self.C = {c: len(self.S[c]) for c in self.classes}
        self.c_min = min(self.C.values())
        self.K = self.c_min // self.m_per_class

    def __len__(self):
        return self.K

    def __iter__(self):
        S_work = {c: list(self.S[c]) for c in self.classes}

        for c in self.classes:
            np.random.shuffle(S_work[c])

        for i in range(self.K):
            batch = []
            for c in self.classes:
                if len(S_work[c]) < self.m_per_class:
                    remaining = S_work[c]
                    S_work[c] = list(self.S[c])
                    np.random.shuffle(S_work[c])
                    for used_idx in remaining:
                        if used_idx in S_work[c]:
                            S_work[c].remove(used_idx)
                    if len(S_work[c]) < self.m_per_class:
                        S_work[c] = list(self.S[c])
                        np.random.shuffle(S_work[c])
                chosen = S_work[c][: self.m_per_class]
                batch.extend(chosen)
                S_work[c] = S_work[c][self.m_per_class :]
            yield batch
