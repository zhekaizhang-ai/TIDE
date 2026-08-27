import math
import random

from torch.utils.data import Sampler


class DomainBalancedSampler(Sampler):
    """
    Batch sampler that draws the same number of samples from each domain.

    Epoch length is anchored to the geometric mean of domain sizes, so large
    domains are not truncated to the smallest domain and small domains cycle
    (reshuffle and repeat) within each epoch.

    Pass instances of this class to DataLoader(batch_sampler=...), not sampler=....
    """

    def __init__(self, dataset, batch_size: int, num_domains: int):
        if batch_size % num_domains != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by num_domains ({num_domains})"
            )
        if not hasattr(dataset, 'domain_indices'):
            raise ValueError("dataset must expose domain_indices for domain-balanced sampling")

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_domains = num_domains
        self.per_domain = batch_size // num_domains
        self.domain_indices = {
            domain_id: list(dataset.domain_indices[domain_id])
            for domain_id in range(num_domains)
        }

        missing = [d for d, idxs in self.domain_indices.items() if len(idxs) == 0]
        if missing:
            raise ValueError(f"Domains with no samples: {missing}")

        sizes = [len(idxs) for idxs in self.domain_indices.values()]
        geomean = int(math.prod(sizes) ** (1 / len(sizes)))
        self.num_batches = geomean // self.per_domain

    def _sample_domain(self, idxs: list, n: int) -> list:
        """Draw n indices from idxs, cycling with reshuffle when exhausted."""
        result = []
        pool = list(idxs)
        random.shuffle(pool)
        while len(result) < n:
            need = n - len(result)
            result.extend(pool[:need])
            if need >= len(pool):
                random.shuffle(pool)
        return result

    def __iter__(self):
        need = self.num_batches * self.per_domain
        domain_samples = {
            domain_id: self._sample_domain(idxs, need)
            for domain_id, idxs in self.domain_indices.items()
        }

        for batch_idx in range(self.num_batches):
            start = batch_idx * self.per_domain
            end = start + self.per_domain
            batch = []
            for domain_id in range(self.num_domains):
                batch.extend(domain_samples[domain_id][start:end])
            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches
