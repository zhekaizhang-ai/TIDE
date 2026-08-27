import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


_DATA_ROOT = os.environ.get('TIDE_DATA_ROOT', 'data')

# Set TIDE_DATA_ROOT to a directory containing these dataset folders.
DOMAIN_ROOTS = {
    'LEVIR':   os.path.join(_DATA_ROOT, 'LEVIR-CD'),
    'GZ':      os.path.join(_DATA_ROOT, 'CD_Data_GZ'),
    'WHU':     os.path.join(_DATA_ROOT, 'WHU-CD'),
    'CDD':     os.path.join(_DATA_ROOT, 'CDD'),
    'HRCUS':   os.path.join(_DATA_ROOT, 'HRCUS-CD'),
    'HRCUS50': os.path.join(_DATA_ROOT, 'HRCUS-CD-50'),
    'SYSU':    os.path.join(_DATA_ROOT, 'SYSU-CD'),
    'EGY':     os.path.join(_DATA_ROOT, 'EGY-BCD'),
    'SIBU':    os.path.join(_DATA_ROOT, 'SI-BU-C2'),
    'DSIFN':   os.path.join(_DATA_ROOT, 'DSIFN-CD'),
}


def _sync_augment(img_a: Image.Image, img_b: Image.Image,
                  label: Image.Image):
    """Random horizontal and vertical flip, applied identically to A, B, label."""
    if random.random() < 0.5:
        img_a = img_a.transpose(Image.FLIP_LEFT_RIGHT)
        img_b = img_b.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img_a = img_a.transpose(Image.FLIP_TOP_BOTTOM)
        img_b = img_b.transpose(Image.FLIP_TOP_BOTTOM)
        label = label.transpose(Image.FLIP_TOP_BOTTOM)
    return img_a, img_b, label


class LEVIRDataset(Dataset):
    """
    LEVIR-CD flat-directory dataset.

    root/
      A/          T1 images (RGB PNG)
      B/          T2 images (RGB PNG)
      label/      change masks (grayscale, 0 / 255)
      list/       train.txt  val.txt  test.txt
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, root: str, split: str, img_size: int = 224,
                 augment: bool = False):
        self.root    = root
        self.augment = augment

        list_path = os.path.join(root, 'list', f'{split}.txt')
        with open(list_path) as f:
            self.names = [l.strip() for l in f if l.strip()]

        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name  = self.names[idx]
        img_a = Image.open(os.path.join(self.root, 'A',     name)).convert('RGB')
        img_b = Image.open(os.path.join(self.root, 'B',     name)).convert('RGB')
        label = Image.open(os.path.join(self.root, 'label', name)).convert('L')

        if self.augment:
            img_a, img_b, label = _sync_augment(img_a, img_b, label)

        img_a = self.img_tf(img_a)
        img_b = self.img_tf(img_b)

        # resize label to decoder output size (256×256) then binarize
        label = label.resize((256, 256), Image.NEAREST)
        label = torch.from_numpy(
            (np.array(label, dtype=np.uint8) > 127).astype(np.int64)
        )                                           # (256, 256)

        return img_a, img_b, label


class MultiSourceCDDataset(Dataset):
    """
    Multi-source change-detection dataset over LEVIR-style flat directories.

    Each sample returns:
        T1, T2, mask, domain_id, domain_name

    domain_id follows the order of the sources argument.
    Augmentation (random h/v flip) is applied synchronously to T1, T2 and mask
    when augment=True, ensuring spatial alignment is preserved.
    """

    MEAN = LEVIRDataset.MEAN
    STD  = LEVIRDataset.STD

    def __init__(self, sources: list[str], split: str, img_size: int = 224,
                 augment: bool = False, domain_caps: dict | None = None):
        if not sources:
            raise ValueError("sources must contain at least one domain name")

        unknown = [s for s in sources if s not in DOMAIN_ROOTS]
        if unknown:
            raise ValueError(
                f"Unknown source(s): {unknown}. Expected one of {sorted(DOMAIN_ROOTS)}"
            )

        self.sources = list(sources)
        self.split   = split
        self.augment = augment
        self.samples = []
        self.domain_indices = {domain_id: [] for domain_id in range(len(self.sources))}
        caps = domain_caps or {}

        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
        ])

        for domain_id, domain_name in enumerate(self.sources):
            root = DOMAIN_ROOTS[domain_name]
            list_path = os.path.join(root, 'list', f'{split}.txt')
            with open(list_path) as f:
                names = [l.strip() for l in f if l.strip()]

            if domain_name in caps and split == 'train':
                import random as _rnd
                rng = _rnd.Random(42)
                names = rng.sample(names, min(caps[domain_name], len(names)))

            for name in names:
                sample_idx = len(self.samples)
                self.samples.append({
                    'name':        name,
                    'root':        root,
                    'domain_id':   domain_id,
                    'domain_name': domain_name,
                })
                self.domain_indices[domain_id].append(sample_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        root   = sample['root']
        name   = sample['name']

        img_a = Image.open(os.path.join(root, 'A',     name)).convert('RGB')
        img_b = Image.open(os.path.join(root, 'B',     name)).convert('RGB')
        label = Image.open(os.path.join(root, 'label', name)).convert('L')

        if self.augment:
            img_a, img_b, label = _sync_augment(img_a, img_b, label)

        img_a = self.img_tf(img_a)
        img_b = self.img_tf(img_b)
        label = label.resize((256, 256), Image.NEAREST)
        label = torch.from_numpy(
            (np.array(label, dtype=np.uint8) > 127).astype(np.int64)
        )

        return img_a, img_b, label, sample['domain_id'], sample['domain_name']
