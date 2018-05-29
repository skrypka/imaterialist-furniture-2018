import json
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from augmentation import HorizontalFlip

NB_CLASSES = 128


class FurnitureDataset(Dataset):
    def __init__(self, preffix: str, transform=None):
        self.preffix = preffix
        if preffix == 'val':
            path = 'validation'
        else:
            path = preffix
        path = f'data/{path}.json'
        self.transform = transform
        img_idx = {int(p.name.split('.')[0])
                   for p in Path(f'tmp/{preffix}').glob('*.jpg')}
        data = json.load(open(path))
        if 'annotations' in data:
            data = pd.DataFrame(data['annotations'])
        else:
            data = pd.DataFrame(data['images'])
        self.full_data = data
        nb_total = data.shape[0]
        data = data[data.image_id.isin(img_idx)].copy()
        data['path'] = data.image_id.map(lambda i: f"tmp/{preffix}/{i}.jpg")
        self.data = data
        print(f'[+] dataset `{preffix}` loaded {data.shape[0]} images from {nb_total}')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row['path'])
        if self.transform:
            img = self.transform(img)
        target = row['label_id'] - 1 if 'label_id' in row else -1
        return img, target


normalize_torch = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
normalize_05 = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)


# IMAGE_SIZE = 224
# IMAGE_SIZE = 299


def preprocess(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])


def preprocess_hflip(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        HorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def preprocess_with_augmentation(normalize, image_size):
    return transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),
        transforms.RandomRotation(15, expand=True),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        normalize
    ])
