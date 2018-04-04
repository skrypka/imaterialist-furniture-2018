import json
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from augmentation import HorizontalFlip

NB_CLASSES = 128
IMAGE_SIZE = 224


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


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize
])
preprocess_hflip = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    HorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
preprocess_with_augmentation = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3,
                           contrast=0.3,
                           saturation=0.3),
    transforms.ToTensor(),
    normalize
])
