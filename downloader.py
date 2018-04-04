import multiprocessing
import os
import json
import random
from io import BytesIO

import requests
from tqdm import tqdm
from PIL import Image

train_json = json.load(open('data/train.json'))
test_json = json.load(open('data/test.json'))
val_json = json.load(open('data/validation.json'))

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36',
}


def download_one(d):
    assert len(d['url']) == 1
    path = f'tmp/{d["preffix"]}/{d["image_id"]}.jpg'
    if os.path.isfile(path):
        return
    try:
        r = requests.get(d['url'][0], allow_redirects=True, timeout=60,
                         headers=headers)
    except:
        print('error downloading', d['url'][0])
        return
    if r.status_code != 200:
        print('status code != 200', r.status_code, d['url'][0])
        return
    try:
        pil_image = Image.open(BytesIO(r.content))
        pil_image_rgb = pil_image.convert('RGB')
        pil_image_rgb.save(path, format='JPEG', quality=90)
    except:
        print('error converting data to image', d)


def download_data(preffix, data):
    print(f'downloading {preffix}')
    data = data['images']
    random.shuffle(data)

    pool = multiprocessing.Pool(processes=20)
    for d in data:
        d['preffix'] = preffix

    with tqdm(total=len(data)) as t:
        for _ in pool.imap_unordered(download_one, data):
            t.update(1)


files = [
    ('val', val_json),
    ('test', test_json),
    ('train', train_json),
]
random.shuffle(files)
for preffix, data in files:
    download_data(preffix, data)
