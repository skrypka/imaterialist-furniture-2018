from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import models
import utils
from utils import RunningMean, use_gpu
from misc import FurnitureDataset, preprocess, preprocess_with_augmentation, NB_CLASSES, normalize_05

BATCH_SIZE = 16
IMAGE_SIZE = 224


def get_model():
    print('[+] loading model... ', end='', flush=True)
    model = models.resnext101_finetune(NB_CLASSES)
    if use_gpu:
        model.cuda()
    print('done')
    return model


def train():
    train_dataset = FurnitureDataset('train', transform=preprocess_with_augmentation(normalize_05, IMAGE_SIZE))
    val_dataset = FurnitureDataset('val', transform=preprocess(normalize_05, IMAGE_SIZE))
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=8,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)
    validation_data_loader = DataLoader(dataset=val_dataset, num_workers=1,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)

    model = get_model()

    criterion = nn.CrossEntropyLoss().cuda()

    nb_learnable_params = sum(p.numel() for p in model.fresh_params())
    print(f'[+] nb learnable params {nb_learnable_params}')

    lx, px = utils.predict(model, validation_data_loader)
    min_loss = criterion(Variable(px), Variable(lx)).data[0]

    lr = 0
    patience = 0
    for epoch in range(20):
        print(f'epoch {epoch}')
        if epoch == 1:
            lr = 0.00003
            print(f'[+] set lr={lr}')
        if patience == 2:
            patience = 0
            model.load_state_dict(torch.load('resnext.pth'))
            lr = lr / 10
            print(f'[+] set lr={lr}')
        if epoch == 0:
            lr = 0.001
            print(f'[+] set lr={lr}')
            optimizer = torch.optim.Adam(model.fresh_params(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

        running_loss = RunningMean()
        running_score = RunningMean()

        model.train()
        pbar = tqdm(training_data_loader, total=len(training_data_loader))
        for inputs, labels in pbar:
            batch_size = inputs.size(0)

            inputs = Variable(inputs)
            labels = Variable(labels)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, dim=1)

            loss = criterion(outputs, labels)
            running_loss.update(loss.data[0], 1)
            running_score.update(torch.sum(preds != labels.data), batch_size)

            loss.backward()
            optimizer.step()

            pbar.set_description(f'{running_loss.value:.5f} {running_score.value:.3f}')
        print(f'[+] epoch {epoch} {running_loss.value:.5f} {running_score.value:.3f}')

        lx, px = utils.predict(model, validation_data_loader)
        log_loss = criterion(Variable(px), Variable(lx))
        log_loss = log_loss.data[0]
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds != lx).float())
        print(f'[+] val {log_loss:.5f} {accuracy:.3f}')

        if log_loss < min_loss:
            torch.save(model.state_dict(), 'resnext.pth')
            print(f'[+] val score improved from {min_loss:.5f} to {log_loss:.5f}. Saved!')
            min_loss = log_loss
            patience = 0
        else:
            patience += 1


if __name__ == "__main__":
    train()
