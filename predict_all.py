import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import models
import utils
from augmentation import five_crops, HorizontalFlip, make_transforms
from misc import FurnitureDataset, preprocess, NB_CLASSES, preprocess_hflip, normalize_05, normalize_torch

BATCH_SIZE = 16
TTA = False
use_gpu = torch.cuda.is_available()


def get_model(model_class):
    print('[+] loading model... ', end='', flush=True)
    model = model_class(NB_CLASSES)
    if use_gpu:
        model.cuda()
    print('done')
    return model


def predict(model_name, model_class, weight_pth, image_size, normalize):
    print(f'[+] predict {model_name}')
    model = get_model(model_class)
    model.load_state_dict(torch.load(weight_pth))
    model.eval()

    tta_preprocess = [preprocess(normalize, image_size), preprocess_hflip(normalize, image_size)]
    tta_preprocess += make_transforms([transforms.Resize((image_size + 20, image_size + 20))],
                                      [transforms.ToTensor(), normalize],
                                      five_crops(image_size))
    tta_preprocess += make_transforms([transforms.Resize((image_size + 20, image_size + 20))],
                                      [HorizontalFlip(), transforms.ToTensor(), normalize],
                                      five_crops(image_size))
    print(f'[+] tta size: {len(tta_preprocess)}')

    data_loaders = []
    for transform in tta_preprocess:
        test_dataset = FurnitureDataset('test', transform=transform)
        data_loader = DataLoader(dataset=test_dataset, num_workers=1,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
        data_loaders.append(data_loader)

    lx, px = utils.predict_tta(model, data_loaders)
    data = {
        'lx': lx.cpu(),
        'px': px.cpu(),
    }
    torch.save(data, f'{model_name}_test_prediction.pth')

    data_loaders = []
    for transform in tta_preprocess:
        test_dataset = FurnitureDataset('val', transform=transform)
        data_loader = DataLoader(dataset=test_dataset, num_workers=1,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
        data_loaders.append(data_loader)

    lx, px = utils.predict_tta(model, data_loaders)
    data = {
        'lx': lx.cpu(),
        'px': px.cpu(),
    }
    torch.save(data, f'{model_name}_val_prediction.pth')


def predict_all():
    predict("inceptionv4", models.inceptionv4_finetune, 'inception4_052382.pth', 299, normalize_05)
    predict("densenet161", models.densenet161_finetune, 'densenet161_15130.pth', 224, normalize_torch)
    predict("densenet201", models.densenet201_finetune, 'densenet201_15755.pth', 224, normalize_torch)
    predict("inceptionresnetv2", models.inceptionresnetv2_finetune, 'inceptionresnetv2_049438.pth', 299, normalize_05)
    predict("xception", models.xception_finetune, 'xception_053719.pth', 299, normalize_05)
    predict("resnext", models.resnext101_finetune, 'resnext.pth', 224, normalize_05)
    predict("se_resnet152", models.se_resnet152_finetune, 'se_resnet152.pth', 224, normalize_torch)
    predict("se_resnet101", models.se_resnet101_finetune, 'se_resnet101.pth', 224, normalize_torch)
    predict("dpn92", models.dpn92_finetune, 'dpn92.pth', 224, normalize_torch)
    predict("senet154", models.senet154_finetune, 'senet154.pth', 224, normalize_torch)
    predict("nasnet", models.nasnet_finetune, 'nasnet.pth', 331, normalize_05)


if __name__ == "__main__":
    predict_all()
