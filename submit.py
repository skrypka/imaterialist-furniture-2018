import json

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean

from misc import FurnitureDataset, preprocess

train_json = json.load(open('data/train.json'))
train_df_0 = pd.DataFrame(train_json['annotations'])
train_df_1 = pd.DataFrame(train_json['images'])
train_df = pd.merge(train_df_0, train_df_1)


def calibrate_prob(positive_prob_train, positive_prob_test, prob):
    return (positive_prob_test * prob) / (positive_prob_test * prob + positive_prob_train * (1 - prob))


def calibrate_probs(prob):
    nb_train = train_df.shape[0]
    for class_ in range(128):
        nb_positive_train = ((train_df.label_id - 1) == class_).sum()

        positive_prob_train = nb_positive_train / nb_train
        positive_prob_test = 1 / 128  # balanced class distribution
        for i in range(prob.shape[0]):
            old_p = prob[i, class_]
            new_p = calibrate_prob(positive_prob_train, positive_prob_test, old_p)
            prob[i, class_] = new_p


test_pred0 = torch.load('inceptionv4_test_prediction.pth', map_location={'cuda:0': 'cpu'})
test_pred1 = torch.load('densenet161_test_prediction.pth', map_location={'cuda:0': 'cpu'})
test_pred2 = torch.load('densenet201_test_prediction.pth', map_location={'cuda:0': 'cpu'})
test_pred3 = torch.load('inceptionresnetv2_test_prediction.pth', map_location={'cuda:0': 'cpu'})
test_pred4 = torch.load('xception_test_prediction.pth', map_location={'cuda:0': 'cpu'})
test_pred5 = torch.load('resnext_test_prediction.pth', map_location={'cuda:0': 'cpu'})
test_pred6 = torch.load('se_resnet152_test_prediction.pth', map_location={'cuda:0': 'cpu'})
test_pred7 = torch.load('se_resnet101_test_prediction.pth', map_location={'cuda:0': 'cpu'})
test_pred8 = torch.load('dpn92_test_prediction.pth', map_location={'cuda:0': 'cpu'})
test_pred9 = torch.load('senet154_test_prediction.pth', map_location={'cuda:0': 'cpu'})
test_pred10 = torch.load('nasnet_test_prediction.pth', map_location={'cuda:0': 'cpu'})

test_prob = F.softmax(Variable(torch.cat((
    test_pred1['px'],
    test_pred2['px'],
    test_pred3['px'],
    test_pred4['px'],
    test_pred5['px'],
), dim=2)), dim=1).data.numpy()

test_prob = gmean(test_prob, axis=2)
test_predicted = np.argmax(test_prob, axis=1) + 1

calibrate_probs(test_prob)
calibrated_predicted = np.argmax(test_prob, axis=1) + 1

test_dataset = FurnitureDataset('test', transform=preprocess)
sx = pd.read_csv('data/sample_submission_randomlabel.csv')
sx.loc[sx.id.isin(test_dataset.data.image_id), 'predicted'] = calibrated_predicted
sx.to_csv('sx_calibrated.csv', index=False)
