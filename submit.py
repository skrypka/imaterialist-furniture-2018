import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
from misc import FurnitureDataset, preprocess

test_dataset = FurnitureDataset('test', transform=preprocess)

test_pred = torch.load('test_prediction.pth')
test_prob = F.softmax(Variable(test_pred['px']), dim=1).data.numpy()
test_prob = test_prob.mean(axis=2)

test_predicted = np.argmax(test_prob, axis=1)
test_predicted += 1
result = test_predicted

sx = pd.read_csv('data/sample_submission_randomlabel.csv')
sx.loc[sx.id.isin(test_dataset.data.image_id), 'predicted'] = result
sx.to_csv('sx.csv', index=False)
