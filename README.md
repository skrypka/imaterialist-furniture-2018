## Kaggle: imaterialist-challenge-furniture-2018

Competition: image classification with 128 classes
Link: https://www.kaggle.com/c/imaterialist-challenge-furniture-2018
Result: public leaderboard score 0.15026

# How to run
1. Download data from kaggle to `./data/`
2. Download images `python downloader.py`
3. Train models `python cnn_runner.py`
4. Predict `python predict_all.py`
5. Generate submission `Submit.ipynb`

# Overview:
- ensemble inceptionv4, densenet161, densenet201, inceptionresnetv2, xception
- 12TTA
- probability calibration from train distribution to balanced distribution

# Some results:

- resnet34 - epoch 18 val 0.62678 0.172 (15min*18epoch)
- resnet50 - epoch 4 val 0.63055 0.171 (30min*4)
- resnet101 - epoch 6 val 0.59619 0.157 (43min*6)
- inception3 - epoch 8 val 0.60509 0.154 (37min*8) = 0.16927 LB
- densenet121 - epoch 17 val 0.60620 0.167 (28min*17)
- densenet161 - epoch 7 val 0.57006 0.149 (60min*7) = 0.16406 LB
- densenet201 - epoch 11 val 0.54275 0.145 (45min*11) = 0.15755 LB
- densenet161 - epoch 13 val 0.53795 0.150 = 0.15130 LB
- inceptionv4 - epoch 15 val 0.52382 0.137 (70min*15)
- inceptionresnetv2 - epoch 11 val 0.49438 0.139 (65min*11)
- xception - epoch 14 val 0.53719 0.149 (65min*15)
