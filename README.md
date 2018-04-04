## Kaggle: imaterialist-challenge-furniture-2018

Competition: image classification with 128 classes
Link: https://www.kaggle.com/c/imaterialist-challenge-furniture-2018
Result: public leaderboard score 0.15026

# How to run
1. Download data from kaggle to `./data/`
2. Download images `python downloader.py`
3. Train model `python cnn_runner.py train`
4. Predict `python cnn_runner.py predict`
5. Generate submission `python submit.py`

Next steps:
- add 12TTA: 0.14583
- add other model: 0.14244
