Hello!

Below you can find a outline of how to reproduce my solution for the "iMaterialist Challenge (Furniture) at FGVC5" competition.
If you run into any trouble with the setup/code or have any questions please contact me at ivankivalov1980@gmail.com

# HARDWARE: (The following specs were used to create the original solution)
Ubuntu 16.04 LTS (512 GB boot disk)
16 vCPUs, 24 GB memory
2 x GeForce GTX 1080

# SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.6.4
CUDA 8.0
CuDNN 7.1.4.18

# DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
Data from Kaggle inside ./data/

# DATA PROCESSING
`python downloader.py`

# MODEL BUILD:
sh ./train.sh
sh ./predict.sh
