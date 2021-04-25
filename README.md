# BEMajorProject

## All Databases:
https://drive.google.com/drive/folders/1cCF5x3Hly36iifxcLRRDgv5ejgkePK1h?usp=sharing

## System Information
- Tensorflow-GPU v2.1.0
- cuDNN v8.0.5 (only for devices with nvidia-gpu)
- CUDA v10.1 (only for devices with nvidia-gpu)

## Setting Up Miniconda
- Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html
- conda install -y jupyter
- conda create --name tf python=3.7
- conda activate tf
- conda install nb_conda
- conda install tensorflow=2.1.0
- conda install scikit-learn scipy pandas pandas-datareader matplotlib pillow tqdm requests h5py pyyaml flask boto3 kaggle gym bayesian-optimization keras spacy
- python -m ipykernel install --user --name tf --display-name "kernel: df"

## 'model-sa' Folder
- This folder contains:
  - Keras Models (For CPU and GPU) (Saved as YAML Files)
  - Keras Model Weights (For CPU and GPU) (Saved as H5 Files)
  - Tokenizer used (as a PICKLE File)
