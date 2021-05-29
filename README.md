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
- conda create --name project python=3.7
- conda activate project
- conda install nb_conda
- conda install tensorflow=2.1.0
- conda install scikit-learn scipy pandas pandas-datareader matplotlib pillow tqdm requests h5py pyyaml flask boto3 kaggle gym bayesian-optimization keras spacy
- conda install tweepy langid langdetect nltk glob pickle
- pip install regex collection joblib
- python -m ipykernel install --user --name project --display-name "kernel: df"

## 'db' directory
- Since the size of the CSV files are huge, they have been uploaded to MediaFire.
- Download all the CSV files
- Create a new directory named 'db' in the root folder
- Paste all the CSV files in this folder
- link: https://www.mediafire.com/folder/u9eaoxz0g6gvw/db

## 'sentiment-analysis-model' directory
- This folder contains:
  - Tensorflow/Keras Models (For CPU and GPU) (Saved as YAML Files)
  - Tensorflow/Keras Model Weights (For CPU and GPU) (Saved as H5 Files)
  - Tokenizer used (as a PICKLE File)

## 'graphs' directory
- This folder contains the accuracy and loss graphs.
- The graphs plots training results against validation results
- It containts graphs for both CPU and GPU based Models

## 'classification-model' directory
- This directory contains the final classification model that will be used to fix any errors that our original model made while predicting, hencing improving accuracy.
- The Classifier was made using scikit-learn's SVM Classifier.
- This Classifier takes 3 things as an input:
  - Prediction from the Sentiment Analysis Model
  - Rating of the User (indicating tendency of the user to say something positive or negative)
  - Sum of the word score of the sentence

## 'buffer' directory
- This directory contains all the tempory files that were created while filtering our all the tweets that were note in english.

## 'notebook' directory
- Good Ol' IPYNB files that can be opened using Jupyter Notebooks
- This is the place where you'd wanna tinker with my code for fun
- Just don't forget to make the same changes in the python files in the root directory
- Have Fun Fellas!!!
