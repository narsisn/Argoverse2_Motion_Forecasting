# MFTF: Map Free Motion Forecasting Using Transformers
**Argoverse2 Motion Forecasting**

## TGR Model 
![TGR](https://user-images.githubusercontent.com/41056415/172166875-208c157b-9af9-4956-a8f2-198cfed98da3.jpg)

## MFTF Model 
![TR](https://user-images.githubusercontent.com/41056415/174430533-e808d059-03ee-4150-92b6-7b4c4a5e22b0.jpg)

Overview
=================
  * [Set up Virtual Environment](#set_up_virtual_environment)
  * [Download Argovers2 Dataset](#download_argovers2_dataset)
  * [Data Cleaning](#data_cleaning)
  * [Raw Feature Extracting](#Extract_Raw_Features_from_arg2_Dataset)
  * [Training](#training)
  * [Testing](#testing)
  * [Checkpoints](#Checkpoints)

## Set up Virtual Environment
Create your virtual environment to run the code: 

```sh
conda env create -f MFTF.yml

# Active the environment
conda activate MFTF

# install argoverse api_1
pip3 install  git+https://github.com/argoai/argoverse-api.git

# install argoverse api_2
pip3 install  av2

```
## Download Argovers2 Dataset

Run the following script to download the Argovers Motion Forecasting Version2. [Dataset Link](https://github.com/argoai/av2-api/blob/main/src/av2/datasets/motion_forecasting/README.md)

```sh
bash scripts/download_dataset.sh
```
## Data Cleaning
According to the paper of [Argoverse 2](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/4734ba6f3de83d861c3176a6273cac6d-Paper-round2.pdf), the focal agent should always be observed over the full 11 seconds, which then corresponds to 110 observations:

*Within each scenario, we mark a single track as the **focal agent**. Focal tracks are guaranteed to be fully observed throughout the duration of the scenario and have been specifically selected to maximize interesting interactions with map features and other nearby actors (see Section 3.3.2)*

However, this is not the case for some scenarios (~3% of the scenarios).
One example: Scenario '0215552f-6951-47e5-8cf6-3d1351d28957' of the validation set has a trajectory with only 104 observations.
To clean these scenarios from the dataset run the following script. This code cleans the train and val directories. 
```sh
python3 scripts/clean_data.py
```
## Raw Feature Extracting
To preprocess and extract the posotion_x and position_y displacements run the following command. This code creates three .pkl files for train, val, and, test.  
```sh
python3 preprocess.py
```  
## Training
This repository contains two models: 

**1- CRAT-Pred Model (LSTM + Graph + Multi-Head Self-Attention + Residual):** Vehicle Trajectory Prediction with Crystal Graph Convolutional Neural Networks and Multi-Head Self-Attention. [Paper](https://arxiv.org/abs/2202.04488) 

**2-TGR Model (Transformer Encoder + Graph + Residual):** Replacing the LSTM + Multi-Head Self-Attention sub-networks with a single Transformer Encoder

For training the model please use the model name, possible values for this option are train_TGR and train_Crat_pred: 
```sh
python3 train_TGR.py --use_preprocessed=True  # for using the offline preprocessing step please use True for --use_preprocessed  
```

## Testing
To test model on validation data on models:

1- Please comment line 217 in the following file:
https://github.com/argoai/argoverse-api/blob/master/argoverse/evaluation/eval_forecasting.py 
First, find the path of this file in your system after installing the Argoverse api_1 package. (pip3 install git+https://github.com/argoai/argoverse-api.git)
Then comment the line 217 which uses the Map information to compute the DAC parameter. 

```sh
python3 test_TGR.py --ckpt_path=/path/to/checkpoints --split=val 
#Also you can use this file to generate prediction on test dataset. simply use --split-test
```
## Checkpoints 
```sh
#Regenerate the evaluation parameters using pre-trained checkpoints: TGR.ckpt and Crat_Pred.ckpt
python3 test_TGR.py --ckpt_path=checkpoints/TGR.ckpt --split=val --use_preprocessed=True  
```

Base Code: The base code of [repository](https://github.com/schmidt-ju/crat-pred) has been used.
