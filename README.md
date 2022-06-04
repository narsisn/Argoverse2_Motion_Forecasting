# MFTF: Map Free Motion Forecasting Using Transformers
**Argoverse2 Motion Forecasting**

IMAGE  ARCHTITECTURE 

Overview
=================
  * [Set up Virtual Environment](#Set_up__Virtual_Environment)
  * [Download Argovers2 Dataset](#Download_Argovers2_Dataset)
  * [Data Cleaning](#Data_Cleaning)
  * [Raw Feature Extracting](#Extract_Raw_Features_from_arg2_Dataset)
  * [Training](#training)
  * [Testing](#testing)

## Set up Virtual Environment
Create your virtual enviroment to run the code: 

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
To clean this scenarios from data set run the following script. This code cleans the train and val directories. 
```sh
bash scripts/clean_data.py
```
## Raw Feature Extracting
To preprocess and extraxct the posotion_x and position_y run the following command.

## Training

## Testing
