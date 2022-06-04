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
bash download_dataset.sh
```
