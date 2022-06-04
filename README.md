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

