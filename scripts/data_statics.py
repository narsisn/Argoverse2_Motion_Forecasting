import pandas as pd
import glob 
import os 
import numpy as np


path_list = ['dataset/argoverse/train/**/*.parquet','dataset/argoverse/val/**/*.parquet','dataset/argoverse/test/**/*.parquet']

for plist in path_list:

    paths = glob.glob(plist)
    index = 0 
    std=[]
    for path in paths:
        df = pd.read_parquet(path)
        agent = df[df['track_id']==df['focal_track_id'][0]]

        agent = agent["position_x"][0:50]
        std.append(agent.describe().iloc[2])
    
    np.savetxt(plist.split('/')[2]+".csv", 
           std,
           delimiter =",", 
           fmt ='% s')
    print("min is ", min(std) )
    print("max is ", max(std))
    print("mean is ", sum(std)/len(std))


df = pd.read_csv('train.csv')
print("train_decribe: ", df.describe())
print(df.round(1).value_counts())
df = df.sort_values(df.columns[0])
df.round(1).value_counts().to_csv('train_stat.csv')

df = pd.read_csv('val.csv')
print("val_decribe: ", df.describe())
df = df.sort_values(df.columns[0])
df.round(1).value_counts().to_csv('val_stat.csv')

df = pd.read_csv('test.csv')
print("test_decribe: ", df.describe())
df = df.sort_values(df.columns[0])
df.round(1).value_counts().to_csv('test_stat.csv')