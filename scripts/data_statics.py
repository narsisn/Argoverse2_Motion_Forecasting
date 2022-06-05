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

