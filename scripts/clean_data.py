import pandas as pd
import glob 
import os 


path_list = ['dataset/argoverse/train/**/*.parquet','dataset/argoverse/val/**/*.parquet']

for plist in path_list:

    paths = glob.glob(plist)
    index = 0 
    for path in paths:
        df = pd.read_parquet(path)
        agent = df[df['track_id']==df['focal_track_id'][0]]
        len_agent = len (agent)
        if len_agent < 110:
            print(path)
            os.remove(path)
            index += 1 
            print(index , " ", len_agent)
    print("The number of invalid agents is: ", index)