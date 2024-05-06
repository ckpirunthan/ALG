
from sklearn import  metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from matplotlib.pylab import rcParams
import pickle
import xgboost as xgb
import shap
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from xgboost import cv
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
#import pandas as pd

#import numpy as np
#import matplotlib.pyplot as plt

#from sklearn.preprocessing import MinMaxScaler

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data augmentation parameters
def load():
    df_alg=pd.read_csv("data/training/site1/alg_site1_window.csv")
    df_algtemp=df_alg
    #df_alg=pd.DataFrame({})
    # Load the data from CSV


    df_algtemp=df_alg
    print("before nonalg hi")
    print(len(df_alg))
    df_alg=pd.DataFrame({})

    for i in range(8):


        heading_list=df_algtemp.columns
        band_mapping = {
            "11": 12,
            "12": 13,
            "13": 23,
            "23": 33,
            "33": 32,
            "32": 31,
            "31": 21,
            "21": 11
        }

        # Update indices of bands in the shuffled list according to the mapping
        updated_list = [f"{str(band_mapping[s[:2]])}_band{s[-1]}" if s[:2] in band_mapping else s for s in heading_list]

        print(len(updated_list))


        df_algtemp.columns = updated_list
        df_alg = pd.concat([df_alg, df_algtemp], ignore_index=True)


    print("after alg")
    print(len(df_alg))
    for i in range(1,4):
      for j in range(1,4):
        df_alg[str(i)+str(j)+"_"+"band6"] = (df_alg[str(i)+str(j)+"_"+"band5"] - df_alg[str(i)+str(j)+"_"+"band3"]) / (df_alg[str(i)+str(j)+"_"+"band5"] + df_alg[str(i)+str(j)+"_"+"band3"])
        df_alg[str(i)+str(j)+"_"+"band7"]= (df_alg[str(i)+str(j)+"_"+"band2"]-df_alg[str(i)+str(j)+"_"+"band5"])/(df_alg[str(i)+str(j)+"_"+"band2"]+df_alg[str(i)+str(j)+"_"+"band5"])
        df_alg[str(i)+str(j)+"_"+"band8"]= (df_alg[str(i)+str(j)+"_"+"band5"]/df_alg[str(i)+str(j)+"_"+"band2"])-1
        df_alg[str(i)+str(j)+"_"+"band9"]= (2*df_alg[str(i)+str(j)+"_"+"band2"]-df_alg[str(i)+str(j)+"_"+"band1"]-df_alg[str(i)+str(j)+"_"+"band3"])/(2*df_alg[str(i)+str(j)+"_"+"band2"]+df_alg[str(i)+str(j)+"_"+"band1"]+df_alg[str(i)+str(j)+"_"+"band3"])
        df_alg[str(i)+str(j)+"_"+"band10"]= (df_alg[str(i)+str(j)+"_"+"band5"]-df_alg[str(i)+str(j)+"_"+"band4"])/(df_alg[str(i)+str(j)+"_"+"band5"]+df_alg[str(i)+str(j)+"_"+"band4"])
        df_alg[str(i)+str(j)+"_"+"band11"]= (df_alg[str(i)+str(j)+"_"+"band1"])/(df_alg[str(i)+str(j)+"_"+"band1"]+df_alg[str(i)+str(j)+"_"+"band2"]+df_alg[str(i)+str(j)+"_"+"band3"]+df_alg[str(i)+str(j)+"_"+"band4"]+df_alg[str(i)+str(j)+"_"+"band5"])
        df_alg[str(i)+str(j)+"_"+"band12"]= (df_alg[str(i)+str(j)+"_"+"band2"])/(df_alg[str(i)+str(j)+"_"+"band1"]+df_alg[str(i)+str(j)+"_"+"band2"]+df_alg[str(i)+str(j)+"_"+"band3"]+df_alg[str(i)+str(j)+"_"+"band4"]+df_alg[str(i)+str(j)+"_"+"band5"])
        df_alg[str(i)+str(j)+"_"+"band13"]= (df_alg[str(i)+str(j)+"_"+"band3"])/(df_alg[str(i)+str(j)+"_"+"band1"]+df_alg[str(i)+str(j)+"_"+"band2"]+df_alg[str(i)+str(j)+"_"+"band3"]+df_alg[str(i)+str(j)+"_"+"band4"]+df_alg[str(i)+str(j)+"_"+"band5"])
        df_alg[str(i)+str(j)+"_"+"band14"]= (df_alg[str(i)+str(j)+"_"+"band4"])/(df_alg[str(i)+str(j)+"_"+"band1"]+df_alg[str(i)+str(j)+"_"+"band2"]+df_alg[str(i)+str(j)+"_"+"band3"]+df_alg[str(i)+str(j)+"_"+"band4"]+df_alg[str(i)+str(j)+"_"+"band5"])
        df_alg[str(i)+str(j)+"_"+"band15"]= (df_alg[str(i)+str(j)+"_"+"band5"])/(df_alg[str(i)+str(j)+"_"+"band1"]+df_alg[str(i)+str(j)+"_"+"band2"]+df_alg[str(i)+str(j)+"_"+"band3"]+df_alg[str(i)+str(j)+"_"+"band4"]+df_alg[str(i)+str(j)+"_"+"band5"])

    df_alg["Label"]=0
    #########################
    df_nonalg=pd.read_csv("data/training/site1/nonalg_site1_window.cs")

    df_nonalgtemp=df_nonalg
    print("before nonalg hi")
    print(len(df_nonalg))
    df_nonalg=pd.DataFrame({})

    for i in range(8):


        heading_list=df_nonalgtemp.columns
        band_mapping = {
            "11": 12,
            "12": 13,
            "13": 23,
            "23": 33,
            "33": 32,
            "32": 31,
            "31": 21,
            "21": 11
        }

        # Update indices of bands in the shuffled list according to the mapping
        updated_list = [f"{str(band_mapping[s[:2]])}_band{s[-1]}" if s[:2] in band_mapping else s for s in heading_list]

        print(len(updated_list))


        df_nonalgtemp.columns = updated_list
        df_nonalg = pd.concat([df_nonalg, df_nonalgtemp], ignore_index=True)

    for i in range(1,4):
      for j in range(1,4):
        df_nonalg[str(i)+str(j)+"_"+"band6"]= (df_nonalg[str(i)+str(j)+"_"+"band5"]-df_nonalg[str(i)+str(j)+"_"+"band3"])/(df_nonalg[str(i)+str(j)+"_"+"band5"]+df_nonalg[str(i)+str(j)+"_"+"band3"])
        df_nonalg[str(i)+str(j)+"_"+"band7"]= (df_nonalg[str(i)+str(j)+"_"+"band2"]-df_nonalg[str(i)+str(j)+"_"+"band5"])/(df_nonalg[str(i)+str(j)+"_"+"band2"]+df_nonalg[str(i)+str(j)+"_"+"band5"])
        df_nonalg[str(i)+str(j)+"_"+"band8"]= (df_nonalg[str(i)+str(j)+"_"+"band5"]/df_nonalg[str(i)+str(j)+"_"+"band2"])-1
        df_nonalg[str(i)+str(j)+"_"+"band9"]= (2*df_nonalg[str(i)+str(j)+"_"+"band2"]-df_nonalg[str(i)+str(j)+"_"+"band1"]-df_nonalg[str(i)+str(j)+"_"+"band3"])/(2*df_nonalg[str(i)+str(j)+"_"+"band2"]+df_nonalg[str(i)+str(j)+"_"+"band1"]+df_nonalg[str(i)+str(j)+"_"+"band3"])
        df_nonalg[str(i)+str(j)+"_"+"band10"]= (df_nonalg[str(i)+str(j)+"_"+"band5"]-df_nonalg[str(i)+str(j)+"_"+"band4"])/(df_nonalg[str(i)+str(j)+"_"+"band5"]+df_nonalg[str(i)+str(j)+"_"+"band4"])

        df_nonalg[str(i)+str(j)+"_"+"band11"]= (df_nonalg[str(i)+str(j)+"_"+"band1"])/(df_nonalg[str(i)+str(j)+"_"+"band1"]+df_nonalg[str(i)+str(j)+"_"+"band2"]+df_nonalg[str(i)+str(j)+"_"+"band3"]+df_nonalg[str(i)+str(j)+"_"+"band4"]+df_nonalg[str(i)+str(j)+"_"+"band5"])
        df_nonalg[str(i)+str(j)+"_"+"band12"]= (df_nonalg[str(i)+str(j)+"_"+"band2"])/(df_nonalg[str(i)+str(j)+"_"+"band1"]+df_nonalg[str(i)+str(j)+"_"+"band2"]+df_nonalg[str(i)+str(j)+"_"+"band3"]+df_nonalg[str(i)+str(j)+"_"+"band4"]+df_nonalg[str(i)+str(j)+"_"+"band5"])
        df_nonalg[str(i)+str(j)+"_"+"band13"]= (df_nonalg[str(i)+str(j)+"_"+"band3"])/(df_nonalg[str(i)+str(j)+"_"+"band1"]+df_nonalg[str(i)+str(j)+"_"+"band2"]+df_nonalg[str(i)+str(j)+"_"+"band3"]+df_nonalg[str(i)+str(j)+"_"+"band4"]+df_nonalg[str(i)+str(j)+"_"+"band5"])
        df_nonalg[str(i)+str(j)+"_"+"band14"]= (df_nonalg[str(i)+str(j)+"_"+"band4"])/(df_nonalg[str(i)+str(j)+"_"+"band1"]+df_nonalg[str(i)+str(j)+"_"+"band2"]+df_nonalg[str(i)+str(j)+"_"+"band3"]+df_nonalg[str(i)+str(j)+"_"+"band4"]+df_nonalg[str(i)+str(j)+"_"+"band5"])
        df_nonalg[str(i)+str(j)+"_"+"band15"]= (df_nonalg[str(i)+str(j)+"_"+"band5"])/(df_nonalg[str(i)+str(j)+"_"+"band1"]+df_nonalg[str(i)+str(j)+"_"+"band2"]+df_nonalg[str(i)+str(j)+"_"+"band3"]+df_nonalg[str(i)+str(j)+"_"+"band4"]+df_nonalg[str(i)+str(j)+"_"+"band5"])

    df_nonalg["Label"]=1
    print("before nonalg")
    print(len(df_nonalg))
    ####################################
    df_nonveg= pd.read_csv("data/training/site1/nonveg_site1_window.cs")

    df_nonvegtemp=df_nonveg
    print("before nonalg hi")
    print(len(df_nonveg))
    df_nonveg=pd.DataFrame({})

    for i in range(8):


        heading_list=df_nonvegtemp.columns
        band_mapping = {
            "11": 12,
            "12": 13,
            "13": 23,
            "23": 33,
            "33": 32,
            "32": 31,
            "31": 21,
            "21": 11
        }

        # Update indices of bands in the shuffled list according to the mapping
        updated_list = [f"{str(band_mapping[s[:2]])}_band{s[-1]}" if s[:2] in band_mapping else s for s in heading_list]

        print(len(updated_list))


        df_nonvegtemp.columns = updated_list
        df_nonveg = pd.concat([df_nonveg, df_nonvegtemp], ignore_index=True)

    for i in range(1,4):
      for j in range(1,4):
        df_nonveg[str(i)+str(j)+"_"+"band6"]= (df_nonveg[str(i)+str(j)+"_"+"band5"]-df_nonveg[str(i)+str(j)+"_"+"band3"])/(df_nonveg[str(i)+str(j)+"_"+"band5"]+df_nonveg[str(i)+str(j)+"_"+"band3"])
        df_nonveg[str(i)+str(j)+"_"+"band7"]= (df_nonveg[str(i)+str(j)+"_"+"band2"]-df_nonveg[str(i)+str(j)+"_"+"band5"])/(df_nonveg[str(i)+str(j)+"_"+"band2"]+df_nonveg[str(i)+str(j)+"_"+"band5"])
        df_nonveg[str(i)+str(j)+"_"+"band8"]= (df_nonveg[str(i)+str(j)+"_"+"band5"]/df_nonveg[str(i)+str(j)+"_"+"band2"])-1
        df_nonveg[str(i)+str(j)+"_"+"band9"]= (2*df_nonveg[str(i)+str(j)+"_"+"band2"]-df_nonveg[str(i)+str(j)+"_"+"band1"]-df_nonveg[str(i)+str(j)+"_"+"band3"])/(2*df_nonveg[str(i)+str(j)+"_"+"band2"]+df_nonveg[str(i)+str(j)+"_"+"band1"]+df_nonveg[str(i)+str(j)+"_"+"band3"])
        df_nonveg[str(i)+str(j)+"_"+"band10"]= (df_nonveg[str(i)+str(j)+"_"+"band5"]-df_nonveg[str(i)+str(j)+"_"+"band4"])/(df_nonveg[str(i)+str(j)+"_"+"band5"]+df_nonveg[str(i)+str(j)+"_"+"band4"])

        df_nonveg[str(i)+str(j)+"_"+"band11"]= (df_nonveg[str(i)+str(j)+"_"+"band1"])/(df_nonveg[str(i)+str(j)+"_"+"band1"]+df_nonveg[str(i)+str(j)+"_"+"band2"]+df_nonveg[str(i)+str(j)+"_"+"band3"]+df_nonveg[str(i)+str(j)+"_"+"band4"]+df_nonveg[str(i)+str(j)+"_"+"band5"])
        df_nonveg[str(i)+str(j)+"_"+"band12"]= (df_nonveg[str(i)+str(j)+"_"+"band2"]/(df_nonveg[str(i)+str(j)+"_"+"band1"]+df_nonveg[str(i)+str(j)+"_"+"band2"]+df_nonveg[str(i)+str(j)+"_"+"band3"]+df_nonveg[str(i)+str(j)+"_"+"band4"]+df_nonveg[str(i)+str(j)+"_"+"band5"]))
        df_nonveg[str(i)+str(j)+"_"+"band13"]= (df_nonveg[str(i)+str(j)+"_"+"band3"]/(df_nonveg[str(i)+str(j)+"_"+"band1"]+df_nonveg[str(i)+str(j)+"_"+"band2"]+df_nonveg[str(i)+str(j)+"_"+"band3"]+df_nonveg[str(i)+str(j)+"_"+"band4"]+df_nonveg[str(i)+str(j)+"_"+"band5"]))
        df_nonveg[str(i)+str(j)+"_"+"band14"]= (df_nonveg[str(i)+str(j)+"_"+"band4"]/(df_nonveg[str(i)+str(j)+"_"+"band1"]+df_nonveg[str(i)+str(j)+"_"+"band2"]+df_nonveg[str(i)+str(j)+"_"+"band3"]+df_nonveg[str(i)+str(j)+"_"+"band4"]+df_nonveg[str(i)+str(j)+"_"+"band5"]))
        df_nonveg[str(i)+str(j)+"_"+"band15"]= (df_nonveg[str(i)+str(j)+"_"+"band5"]/(df_nonveg[str(i)+str(j)+"_"+"band1"]+df_nonveg[str(i)+str(j)+"_"+"band2"]+df_nonveg[str(i)+str(j)+"_"+"band3"]+df_nonveg[str(i)+str(j)+"_"+"band4"]+df_nonveg[str(i)+str(j)+"_"+"band5"]))

    df_nonveg["Label"]=2
    print("before nonveg")
    print(len(df_nonveg))


    print("after nonveg")
    print(len(df_alg))
    print(len(df_nonalg))
    print(len(df_nonveg))
    return df_alg,df_nonalg,df_nonveg
