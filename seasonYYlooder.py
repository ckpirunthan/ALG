import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
# Define data augmentation parameters




def load():
    #df_alg_site1=pd.read_csv("site1/alg_site1_window.csv")
    #df_alg_site2=pd.read_csv("site2/alg_site2_window.csv")
    #df_alg= pd.concat([df_alg_site1,df_alg_site2],ignore_index=True)
    df_alg=pd.read_csv("F:/CSU/site4/alg_site4_window.csv")
    df_alg = df_alg[df_alg.index % 3 == 0]
    df_alg.to_csv("site4/alg_site1_window_combined.csv", index=False)
    #df_alg= df_alg[((df_alg['22_band5'] -df_alg['22_band3'])/(df_alg['22_band5'] +df_alg['22_band3']))> 0.4]
    print("before alg hi")
    print(len(df_alg))
    df_algtemp=df_alg
    df_alg=pd.DataFrame({})
    # Load the data from CSV
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
    #df_nonalg_site1=pd.read_csv("site1/non_alg_site1_window.csv")
    #df_nonalg_site2=pd.read_csv("site2/non_alg_site2_window.csv")
    #df_nonalg= pd.concat([df_nonalg_site1,df_nonalg_site2],ignore_index=True)
    df_nonalg1=pd.read_csv("F:/CSU/site4/nonalg_site4_window.csv")
    #df_nonalg1= df_nonalg1[((df_nonalg1['22_band5'] -df_nonalg1['22_band3'])/(df_nonalg1['22_band5'] +df_nonalg1['22_band3']))> 0.4]
    df_extra=pd.read_csv("F:/CSU/site4/ndvithreshold5&6_points_window.csv")
    filtered_nonalg = df_extra[((df_extra['22_band5'] -df_extra['22_band3'])/(df_extra['22_band5'] +df_extra['22_band3']))> 0.4]
    df_nonalg=pd.concat([filtered_nonalg,df_nonalg1],ignore_index=True)
    df_nonalg.to_csv("site4/nonalg_site1_window_combined.csv", index=False)

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
    print("after nonalg")
    print(len(df_nonalg))

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

    ####################################
    #df_nonveg_site1= pd.read_csv("site1/non_veg_site1_window.csv")
    #df_nonveg_site2=pd.read_csv("site2/non_veg_site2_window.csv")
    #df_nonveg= pd.concat([df_nonveg_site1,df_nonveg_site2],ignore_index=True)
    df_nonveg1=pd.read_csv("F:/CSU/site4/nonveg_site4_window.csv")
    filtered_nonveg = df_extra[((df_extra['22_band5'] -df_extra['22_band3'])/(df_extra['22_band5'] +df_extra['22_band3']))< 0.4]
    df_nonveg=pd.concat([filtered_nonveg,df_nonveg1],ignore_index=True)
    df_nonveg = df_nonveg[df_nonveg.index % 3 == 0]
    df_nonveg.to_csv("site4/nonveg_site1_window_combined.csv", index=False)
    df_nonvegtemp = df_nonveg
    print("before nonveg")
    print(len(df_nonveg))
    df_nonveg = pd.DataFrame({})
    for i in range(8):
      heading_list = df_nonvegtemp.columns
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
    print("after nonalg")
    print(len(df_nonveg))

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
    #print("before nonveg")
    #print(len(df_nonveg))
    #df_nonveg = df_nonveg[df_nonveg.index % 20 == 0]
    #print("after nonveg")
    #print(len(df_nonveg))
    #df_alg = df_alg[df_alg.index % 3 == 0]
    #df_nonalg= df_nonalg[df_nonalg.index % 20 == 0]
    #df_nonveg = df_nonveg[df_nonveg.index % 3 == 0]

    print(len(df_alg))
    print(len(df_nonalg))
    print(len(df_nonveg))
    return df_alg,df_nonalg,df_nonveg
