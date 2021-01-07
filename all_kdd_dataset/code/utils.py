import pandas as pd
import numpy as np
import os
import random
from subprocess import check_output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from config import *

def get_train_val(val_split = 0.2, cols=None, cat_cols=None, ohe=None, exclude_cat=False):
    
    df_all = pd.read_csv(train_val +"KDDTrain+.txt", header=None, names=cols)
    cont_cols = [x for x in cols if x not in cat_cols]
    
    if exclude_cat:
        df_encoded = df_all[cont_cols]

    else:
        sorted_cols = cont_cols + cat_cols
        df_all = df_all[sorted_cols]

        df_cat = df_all[cat_cols].copy()        
        ohe = OneHotEncoder()
        array_hot_encoded = ohe.fit_transform(df_cat)
        df_hot_encoded = pd.DataFrame(array_hot_encoded.toarray(), index=df_cat.index)
        
        df_other_cols = df_all.drop(columns=cat_cols, axis = 1)
        df_encoded = pd.concat([df_other_cols, df_hot_encoded], axis=1)
    
    df_encoded['label'] = df_encoded['label'].apply(lambda x: 1 if x=='normal' else 0)
    df_encoded = df_encoded[df_encoded['label']==0]
    df_encoded = df_encoded.drop('difficulty', axis = 1)
        
    trainx, valx, trainy, valy = train_test_split(df_encoded.drop(columns='label'), df_encoded['label'], test_size = val_split)
        
    return trainx.values, valx.values, ohe
