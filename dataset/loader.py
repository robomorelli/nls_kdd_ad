import pandas as pd
import numpy as np
import os
import random
# from subprocess import check_output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
# from config import *

def _get_dataset(scale):
    """ Gets the basic dataset
    Returns :
            dataset (dict): containing the data
                dataset['x_train'] (np.array): training images shape
                (?, 120)
                dataset['y_train'] (np.array): training labels shape
                (?,)
                dataset['x_test'] (np.array): testing images shape
                (?, 120)
                dataset['y_test'] (np.array): testing labels shape
                (?,)
    """
    col_names = _col_names()
    df = pd.read_csv("data/kddcup.data_10_percent_corrected", header=None, names=col_names)
    text_l = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']

    for name in text_l:
        _encode_text_dummy(df, name)

    labels = df['label'].copy()
    labels[labels != 'normal.'] = 0
    labels[labels == 'normal.'] = 1

    df['label'] = labels

    df_train = df.sample(frac=0.5, random_state=42)
    df_test = df.loc[~df.index.isin(df_train.index)]
    df_valid = df_train.sample(frac=0.1, random_state=42)
    # df_train = df_train.loc[~df_train.index.isin(df_valid.index)]

    x_train, y_train = _to_xy(df_train, target='label')
    x_valid, y_valid = _to_xy(df_valid, target='label')
    x_test, y_test = _to_xy(df_test, target='label')

    y_train = y_train.flatten().astype(int)
    y_valid = y_valid.flatten().astype(int)
    y_test = y_test.flatten().astype(int)
    x_train = x_train[y_train != 1]
    y_train = y_train[y_train != 1]
    x_valid = x_valid[y_valid != 1]
    y_valid = y_valid[y_valid != 1]

    if scale:
        print("Scaling KDD dataset")
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(x_test)

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_valid'] = x_valid.astype(np.float32)
    dataset['y_valid'] = y_valid.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    return dataset


#######################################################################
#######################################################################

def get_train_val(val_split = 0.2, cols=None, cat_cols=None, ohe=None, exclude_cat=False, preprocessing = None):

    df_all = pd.read_csv("all_kdd_dataset/KDDTrain+.txt", header=None, names=cols)
    cont_cols = [x for x in cols if x not in cat_cols]
    
    if preprocessing == 'log':
        for c in cont_cols:
#             print(c)
            if (c != 'label') & (c != 'difficulty'):
#             df.apply(lambda x: np.log(x['src_bytes']) if x['src_bytes']!= 0 else 0,axis = 1)
                df_all[c] = df_all[c].apply(lambda x: np.log(x+1))

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

    df_encoded['label'] = df_encoded['label'].apply(lambda x: 0 if x=='normal' else 1)
    df_encoded = df_encoded[df_encoded['label']==0]
    df_encoded = df_encoded.drop('difficulty', axis = 1)

    trainx, valx, trainy, valy = train_test_split(df_encoded.drop(columns='label'), df_encoded['label'], test_size = val_split)

    return trainx.values, valx.values, ohe


def get_test(cols=None, cat_cols=None, ohe=None, exclude_cat=False, preprocessing = None):

    df_all = pd.read_csv("all_kdd_dataset/KDDTest+.txt", header=None, names=cols)
    cont_cols = [x for x in cols if x not in cat_cols]
    
    if preprocessing == 'log':
        for c in cont_cols:
#             print(c)
            if (c != 'label') & (c != 'difficulty'):
#                 print(c)
#             df.apply(lambda x: np.log(x['src_bytes']) if x['src_bytes']!= 0 else 0,axis = 1)
                df_all[c] = df_all[c].apply(lambda x: np.log(x+1))

    if exclude_cat:
        df_encoded = df_all[cont_cols]

    else:
        sorted_cols = cont_cols + cat_cols
        df_all = df_all[sorted_cols]

        df_cat = df_all[cat_cols].copy()
        array_hot_encoded = ohe.transform(df_cat)
        df_hot_encoded = pd.DataFrame(array_hot_encoded.toarray(), index=df_cat.index)

        df_other_cols = df_all.drop(columns=cat_cols, axis = 1)
        df_encoded = pd.concat([df_other_cols, df_hot_encoded], axis=1)

    original_labels =  df_encoded['label']
    df_encoded['label'] = df_encoded['label'].apply(lambda x: 0 if x=='normal' else 1)
    labels = df_encoded['label'].values
    df_encoded = df_encoded.drop(['label','difficulty'], axis = 1)

    return df_encoded.values, labels, original_labels