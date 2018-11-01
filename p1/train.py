import pandas as pd
import numpy as np
import pickle
import sklearn
import os
from sklearn import mixture



#Training code
phonemes = np.array(['', 'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 'sil', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z'])

def train_and_save_model(n_components,dropout,save_path,features,labels,phonemes):
    if dropout:
        path  = os.path.join(save_path,"without",str(n_components))
        X_train = features[:,np.arange(features.shape[1])%13!=0]
    else:
        X_train = features
        path  = os.path.join(save_path,"with",str(n_components))
    y_train = labels    
    trained_models = {}
    for p in phonemes:
        gmm = mixture.GaussianMixture(n_components=n_components,covariance_type='diag')
        gmm.fit(X_train[y_train==p])
        trained_models[p]=gmm
    with open(os.path.join(path,"gmm.pkl"),'wb') as fd:
        pickle.dump(trained_models,fd)

#mfcc
#a)
df = pd.read_hdf("./features/mfcc/timit_train.hdf")
features = np.array(df["features"].tolist())
labels = np.array(df["labels"].tolist())
save_path = "./models/mfcc"

## i)
print("mfcc, with, n_components=",64)
train_and_save_model(64,False,save_path,features,labels,phonemes)
## ii)
n_c = 2
while n_c<=256:
    print("mfcc, without, n_components=",n_c)
    train_and_save_model(n_c,True,save_path,features,labels,phonemes)
    n_c = n_c*2

#mfcc-delta
#b)
df = pd.read_hdf("./features/mfcc_delta/timit_train.hdf")
features = np.array(df["features"].tolist())
labels = np.array(df["labels"].tolist())
save_path = "./models/mfcc_delta"

## i)
print("mfcc_delta, with, n_components=",64)
train_and_save_model(64,False,save_path,features,labels,phonemes)
## ii)
print("mfcc_delta, without, n_components=",64)
train_and_save_model(64,True,save_path,features,labels,phonemes)

#mfcc-delta-delta
#c)
df = pd.read_hdf("./features/mfcc_delta_delta/timit_train.hdf")
features = np.array(df["features"].tolist())
labels = np.array(df["labels"].tolist())
save_path = "./models/mfcc_delta_delta"

## i)
print("mfcc_delta_delta, with, n_components=",64)
train_and_save_model(64,False,save_path,features,labels,phonemes)
## ii)
print("mfcc_delta_delta, without, n_components=",64)
train_and_save_model(64,True,save_path,features,labels,phonemes)
