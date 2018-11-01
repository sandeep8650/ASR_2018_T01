import pandas as pd
import numpy as np
import os
import pickle

phonemes = np.array(['', 'aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 'sil', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z'])

def predict_phoneme(features,phonemes,dropout,gmm_dict):
    N,d = features.shape
    if dropout:
        X = features[:,np.arange(d)%13!=0]
    else:
        X = features
    #N,_ = X.shape
    scores = np.zeros((phonemes.shape[0],N))
    for i in range(phonemes.shape[0]):
        scores[i] = np.array(gmm_dict[phonemes[i]].score_samples(X))
    
    predicted_phoneme_index = np.argmax(scores,axis=0)
    predicted_phoneme = phonemes[predicted_phoneme_index]
    
    return predicted_phoneme

def fer(X,y,phonemes,dropout,gmm_dict):
    prediction = predict_phoneme(X,phonemes,dropout,gmm_dict)
    return np.sum(prediction==y)/y.size * 100

def generate_ground_truth(labels,ids,save_path):
    assert(labels.size == ids.size)
    ns = list(set(ids))
    with open(os.path.join(save_path,"ground_truth.txt"),'w') as fid:
        for s in ns:
            for p in labels[ids==s]:
                fid.write(p+" ")
            fid.write('\n')       
        
def generate_hypothesis(X,ids,phonemes,dropout,n_components,gmm_dict,save_path):
    assert(X.shape[0] == ids.size)
    ns = list(set(ids))
    prediction = predict_phoneme(X,phonemes,dropout,gmm_dict)
    if dropout:
        path  = os.path.join(save_path,"without",str(n_components))
    else:
        path  = os.path.join(save_path,"with",str(n_components))
    with open(os.path.join(path,"hypothesis.txt"),'w') as fid:
        for s in ns:
            for p in prediction[ids==s]:
                fid.write(p+" ")
            fid.write('\n')

#mfcc
#a)
df = pd.read_hdf("./features/mfcc/timit_test.hdf")
features = np.array(df["features"].tolist())
labels = np.array(df["labels"].tolist())
ids = np.array(df["ids"].tolist())
save_path = "./hypothesis/mfcc/"
generate_ground_truth(labels,ids,"./ground_truth/mfcc")

## i)
print("mfcc, with, n_components=",64)
with open("./models/mfcc/with/64/gmm.pkl",'rb') as fid:
    gmms = pickle.load(fid)
    generate_hypothesis(features,ids,phonemes,False,64,gmms,save_path)
    print(fer(features,labels,phonemes,False,gmms))

## ii)
n_c = 2
while n_c<=256:
    print("mfcc, without, n_components=",n_c)
    with open(os.path.join("./models/mfcc/without",str(n_c),"gmm.pkl"),'rb') as fid:
        gmms = pickle.load(fid)
        generate_hypothesis(features,ids,phonemes,True,n_c,gmms,save_path)
        print(fer(features,labels,phonemes,True,gmms))
    n_c = n_c*2
    
#mfcc-delta
#b)
df = pd.read_hdf("./features/mfcc_delta/timit_test.hdf")
features = np.array(df["features"].tolist())
labels = np.array(df["labels"].tolist())
ids = np.array(df["ids"].tolist())
save_path = "./hypothesis/mfcc_delta"
generate_ground_truth(labels,ids,"./ground_truth/mfcc_delta")

## i)
print("mfcc_delta, with, n_components=",64)
with open("./models/mfcc_delta/with/64/gmm.pkl",'rb') as fid:
    gmms = pickle.load(fid)
    generate_hypothesis(features,ids,phonemes,False,64,gmms,save_path)
    print(fer(features,labels,phonemes,False,gmms))
              
## ii)
print("mfcc_delta, without, n_components=",64)
with open("./models/mfcc_delta/without/64/gmm.pkl",'rb') as fid:
    gmms = pickle.load(fid)
    generate_hypothesis(features,ids,phonemes,True,64,gmms,save_path)
    print(fer(features,labels,phonemes,True,gmms))

#mfcc-delta-delta
#c)
df = pd.read_hdf("./features/mfcc_delta_delta/timit_test.hdf")
features = np.array(df["features"].tolist())
labels = np.array(df["labels"].tolist())
ids = np.array(df["ids"].tolist())
save_path = "./hypothesis/mfcc_delta_delta"
generate_ground_truth(labels,ids,"./ground_truth/mfcc_delta_delta")

## i)
print("mfcc_delta_delta, with, n_components=",64)
with open("./models/mfcc_delta_delta/with/64/gmm.pkl",'rb') as fid:
    gmms = pickle.load(fid)
    generate_hypothesis(features,ids,phonemes,False,64,gmms,save_path)
    print(fer(features,labels,phonemes,False,gmms))
## ii)
print("mfcc_delta_delta, without, n_components=",64)
with open("./models/mfcc_delta_delta/without/64/gmm.pkl",'rb') as fid:
    gmms = pickle.load(fid)
    generate_hypothesis(features,ids,phonemes,True,64,gmms,save_path)
    print(fer(features,labels,phonemes,True,gmms))
