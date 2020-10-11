import os
import librosa
import numpy as np

def load_data(path, n=None, max_len=None):
    files=os.listdir(path)
    if (n == None):
        nb_sample = len(files)
    else:
        nb_sample = n
    if max_len == None:
        data=np.empty((nb_sample,),dtype=np.ndarray)
    else:
        data = np.zeros((nb_sample,max_len))
        print(data)
    for idx, file in enumerate(files):
        if idx >= nb_sample:
            break
        y, sr = librosa.load(path + '/' + file)
        if max_len == None:
            data[idx] = y
        else:
            data[idx,0:len(y)] = y
    return data
