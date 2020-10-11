import os
import librosa
import matplotlib.pyplot as plt
import numpy as np

PATH1='res/noisy_testset_wav'
files1=os.listdir(PATH1)
PATH2='res/clean_testset_wav'
files2=os.listdir(PATH2)

def print_len_of_speech(maxidx=30):
    noisys=[]
    for idx, file in enumerate(files1):
        if idx < maxidx:
            y, sr = librosa.load(PATH1 + '/' + file)
            noisys.append(y)

    for i in noisys:
        print(len(i))

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


def find_closest_speech_lengths(how_many, maxidx = None):
    if maxidx is None:
        noisys = [[0]] * len(files1)
    else:
        noisys = [[0]] * maxidx

    lens=[0] * len(noisys)
    #read all noisy speech
    for idx, file in enumerate(files1):
        if maxidx is not None:
            if idx < maxidx:
                y, sr = librosa.load(PATH1 + '/' + file)
                noisys[idx] = y
                lens[idx] = len(y)
        else:
            y, sr = librosa.load(PATH1 + '/' + file)
            noisys[idx] = y
            lens[idx] = len(y)

    #sort the lengths array
    lens.sort()

    #find the n speech closest to each other
    minlendiff=9999999999
    minlenidx=0
    for i in range(len(lens)-how_many):
        songlendiff=lens[i+how_many]-lens[i]
        if songlendiff<minlendiff:
            minlendiff=songlendiff
            minlenidx=i

    #put the good lengths into an array
    goodlens=[0] * how_many
    idx=0
    for i in range(minlenidx, minlenidx+how_many):
        goodlens[idx]=lens[i]
        idx+=1

    #put the speechs with the good lengths into an array
    goodlenspeech=[[0]] * how_many
    idx=0
    for i in noisys:
        if goodlens.__contains__(len(i)):
            goodlenspeech[idx]=i
            idx+=1

    #parse the clean speeches with good lengths
    cleans=[[0]] * how_many
    idx=0
    for i, file in enumerate(files2):
        if maxidx is not None:
            if i < maxidx:
                y, sr = librosa.load(PATH2 + '/' + file)
                if goodlens.__contains__(len(y)):
                    cleans[idx] = y
                    idx += 1
        else:
            y, sr = librosa.load(PATH2 + '/' + file)
            if goodlens.__contains__(len(y)):
                cleans[idx] = y
                idx += 1

    #point the pointer to the right direction
    noisys=goodlenspeech

    for i in noisys:
        plt.plot(i)
        plt.show()

    for i in cleans:
        plt.plot(i)
        plt.show()


find_closest_speech_lengths(7, 20)