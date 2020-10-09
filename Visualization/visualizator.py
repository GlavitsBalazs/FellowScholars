import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

PATH1='res/noisy_testset_wav'
files1=os.listdir(PATH1)
PATH2='res/clean_testset_wav'
files2=os.listdir(PATH2)

y, sr = librosa.load(PATH1 + '/' + files1[0])
plt.specgram(y, Fs=sr)
plt.show()
plt.plot(y)
plt.show()

y, sr = librosa.load(PATH2 + '/' + files2[0])
plt.specgram(y, Fs=sr)
plt.show()
plt.plot(y)
plt.show()



