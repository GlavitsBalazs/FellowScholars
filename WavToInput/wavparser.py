import os
import librosa

PATH1='res/noisy_testset_wav'
files1=os.listdir(PATH1)
PATH2='res/clean_testset_wav'
files2=os.listdir(PATH2)

noisys=[]
for idx, file in enumerate(files1):
    if idx < 30:
        y, sr = librosa.load(PATH1 + '/' + file)
        noisys.append(y)

for i in noisys:
    print(len(i))

