import os
import librosa
import matplotlib.pyplot as plt
import numpy as np


noisy_trainset_directory = "datasets/DS_10283_2791/noisy_trainset_28spk_wav"
clean_trainset_directory = "datasets/DS_10283_2791/clean_trainset_28spk_wav"


def load_data(path, max_file_count=None, max_sample_count=None):
    files = os.listdir(path)
    if max_file_count is None:
        max_file_count = len(files)
    if max_sample_count is None:
        data = np.empty((max_file_count,), dtype=np.ndarray)
    else:
        data = np.zeros((max_file_count, max_sample_count))
        print(data)
    for i, file in enumerate(files):
        if i >= max_file_count:
            break
        samples, sample_rate = librosa.load(os.path.join(path, file))
        if max_sample_count is None:
            data[i] = samples
        else:
            data[i, 0:len(samples)] = samples
    return data


def visualize(noisy, clean, n=None):
    if len(noisy) != len(clean):
        return
    if n is None:
        n = len(noisy)
    if n > len(noisy):
        n = len(noisy)
    for i in range(n):
        plt.plot(noisy[i])
        plt.figtext(0.5, 0.01, "noisy {}".format(i))
        plt.show()
        plt.plot(clean[i])
        plt.figtext(0.5, 0.01, "clean {}".format(i))
        plt.show()


def zero_pad(noisy, clean):
    ret_noisy = np.copy(noisy)
    ret_clean = np.copy(clean)
    if len(noisy) != len(clean):
        return
    max_n = 0
    for i in noisy:
        if len(i) > max_n:
            max_n = len(i)
    for i in range(len(noisy)):
        pad = max_n - len(ret_noisy[i])
        if pad > 0:
            ret_noisy[i] = np.pad(ret_noisy[i], (0,pad), 'constant')
            ret_clean[i] = np.pad(ret_clean[i], (0,pad), 'constant')

    return ret_noisy, ret_clean


def find_closest_speech_lengths(target_count, max_file_count=None, visualize=False):
    noisy_files = []
    if max_file_count is None:
        files=os.listdir(noisy_trainset_directory)
        max_file_count = len(files)
    for i, filename in enumerate(os.listdir(noisy_trainset_directory)):
        if (max_file_count is not None) and i < max_file_count:
            duration = librosa.get_duration(filename=os.path.join(noisy_trainset_directory, filename))
            noisy_files.append((filename, duration))
        else:
            break
    noisy_files.sort(key=lambda x: x[1])
    print(noisy_files)
    
    # Find the subset of target count with minimum duration difference.
    min_i = np.argmin([noisy_files[i + target_count][1] - noisy_files[i][1]
                       for i in range(len(noisy_files) - target_count)])
    target_noisy_files = noisy_files[min_i: min_i + target_count]
    noisy_samples=np.empty((target_count,), dtype=np.ndarray)
    clean_samples=np.empty((target_count,), dtype=np.ndarray)
    i = 0
    for filename, duration in target_noisy_files:
            noisy_samples[i], _ = librosa.load(os.path.join(noisy_trainset_directory, filename))
            clean_samples[i], _ = librosa.load(os.path.join(clean_trainset_directory, filename))
            i += 1
    return noisy_samples, clean_samples


#noisy, clean = find_closest_speech_lengths(10000)
noisy, clean = find_closest_speech_lengths(7, 20)
noisy_padded, clean_padded = zero_pad(noisy, clean)
visualize(noisy, clean, 10)
visualize(noisy_padded, clean_padded, 10)
