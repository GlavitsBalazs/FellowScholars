import os
import librosa
import matplotlib.pyplot as plt
import numpy as np


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


noisy_trainset_directory = "datasets/DS_10283_2791/noisy_trainset_28spk_wav"
clean_trainset_directory = "datasets/DS_10283_2791/clean_trainset_28spk_wav"


def find_closest_speech_lengths(target_count, max_file_count=None):
    noisy_files = []
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

    for filename, duration in target_noisy_files:
        noisy_samples, _ = librosa.load(os.path.join(noisy_trainset_directory, filename))
        clean_samples, _ = librosa.load(os.path.join(clean_trainset_directory, filename))
        plt.plot(noisy_samples)
        plt.figtext(0.5, 0.01, "noisy " + filename)
        plt.show()
        plt.plot(clean_samples)
        plt.figtext(0.5, 0.01, "clean " + filename)
        plt.show()


find_closest_speech_lengths(7, 20)
