import os
import numpy as np
import librosa
import itertools
from scipy.io import wavfile
from sklearn.model_selection import train_test_split


def peak_amplitude(samples):
    return max(abs(min(samples)), abs(max(samples)))


def mulaw_encode(samples):
    # Rescale to -1.0..1.0. Encode to -128..127. Return 0..255.
    return (librosa.mu_compress(samples / peak_amplitude(samples), quantize=True) + 128).astype('uint8')


def mulaw_decode(samples):
    # Rescale from 0..255 to -128..127. Decode to -1.0..1.0. Return -2**15-1..2**15-1.
    return (librosa.mu_expand(samples.astype('int16') - 128, quantize=True) * (2 ** 15 - 1)).astype('int16')


# A pair of clean speech and noisy speech data.
class DataPoint:
    samples_per_second = 48000
    riff_header_bytes = 44
    bytes_per_sample = 2

    def __init__(self, clean_path, noisy_path):
        self.name = os.path.basename(clean_path)
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        clean_file_bytes = os.path.getsize(clean_path)
        self.sample_count = (clean_file_bytes - self.riff_header_bytes) // self.bytes_per_sample
        self.duration = self.sample_count / self.samples_per_second
        self.clean_audio = None
        self.noisy_audio = None

    def load_audio(self):
        if self.clean_audio is None and self.noisy_audio is None:
            # Use scipy.io.wavfile instead of librosa, because it supports integer samples.
            sr_clean, self.clean_audio = wavfile.read(self.clean_path)
            sr_noisy, self.noisy_audio = wavfile.read(self.noisy_path)
            assert len(self.noisy_audio) == len(self.clean_audio) == self.sample_count
            assert sr_clean == sr_noisy == self.samples_per_second
            self.clean_audio = mulaw_encode(self.clean_audio)
            self.noisy_audio = mulaw_encode(self.noisy_audio)


class Dataset:
    def __init__(self):
        self.noisy_train = None
        self.clean_train = None
        self.noisy_val = None
        self.clean_val = None
        self.noisy_test = None
        self.clean_test = None

    def load(self, train_data, test_data, validation_split=0.2):
        training_data, validation_data = train_test_split(train_data, test_size=validation_split)
        # expand dims: Keras expects 3d tensors
        self.noisy_train = np.expand_dims([dp.noisy_audio for dp in training_data], axis=2)
        self.clean_train = np.expand_dims([dp.clean_audio for dp in training_data], axis=2)
        self.noisy_val = np.expand_dims([dp.noisy_audio for dp in validation_data], axis=2)
        self.clean_val = np.expand_dims([dp.clean_audio for dp in validation_data], axis=2)
        self.noisy_test = np.expand_dims([dp.noisy_audio for dp in test_data], axis=2)
        self.clean_test = np.expand_dims([dp.clean_audio for dp in test_data], axis=2)

    def load_npy(self, directory):
        self.noisy_train = np.load(os.path.join(directory, 'noisy_train.npy'))
        self.clean_train = np.load(os.path.join(directory, 'clean_train.npy'))
        self.noisy_val = np.load(os.path.join(directory, 'noisy_val.npy'))
        self.clean_val = np.load(os.path.join(directory, 'clean_val.npy'))
        self.noisy_test = np.load(os.path.join(directory, 'noisy_test.npy'))
        self.clean_test = np.load(os.path.join(directory, 'clean_test.npy'))

    def export_npy(self, directory):
        np.save(os.path.join(directory, 'noisy_train.npy'), self.noisy_train)
        np.save(os.path.join(directory, 'clean_train.npy'), self.clean_train)
        np.save(os.path.join(directory, 'noisy_val.npy'), self.noisy_val)
        np.save(os.path.join(directory, 'clean_val.npy'), self.clean_val)
        np.save(os.path.join(directory, 'noisy_test.npy'), self.noisy_test)
        np.save(os.path.join(directory, 'clean_test.npy'), self.clean_test)


def find_closest_speech_lengths(data_points, target_count):
    # Find the subset of target count with minimum duration difference.
    data_points.sort(key=lambda t: t.duration)
    min_i = np.argmin([data_points[i + target_count].duration - data_points[i].duration
                       for i in range(len(data_points) - target_count)])
    return data_points[min_i: min_i + target_count]


def zero_pad(data_points, target):
    for dp in data_points:
        pad = target - dp.sample_count
        if pad > 0:
            dp.sample_count = target
            dp.duration = dp.sample_count / dp.samples_per_second
            dp.clean_audio = np.pad(dp.clean_audio, (0, pad), 'constant')
            dp.noisy_audio = np.pad(dp.noisy_audio, (0, pad), 'constant')


def load_training_data():
    clean_trainset_28spk_directory = "datasets/DS_10283_2791/clean_trainset_28spk_wav"
    noisy_trainset_28spk_directory = "datasets/DS_10283_2791/noisy_trainset_28spk_wav"
    clean_trainset_56spk_directory = "datasets/DS_10283_2791/clean_trainset_56spk_wav"
    noisy_trainset_56spk_directory = "datasets/DS_10283_2791/noisy_trainset_56spk_wav"
    return [DataPoint(os.path.join(dirs[0], filename), os.path.join(dirs[1], filename))
            for dirs in [(clean_trainset_28spk_directory, noisy_trainset_28spk_directory),
                         (clean_trainset_56spk_directory, noisy_trainset_56spk_directory)]
            for filename in os.listdir(dirs[0])]


def load_test_data():
    clean_testset_directory = "datasets/DS_10283_2791/clean_testset_wav"
    noisy_testset_directory = "datasets/DS_10283_2791/noisy_testset_wav"
    return [DataPoint(os.path.join(clean_testset_directory, filename),
                      os.path.join(noisy_testset_directory, filename))
            for filename in os.listdir(clean_testset_directory)]


def onehot_256(value):
    return np.eye(256)[value]


# Generate mini-batches of (noisy slice, encoded clean sample) pairs.
def training_data_generator(data_points, slice_size, minibatch_size, slice_density=1.0, output_encoding=onehot_256):
    minibatch = []
    for dp in data_points:
        dp.load_audio()
        slice_count = int(dp.sample_count / slice_size * slice_density)
        for _ in range(slice_count):
            i = np.random.randint(dp.sample_count - slice_size - 1)
            noisy_slice = dp.noisy_audio[i:i + slice_size]
            next_clean_sample = dp.clean_audio[i + slice_size]
            noisy_slice = np.expand_dims(noisy_slice, axis=2) # 3D tensors for Keras
            next_clean_sample = output_encoding(next_clean_sample)
            minibatch.append((noisy_slice, next_clean_sample))
            if len(minibatch) == minibatch_size:
                yield minibatch
                minibatch = []


if __name__ == '__main__':
    print("Loading: 0%")
    training_data = load_training_data()
    test_data = load_test_data()
    closest_data = find_closest_speech_lengths(training_data, 1000)
    max_length = np.max([dp.sample_count for dp in closest_data])
    test_data = [dp for dp in test_data if dp.sample_count <= max_length]
    for i, dp in enumerate(itertools.chain(closest_data, test_data)):
        dp.load_audio()
        percentage = round(100 * (i + 1) / (len(closest_data) + len(test_data)))
        if i % 10 == 0:
            print("Loading: {}%".format(percentage))
    zero_pad(closest_data, max_length)
    zero_pad(test_data, max_length)
    dataset = Dataset()
    dataset.load(closest_data, test_data)
    dataset.export_npy('datasets\saves')
