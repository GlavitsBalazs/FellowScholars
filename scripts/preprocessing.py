import os
import numpy as np
import librosa
import itertools
from scipy.io import wavfile
from sklearn.model_selection import train_test_split


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

    @staticmethod
    def mulaw_encode(samples):
        # Rescale to -1.0..1.0. Encode to -128..127. Return 0..255.
        max_amplitude = max(abs(samples.min()), abs(samples.max()))
        return (librosa.mu_compress(samples / max_amplitude, quantize=True) + 128).astype('uint8')

    @staticmethod
    def mulaw_decode(samples):
        # Rescale from 0..255 to -128..127. Decode to -1.0..1.0. Return -2**15-1..2**15-1.
        return (librosa.mu_expand(samples.astype('int16') - 128, quantize=True) * (2 ** 15 - 1)).astype('int16')

    def load_audio(self):
        if self.clean_audio is None and self.noisy_audio is None:
            # Use scipy.io.wavfile instead of librosa, because it supports integer samples.
            sr_clean, self.clean_audio = wavfile.read(self.clean_path)
            sr_noisy, self.noisy_audio = wavfile.read(self.noisy_path)
            assert len(self.noisy_audio) == len(self.clean_audio) == self.sample_count
            assert sr_clean == sr_noisy == self.samples_per_second
            self.clean_audio = DataPoint.mulaw_encode(self.clean_audio)
            self.noisy_audio = DataPoint.mulaw_encode(self.noisy_audio)


class Dataset:
    def __init__(self, train_data, test_data, validation_split=0.2):
        training_data, validation_data = train_test_split(train_data, test_size=validation_split)
        self.training = training_data
        self.testing = test_data
        self.validation = validation_data

    def export(self, directory):
        def export_clean_and_noisy(data_points, name):
            # Keras expects 3D tensors.
            clean = np.expand_dims([dp.clean_audio for dp in data_points], axis=2)
            np.save(os.path.join(directory, 'clean_' + name), clean)
            noisy = np.expand_dims([dp.noisy_audio for dp in data_points], axis=2)
            np.save(os.path.join(directory, 'noisy_' + name), noisy)

        export_clean_and_noisy(self.training, 'train.npy')
        export_clean_and_noisy(self.testing, 'test.npy')
        export_clean_and_noisy(self.validation, 'val.npy')

    @staticmethod
    def load_numpy_data(directory):
        noisy_train = np.load(os.path.join(directory, 'noisy_train.npy'))
        clean_train = np.load(os.path.join(directory, 'clean_train.npy'))
        noisy_val = np.load(os.path.join(directory, 'noisy_val.npy'))
        clean_val = np.load(os.path.join(directory, 'clean_val.npy'))
        noisy_test = np.load(os.path.join(directory, 'noisy_test.npy'))
        clean_test = np.load(os.path.join(directory, 'clean_test.npy'))
        return noisy_train, clean_train, noisy_val, clean_val, noisy_test, clean_test


def find_closest_speech_lengths(data_points, target_count):
    # Find he subset of target count with minimum duration difference.
    data_points.sort(key=lambda t: t.duration)
    min_i = np.argmin([data_points[i + target_count].duration - data_points[i].duration
                       for i in range(len(data_points) - target_count)])
    return data_points[min_i: min_i + target_count]


def zero_pad(data_points):
    max_sample_count = np.max([dp.sample_count for dp in data_points])
    for dp in data_points:
        pad = max_sample_count - dp.sample_count
        if pad > 0:
            dp.sample_count = max_sample_count
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


if __name__ == '__main__':
    print("Loading: 0%")
    training_data = load_training_data()
    test_data = load_test_data()
    closest_data = find_closest_speech_lengths(training_data, 1000)
    for i, dp in enumerate(itertools.chain(closest_data, test_data)):
        dp.load_audio()
        percentage = round(100 * (i + 1) / (len(closest_data) + len(test_data)))
        if i % 10 == 0:
            print("Loading: {}%".format(percentage))
    zero_pad(closest_data)
    zero_pad(test_data)  # Issue: the test data points will be longer than the train data points.
    dataset = Dataset(closest_data, test_data)
    dataset.export('datasets/saves')
