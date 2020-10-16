import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
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
    def find_instances():
        clean_trainset_28spk_directory = "datasets/DS_10283_2791/clean_trainset_28spk_wav"
        noisy_trainset_28spk_directory = "datasets/DS_10283_2791/noisy_trainset_28spk_wav"
        clean_trainset_56spk_directory = "datasets/DS_10283_2791/clean_trainset_56spk_wav"
        noisy_trainset_56spk_directory = "datasets/DS_10283_2791/noisy_trainset_56spk_wav"
        return [DataPoint(os.path.join(dirs[0], filename), os.path.join(dirs[1], filename))
                for dirs in [(clean_trainset_28spk_directory, noisy_trainset_28spk_directory),
                             (clean_trainset_56spk_directory, noisy_trainset_56spk_directory)]
                for filename in os.listdir(dirs[0])]

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
            sr_clean, self.clean_audio = wavfile.read(self.clean_path)
            sr_noisy, self.noisy_audio = wavfile.read(self.noisy_path)
            assert len(self.noisy_audio) == len(self.clean_audio) == self.sample_count
            assert sr_clean == sr_noisy == self.samples_per_second
            self.clean_audio = DataPoint.mulaw_encode(self.clean_audio)
            self.noisy_audio = DataPoint.mulaw_encode(self.noisy_audio)


class Dataset:
    def __init__(self, data_points):
        train_data, test_data = train_test_split(data_points, test_size=0.2)
        self.training = train_data
        self.testing = test_data
        self.validation = data_points

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


def plot_durations_histogram(data_points):
    durations = [dp.duration for dp in data_points]
    # Histogram from the minimum to the 98-th percentile, to hide the outliers.
    percentiles = np.percentile(durations, [0, 50, 98, 100])
    print(percentiles)  # The median and the maximum are also of interest.
    _, ax = plt.subplots()
    bins = round(np.sqrt([len(data_points)])[0])
    ax.hist(durations, bins=np.linspace(percentiles[0], percentiles[2], num=bins))
    plt.show()


def plot_waveforms(data_points):
    for i, dp in enumerate(data_points):
        plt.plot(dp.noisy_audio)
        plt.figtext(0.5, 0.01, "noisy {}".format(i))
        plt.show()
        plt.plot(dp.clean_audio)
        plt.figtext(0.5, 0.01, "clean {}".format(i))
        plt.show()


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


if __name__ == '__main__':
    print("Loading: 0%")
    all_data_points = DataPoint.find_instances()
    plot_durations_histogram(all_data_points)
    closest_data = find_closest_speech_lengths(all_data_points, 1000)
    plot_durations_histogram(closest_data)
    for i, dp in enumerate(closest_data):
        dp.load_audio()
        percentage = round(100 * (i + 1) / len(closest_data))
        if i % 10 == 0:
            print("Loading: {}%".format(percentage))
    zero_pad(closest_data)
    dataset = Dataset(closest_data)
    dataset.export('datasets/saves')
