import os
import math 
import argparse 
import librosa
import numpy as np
from scipy.io import wavfile
import tensorflow
from tensorflow.keras.models import load_model

SLICE_SIZE = 2048
SAMPLE_RATE = 16000
MODEL_PATH = 'models/model.hdf5'
if 'DENOISE_MODEL_PATH' in os.environ:
	MODEL_PATH = os.environ['DENOISE_MODEL_PATH']

model = None

def _model_predict(slice):
    return np.array(model.predict(slice.reshape((1, SLICE_SIZE)))[0]).reshape((SLICE_SIZE))

def _denoise_audio(audio, result):
    slice_count = audio.shape[0] // SLICE_SIZE
    for i in range(0, slice_count * SLICE_SIZE, SLICE_SIZE):
        result[i:i+SLICE_SIZE] += _model_predict(audio[i:i+SLICE_SIZE])
        
def peak_amplitude(samples):
    return max(abs(min(samples)), abs(max(samples)))
	
def denoise_audio(audio):
    peak = peak_amplitude(audio)
    npad = math.ceil(audio.shape[0] / SLICE_SIZE) * SLICE_SIZE
    padded = np.pad(audio, (0, npad), 'reflect')
    result = np.zeros(padded.shape)
    _denoise_audio(padded / peak, result)
    return result[:audio.shape[0]] * peak

def denoise_audio_multiple(audio, n=8):
    peak = peak_amplitude(audio)
    result = np.zeros((audio.shape[0] + SLICE_SIZE))
    for offset in range(0, SLICE_SIZE - 1, SLICE_SIZE // n):
        _denoise_audio(audio[offset:] / peak, result[offset:])
    clipped = result[:audio.shape[0]]
    average = clipped / math.ceil(SLICE_SIZE / (SLICE_SIZE // n))
    return average  * peak

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description ='Denoise wav files.') 
	parser.add_argument('input', help='The input wav file.')
	parser.add_argument('-o', dest ='output', required = True, 
						action ='store', help ='The output wav file.') 
	parser.add_argument('--rounds', dest ='rounds',  
						action ='store', default ='1') 
	args = parser.parse_args() 

	noisy, _ = librosa.load(args.input, sr=SAMPLE_RATE)
	model = load_model(MODEL_PATH)
	rounds = int(args.rounds)
	if rounds > 1:
		clean = denoise_audio_multiple(noisy, rounds)
	else:
		clean = denoise_audio(noisy)	
	wavfile.write(args.output, SAMPLE_RATE, clean)