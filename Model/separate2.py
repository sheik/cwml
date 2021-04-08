import statistics
import matplotlib.pyplot as plt
import numpy as np
import sys
import pathlib
from scipy.io import wavfile

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from lib.config import ModelConfig

if len(sys.argv) < 2:
    print("Usage: python train-model.py <model-yaml>")
    sys.exit(1)

config = ModelConfig(sys.argv[1])

print('Loading libraries...')

data_path = pathlib.Path(config.value('system.volumes.test'))

data = wavfile.read(str(data_path/'test.wav'))

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_spectrogram(waveform):  
  # cast waveform for tf.signal.stft
  waveform = tf.cast(waveform, tf.float32)

  # generate spectrogram
  spectrogram = tf.signal.stft(
      waveform, frame_length=256, frame_step=128)

  # get the powers out of the STFT 
  spectrogram = tf.abs(spectrogram)

  maxes = get_max(spectrogram)

  boolean_mask = tf.cast(maxes, dtype=tf.bool)              
  no_zeros = tf.boolean_mask(maxes, boolean_mask, axis=0)

  if no_zeros.shape[0] == 0:
      no_zeros = maxes

  # get the most common high power  
  counts = tf.unique_with_counts(no_zeros)

  # get the "max power", avoid root errors by setting to 0
  # by default
  if len(counts[0]) > 0:
    max_power = counts[0][tf.math.argmax(counts[2])]
  else:
    max_power = tf.cast(0, tf.int64)

  # cast properly for the tf.slice command
  max_power = tf.cast(max_power, tf.int32)

  # create a window around the strongest signal
  spectrogram = tf.slice(spectrogram, begin=[0, max_power], size=[-1, 1])

  # pad tensor so all tensors are euqal shape
  #spectrogram = pad_up_to(spectrogram, [330, 3], 0)

  return spectrogram  

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# TensorFlow compatible function for determining the peak frequency
def get_max(spectrogram):
    return tf.math.argmax(spectrogram, axis=1)

audio_binary = tf.io.read_file(str(data_path/'test.wav'))
waveform = decode_audio(audio_binary)
spectrogram = get_spectrogram(waveform)
#print(spectrogram.numpy()[0:100])

print('Separating wav file into individual letters...')

from scipy.io import wavfile
data = wavfile.read(str(data_path/'test.wav'))

output_data = []

low_count = 0
cursor = 0
state = "OUT_OF_LETTER"
space_count = 0
i = 0
max_space_count = 25 
for frame in spectrogram.numpy():
    chunk = data[1][cursor:cursor+128]
    sample = frame[0]
    #print(sample)
    if state == "IN_SPACE":
        if sample > 15.0:
            prev_state = state
            state = "IN_LETTER"
            low_count = 0
            output_data = []
        space_count += 1
        max_space_count = max(space_count, max_space_count)
    if state == "OUT_OF_LETTER":
        if space_count > max_space_count - 5:
            wavfile.write(str(data_path/"output-{:04d}.wav".format(i)), 8000, np.zeros(5000).astype(np.int16))
            prev_state = state
            state = "IN_SPACE"
            i += 1
        if sample > 18.0:
            prev_state = state
            state = "IN_LETTER"
            #print(state)
            low_count = 0
            output_data = []
            #output_data = np.concatenate((output_data,chunk))
        else:
            space_count += 1
            max_space_count -= 0.2
    elif state == "IN_LETTER":
        output_data = np.concatenate((output_data,chunk))
        if sample < 18.0:
            low_count += 1
        else:
            low_count = 0
        
        if low_count > 6:
            prev_state = state
            state = "OUT_OF_LETTER"
            #print(state)
            output = np.array(output_data, dtype=np.int16)
            wavfile.write(str(data_path/"output-{:04d}.wav".format(i)), 8000, output)
            space_count = 0
            i += 1
    
    cursor += 128
