import statistics
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys
import pathlib
from scipy.io import wavfile

from lib.config import ModelConfig

if len(sys.argv) < 2:
    print("Usage: python train-model.py <model-yaml>")
    sys.exit(1)

config = ModelConfig(sys.argv[1])

data_path = pathlib.Path(config.value('system.volumes.test'))

data = wavfile.read(str(data_path/'test.wav'))

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_spectrogram(waveform):
  # Padding for files with less than 256000 samples
  #print("Len: {}".format(tf.shape(waveform)))
  print(tf.shape(waveform))
  l = tf.shape(waveform) / [128]
  r = tf.shape(waveform) % [128]

  zero_padding = 11630+230778 - l[0]
  if r == 0:
      zero_padding -= 1

  # Concatenate audio with padding so that all audio clips will be of the 
  # same length
  waveform = tf.cast(waveform, tf.float32)
  #waveform = tf.concat([waveform, zero_padding], 0)

  spectrogram = tf.signal.stft(
      waveform, frame_length=256, frame_step=128)

  spectrogram = tf.abs(spectrogram)
  maxes = get_max(spectrogram)
    
  # get the most common high power  
  counts = tf.unique_with_counts(maxes)
  max_power = counts[0][tf.math.argmax(counts[2])]
  
  # cast properly for the tf.slice command
  max_power = tf.cast(max_power, tf.int32)

  spectrogram = tf.slice(spectrogram, begin=[0, max_power], size=[-1, 1])

  spectrogram = tf.pad(spectrogram, [[0, zero_padding],[0, 0]])
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

from scipy.io import wavfile
data = wavfile.read(str(data_path/'test.wav'))

output_data = []

low_count = 0
cursor = 0
state = "OUT_OF_LETTER"
space_count = 0
i = 0
for frame in spectrogram.numpy():
    sample = frame[0]
    #print(sample)
    if state == "IN_SPACE":
        if sample > 15.0:
            prev_state = state
            state = "IN_LETTER"
            low_count = 0
            output_data = []
    if state == "OUT_OF_LETTER":
        if space_count > 70:
            wavfile.write(str(data_path/"output-{:04d}.wav".format(i)), 8000, np.zeros(5000).astype(np.int16))
            prev_state = state
            state = "IN_SPACE"
            i += 1
        if sample > 5.0:
            prev_state = state
            state = "IN_LETTER"
            #print(state)
            low_count = 0
            output_data = []
        else:
            space_count += 1
    elif state == "IN_LETTER":
        chunk = data[1][cursor:cursor+128]
        output_data = np.concatenate((output_data,chunk))

        if sample < 10.0:
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
