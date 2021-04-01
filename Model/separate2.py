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

def get_spectrogram_tf(waveform):
  # Padding for files with less than 256000 samples
  #print("Len: {}".format(tf.shape(waveform)))
  zero_padding = tf.zeros([1000000] - tf.shape(waveform), dtype=tf.float32)

  # Concatenate audio with padding so that all audio clips will be of the 
  # same length
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)

  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  get_max(spectrogram)
  spectrogram = tf.abs(spectrogram)
  maxes = get_max(spectrogram)

  # Remove any zeros from the "maxes"  
  boolean_mask = tf.cast(maxes, dtype=tf.bool)              
  no_zeros = tf.boolean_mask(maxes, boolean_mask, axis=0)

  counts = tf.unique_with_counts(no_zeros)    
  median_max = counts[0][0]
    
  # cast properly for the tf.slice command
  median_max = tf.cast(median_max, tf.int32)
  print("Median Max: {}".format(median_max))

  #tf.print("median max: ", median_max)
  #print("Median: {}".format(median_max))
  spectrogram = tf.slice(spectrogram, begin=[0, median_max], size=[-1, 1])

  return spectrogram

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# TensorFlow compatible function for determining the peak frequency
def get_max(spectrogram):  
    max_seq_len = spectrogram.shape[1]
    maxes = tf.TensorArray(tf.int64, size=max_seq_len)
    for i in tf.range(max_seq_len):
        max_n = tf.math.argmax(tf.cast(spectrogram[i], tf.int64))
        maxes = maxes.write(i, max_n)
    return maxes.stack()

audio_binary = tf.io.read_file(str(data_path/'test.wav'))
waveform = decode_audio(audio_binary)
spectrogram = get_spectrogram_tf(waveform)
print(spectrogram.numpy()[0:100])

from scipy.io import wavfile
data = wavfile.read(str(data_path/'test.wav'))

output_data = []

low_count = 0
cursor = 0
state = "OUT_OF_LETTER"
i = 0
for frame in spectrogram.numpy():
    sample = frame[0]
    print(sample)
    if state == "OUT_OF_LETTER":
        if sample > 5.0:
            prev_state = state
            state = "IN_LETTER"
            print(state)
            low_count = 0
            output_data = []
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
            print(state)
            output = np.array(output_data, dtype=np.int16)
            wavfile.write(str(data_path/"output-{:04d}.wav".format(i)), 8000, output)
            i += 1
    
    cursor += 128
