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
  waveform = tf.cast(waveform, tf.float32)
  #waveform = tf.concat([waveform, zero_padding], 0)

  spectrogram = tf.signal.stft(
      waveform, frame_length=256, frame_step=128)
  print(spectrogram.shape)  

  spectrogram = tf.abs(spectrogram)
  maxes = get_max(spectrogram)
    
  # get the most common high power  
  counts = tf.unique_with_counts(maxes)
  max_power = counts[0][tf.math.argmax(counts[2])]
  
  # cast properly for the tf.slice command
  max_power = tf.cast(max_power, tf.int32)

  spectrogram = tf.slice(spectrogram, begin=[0, max_power], size=[-1, 1])

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
