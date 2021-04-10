#!/usr/bin/env python
import statistics
import matplotlib.pyplot as plt
import numpy as np
import sys
import pathlib
from scipy.io import wavfile

import os

# disable tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from lib.config import ModelConfig

if len(sys.argv) < 2:
    print("Usage: python wpm-detect.py <wav-file>")
    sys.exit(1)

wavfile_path = sys.argv[1]

print('Loading libraries...')

data = wavfile.read(wavfile_path)

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

  return spectrogram  

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# TensorFlow compatible function for determining the peak frequency
def get_max(spectrogram):
    return tf.math.argmax(spectrogram, axis=1)

audio_binary = tf.io.read_file(wavfile_path)
waveform = decode_audio(audio_binary)
spectrogram = get_spectrogram(waveform)

print('Separating wav file into individual letters...')

from scipy.io import wavfile
data = wavfile.read(wavfile_path)

symbols = 1
state = "LOW"
for frame in spectrogram.numpy()[0:128]:
    sample = frame[0]
    print(sample)

    if state == "LOW":
        if sample > 10.0:
            state = "HIGH"
            symbols += 1
    elif state == "HIGH":
        if sample < 10.0:
            state = "LOW"
            symbols += 1

print(symbols)
