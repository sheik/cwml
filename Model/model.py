#!/usr/bin/env python

import os
import sys
import pathlib
import random
import glob

# disable tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import statistics

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

from lib.config import ModelConfig

if len(sys.argv) < 2:
    print("Usage: python train-model.py <model-yaml>")
    sys.exit(1)

config = ModelConfig(sys.argv[1])

if config.value('system.gpu_enabled'):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

print("Loading libraries...")

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# data directory
data_dir = pathlib.Path(config.value('system.volumes.data'))
test_data_dir = pathlib.Path(config.value('system.volumes.test'))

checkpoint_path = config.value('system.volumes.model') + "/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
latest = tf.train.latest_checkpoint(checkpoint_dir)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


symbols = np.array(tf.io.gfile.listdir(str(data_dir)))

# get filenames, shuffle
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)

# split the data
num_train = int(num_samples * 0.9)
num_test = int(num_samples * 0.1)
num_val = int(num_samples * 0.0)
train_files = filenames[:num_train]
val_files = filenames[num_train: num_train + num_val]
test_files = filenames[num_train+num_val:]

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  return parts[-2]

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

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

  # calculate window 
  start = max_power
  if max_power >= 1:
      start = max_power - 1

  size = 3
  if spectrogram.shape[1] - start < size:
      size = spectrogram.shape[1] - start

  # create a window around the strongest signal
  spectrogram = tf.slice(spectrogram, begin=[0, start], size=[-1, size])

  # pad tensor so all tensors are euqal shape
  spectrogram = pad_up_to(spectrogram, [330, 3], 0)

  return spectrogram  

# TensorFlow compatible function for determining the peak frequency
def get_max(spectrogram):
    return tf.math.argmax(spectrogram, axis=1)

def pad_up_to(t, max_in_dims, constant_values):
    s = tf.shape(t)
    paddings = [[0, m-s[i]] for (i,m) in enumerate(max_in_dims)]
    return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  spectrogram = tf.expand_dims(spectrogram, -1)
  label_id = tf.argmax(label == symbols)
  return spectrogram, label_id

spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds


# Separate datasets for training, validation, and testing
train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

batch_size = 64 
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

# TODO do some testing with cache / prefetch
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

input_shape = (0, 0, 0)
for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = tf.math.maximum(input_shape, spectrogram.shape)

num_labels = len(symbols)

model = None

model = models.Sequential([
    layers.Input(shape=input_shape),
    preprocessing.Resizing(config.value('model.resize.x'), config.value('model.resize.y')), 
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

#model.summary()

print("Compiling model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )

if latest:
    print("Loading model weights...")
    model.load_weights(latest)
else:
    print("Training model...")
    history = model.fit(
        train_ds, 
        validation_data=val_ds,  
        epochs=config.value('model.epochs'),
        callbacks=[tf.keras.callbacks.EarlyStopping(verbose=1, patience=config.value('model.patience'))],
    )

if not latest:
    test_audio = []
    test_labels = []

    for audio, label in test_ds:
      test_audio.append(audio.numpy())
      test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')

model.save_weights(checkpoint_path)

# test inference
files = sorted(glob.glob(str(test_data_dir/"output*.wav")))

sample_ds = preprocess_dataset([str(f) for f in files])

for spectrogram, label in sample_ds.batch(1):
    try:
        prediction = model(spectrogram)
        predictions = zip(symbols, prediction[0])
        letter = max(predictions, key=lambda x: x[1])[0]
        letter = ' ' if letter == '_' else letter
        print(letter, end='', flush=True)
    except:
        pass

print("\n")

