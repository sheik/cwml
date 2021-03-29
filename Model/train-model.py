#!/usr/bin/env python

import os
import sys
import pathlib
import random

# disable tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#tf.autograph.set_verbosity(10)


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

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

#tf.config.threading.set_inter_op_parallelism_threads(12)
#tf.config.threading.set_intra_op_parallelism_threads(12)

# show plots?
show_plots = False

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


"""Check basic statistics about the dataset."""
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
#print('Commands:', commands)


"""Extract the audio files into a list and shuffle it."""

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
#print('Number of total examples:', num_samples)
#print('Number of examples per label:',
 #     len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
#print('Example file tensor:', filenames[0])

# split the data
num_train = int(num_samples * 0.8)
num_test = int(num_samples * 0.1)
num_val = int(num_samples * 0.1)
train_files = filenames[:num_train]
val_files = filenames[num_train: num_train + num_val]
test_files = filenames[num_train+num_val:]

#print('Training set size', len(train_files))
#print('Validation set size', len(val_files))
#print('Test set size', len(test_files))

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

"""The label for each WAV file is its parent directory."""

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)

  # Note: You'll use indexing here instead of tuple unpacking to enable this 
  # to work in a TensorFlow graph.
  return parts[-2]

"""Let's define a method that will take in the filename of the WAV file and output a tuple containing the audio and labels for supervised training."""

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

"""Let's examine a few audio waveforms with their corresponding labels."""


if show_plots:
    rows = 3
    cols = 3
    n = rows*cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
    for i, (audio, label) in enumerate(waveform_ds.take(n)):
      r = i // cols
      c = i % cols
      ax = axes[r][c]
      ax.plot(audio.numpy())
      ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
      label = label.numpy().decode('utf-8')
      ax.set_title(label)

    plt.show()


def get_spectrogram(waveform):
  # Padding for files with less than 256000 samples
  #print("Len: {}".format(tf.shape(waveform)))
  zero_padding = tf.zeros([100000] - tf.shape(waveform), dtype=tf.float32)

  # Concatenate audio with padding so that all audio clips will be of the 
  # same length
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  #spectrogram = tf.expand_dims(equal_length, axis=0)
  spectrogram = tf.abs(spectrogram)

  return spectrogram

"""Next, you will explore the data. Compare the waveform, the spectrogram and the actual audio of one example from the dataset."""

for waveform, label in waveform_ds.take(1):
  label = label.numpy().decode('utf-8')
  spectrogram = get_spectrogram(waveform)

#print('Label:', label)
#print('Waveform shape:', waveform.shape)
#print('Spectrogram shape:', spectrogram.shape)
#print('Audio playback')
#display.display(display.Audio(waveform, rate=8000))

def plot_spectrogram(spectrogram, ax):
  # Convert to frequencies to log scale and transpose so that the time is
  # represented in the x-axis (columns).
  log_spec = np.log(spectrogram.T)
  height = log_spec.shape[0]
  X = np.arange(256000, step=999)
  Y = range(height)
  ax.pcolormesh(spectrogram.T)


if show_plots:
    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(waveform.shape[0])
    axes[0].plot(timescale, waveform.numpy())
    axes[0].set_title('Waveform')
    #axes[0].set_xlim([0, 256000])
    plot_spectrogram(spectrogram.numpy(), axes[1])
    #axes[1].pcolormesh(spectrogram)
    axes[1].set_title('Spectrogram')
    plt.show()

"""Now transform the waveform dataset to have spectrogram images and their corresponding labels as integer IDs."""

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  spectrogram = tf.expand_dims(spectrogram, -1)
  label_id = tf.argmax(label == commands)
  return spectrogram, label_id

spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

"""Examine the spectrogram "images" for different samples of the dataset."""

if show_plots:
    rows = 3
    cols = 3
    n = rows*cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
      r = i // cols
      c = i % cols
      ax = axes[r][c]
      plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
      ax.set_title(commands[label_id.numpy()])
      ax.axis('off')
      
    plt.show()

"""## Build and train the model

Now you can build and train your model. But before you do that, you'll need to repeat the training set preprocessing on the validation and test sets.
"""

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds

train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

"""Batch the training and validation sets for model training."""

batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

# TODO do some testing with cache / prefetch
"""Add dataset [`cache()`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#cache) and [`prefetch()`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch) operations to reduce read latency while training the model."""

#train_ds = train_ds.cache().prefetch(AUTOTUNE)
#val_ds = val_ds.cache().prefetch(AUTOTUNE)

"""For the model, we use a simple convolutional neural network (CNN), since we have transformed the audio files into spectrogram images.
The model also has the following additional preprocessing layers:
- A [`Resizing`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/Resizing) layer to downsample the input to enable the model to train faster.
"""

for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = spectrogram.shape
#print('Input shape:', input_shape)
num_labels = len(commands)

model = None

with tf.device('/gpu:'+str(random.randint(0,7))):
    if config.value('system.multi_gpu_enabled'):
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

        with strategy.scope():
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
    else:
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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
        )

    if latest:
        model.load_weights(latest)
    else:
        history = model.fit(
            train_ds, 
            validation_data=val_ds,  
            epochs=config.value('model.epochs'),
            callbacks=[tf.keras.callbacks.EarlyStopping(verbose=1, patience=config.value('model.patience'))],
        )

        """Let's check the training and validation loss curves to see how your model has improved during training."""

        metrics = history.history
        if 'val_loss' in metrics and show_plots:
            plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
            plt.legend(['loss', 'val_loss'])
            plt.show()


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

# display confusion matrix (only practical for small datasets)
if False:
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands, 
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


# test inference
import glob, os
files = sorted(glob.glob(str(test_data_dir/"output*.wav")))

sample_ds = preprocess_dataset([str(f) for f in files])

for spectrogram, label in sample_ds.batch(1):
    prediction = model(spectrogram)
    predictions = zip(commands, prediction[0])
    letter = max(predictions, key=lambda x: x[1])[0]
    letter = ' ' if letter == '_' else letter
    print(letter, end='', flush=True)
print("\n")
    #plt.bar(commands, tf.nn.softmax(prediction[0]))
    #plt.title(f'Predictions for "{commands[label[0]]}"')
    #plt.show()

    #for k, v in predictions:
    #    print("{}: {}".format(k, v))
    #print("\n")
#    for i, val in zip(commands, tf.nn.softmax(prediction[0])):
        #print(max(
#        print("{}: {}".format(i, val))


