#!/usr/bin/env python
from scipy.io.wavfile import write as write_wav
import numpy as np
import IPython.display as ipd
from itertools import permutations
import random
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys
import os
from multiprocessing import Pool

from lib.config import ModelConfig

if len(sys.argv) < 2:
    print("Usage: python GenerateData.py <model-yaml>")
    sys.exit(1)

config = ModelConfig(sys.argv[1])

MORSE_CODE_DICT = { 'A':'.-', 'B':'-...', 
                    'C':'-.-.', 'D':'-..', 'E':'.', 
                    'F':'..-.', 'G':'--.', 'H':'....', 
                    'I':'..', 'J':'.---', 'K':'-.-', 
                    'L':'.-..', 'M':'--', 'N':'-.', 
                    'O':'---', 'P':'.--.', 'Q':'--.-', 
                    'R':'.-.', 'S':'...', 'T':'-', 
                    'U':'..-', 'V':'...-', 'W':'.--', 
                    'X':'-..-', 'Y':'-.--', 'Z':'--..', 
                    '1':'.----', '2':'..---', '3':'...--', 
                    '4':'....-', '5':'.....', '6':'-....', 
                    '7':'--...', '8':'---..', '9':'----.', 
                    '0':'-----', ',':'--..--', '.':'.-.-.-', 
                    '?':'..--..', '/':'-..-.', '-':'-....-', 
                    '(':'-.--.', ')':'-.--.-', ' ': ' '}


sample_rate = config.value('data.sample_rate')
freq = 600 #Hz

# how many seconds to jitter the samples
# this should allow for a better model by
# giving more than one "sample" for each character
JITTER_RANGE = config.value('data.jitter')

def generate_silence(time_units, wpm):
    return np.zeros(int(time_units * sample_rate / wpm))

def generate_tone(time_units, wpm):
    jitter = random.uniform(time_units*1.0-JITTER_RANGE,time_units*1.0+JITTER_RANGE)
    t = np.linspace(0.0, jitter / wpm, int(sample_rate*jitter/wpm))
    dit = np.sin(2.0 * np.pi * freq * t)
    return ((dit / max(abs(dit))) * 30000)

generate_word_sep = lambda wpm: generate_silence(7, wpm)
generate_letter_sep = lambda wpm: generate_silence(3, wpm)
generate_symbol_sep = lambda wpm: generate_silence(1, wpm)
generate_dah = lambda wpm: generate_tone(3, wpm) 
generate_dit = lambda wpm: generate_tone(1, wpm)

def encode(s, wpm):
    s = s.upper()
    result = generate_silence(random.uniform(5,15), wpm)
    for char in s:
        symbols = "'".join([i for i in MORSE_CODE_DICT[char]])
        for symbol in symbols:
            if symbol == '-':
                result = np.concatenate((result, generate_dah(wpm)))
            elif symbol == '.':
                result = np.concatenate((result, generate_dit(wpm)))
            elif symbol == "'":
                result = np.concatenate((result, generate_symbol_sep(wpm)))
            elif symbol == ' ':
                result = np.concatenate((result, generate_word_sep(wpm)))
        result = np.concatenate((result, generate_letter_sep(wpm)))
    result = np.concatenate((result, generate_silence(random.uniform(5,15), wpm)))
    return result

def SNR(cw, dB):
    SNR_linear = 10.0**(dB/10.0)
    power = cw.var()
    noise_power = power/SNR_linear
    noise = np.sqrt(noise_power)*np.random.normal(0,1,len(cw))
    return noise + cw

def make_data(si_tup):
    word, i, wpm = si_tup
    snr = random.randint(config.value('data.snr_range.low'), config.value('data.snr_range.high'))
    data = SNR(encode(word, wpm), snr)
    write_wav("{}/{}/{}.wav".format(config.value('system.volumes.data'), word, i), sample_rate, data.astype(np.int16))

def read_in_chunks(file_object):
    i = 0
    while True:
        data = file_object.read(random.randint(12, 16))
        i = i + 1
        if not data:
            break
        yield data

if __name__ == "__main__":
    # generate the data
    with Pool(config.value('system.jobs')) as p:
        chunk = []
        with open(config.value('data.corpus')) as fp:
            try:
                os.mkdir("{}".format(config.value('system.volumes.data')))
                os.mkdir("{}".format(config.value('system.volumes.test')))
            except:
                pass
            for word in list(set(fp.read().split()))[0:config.value('data.max_phrases')]:
                try:
                    os.mkdir("{}/{}".format(config.value('system.volumes.data'), word))
                except:
                    pass

                for i in range(0, config.value('data.samples_per_phrase')):
                    wpm = random.randint(config.value('data.wpm_range.low'), config.value('data.wpm_range.high'))
                    chunk.append((word, i, wpm))

        p.map(make_data, chunk)

