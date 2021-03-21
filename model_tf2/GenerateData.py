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


sample_rate = 8000
freq = 600 #Hz

# how many seconds to jitter the samples
# this should allow for a better model by
# giving more than one "sample" for each character
JITTER_RANGE = 0.8


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
    return result


def SNR(cw, dB):
    SNR_linear = 10.0**(dB/10.0)
    power = cw.var()
    noise_power = power/SNR_linear
    noise = np.sqrt(noise_power)*np.random.normal(0,1,len(cw))
    return noise + cw


def corpus(ngram):
    corpus = []
    for i in permutations(MORSE_CODE_DICT.keys(), ngram):
        corpus.append("".join(i))
    return corpus

def make_data(si_tup):
    word, i, wpm = si_tup
    s = word + str(i)
    f = bytes(s.encode('utf-8')).hex()
    filename = "./data/{}/{}.wav".format(word, i)
    my_cq = SNR(encode(word, wpm), random.randint(6,40))
    #create_image(filename)
    print("Writing WAV {}".format(f))
    write_wav("data/{}/{}.wav".format(word, i), sample_rate, my_cq.astype(np.int16))

if __name__ == "__main__":
    my_cq = SNR(encode("rgr", 25), 40)
    write_wav("test/test.wav", sample_rate, my_cq.astype(np.int16))
    with Pool(96) as p:
        chunk = []
        for word in ["hello", "yes", "no", "maybe", "CQ", "QRZ", "QRM", "QSY", "QRT", "DE", "73", "poop", "KJ7PKQ", "rgr", "hw cpy","rr","test"]:
            try:
                os.mkdir("./data/{}".format(word))
            except:
                pass

            for i in range(0, 150):
                wpm = random.randint(20, 30)
                chunk.append((word, i, wpm))

        p.map(make_data, chunk)

