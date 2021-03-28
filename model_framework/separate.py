from scipy.io import wavfile
import numpy as np

data = wavfile.read('test.wav')

between_samples = 0
state = prev_state = "OUT_OF_WORD" 
low_count = 0
d = []
i = 0
for sample in data[1]:
    if state == "OUT_OF_WORD":
        if abs(sample) > 1500:
            prev_state = state
            state = "IN_WORD"
            low_count = 0
            d = []
    if state == "IN_WORD":
        d.append(sample)
        if abs(sample) < 300:
            low_count += 1
        if low_count > 2000:
            prev_state = state
            state = "OUT_OF_WORD"
            output_data = np.array(d, dtype=np.int16)
            wavfile.write("output-{:03d}.wav".format(i), 8000, output_data)
            i += 1

output_data = np.array(d, dtype=np.int16)
wavfile.write("output-{:03d}.wav".format(i), 8000, output_data)

