from scipy.io import wavfile
import numpy as np

data = wavfile.read('test.wav')

between_samples = 0
state = prev_state = "OUT_OF_LETTER" 
low_count = 0
d = []
i = 0
space_count = 0
for sample in data[1]:
    if state == "OUT_OF_LETTER":
        if space_count > 4000:
            print('space')
            output_data = np.zeros(5000)
            wavfile.write("output-{:04d}.wav".format(i), 8000, output_data.astype(np.int16))
            i += 1
            space_count = 0
        if abs(sample) > 1500:
            prev_state = state
            state = "IN_LETTER"
            print(state)
            low_count = 0
            d = []
        else:
            space_count += 1
    if state == "IN_LETTER":
        d.append(sample)
        if abs(sample) < 500:
            low_count += 1
        else:
            low_count = 0

        if low_count >= 1200:
            prev_state = state
            state = "OUT_OF_LETTER"
            print(state)
            output_data = np.array(d, dtype=np.int16)
            wavfile.write("output-{:04d}.wav".format(i), 8000, output_data)
            space_count = 0
            i += 1

#output_data = np.array(d, dtype=np.int16)
#avfile.write("output-{:03d}.wav".format(i), 8000, output_data)
