import numpy as np
import cv2
from scipy.io import wavfile

from matplotlib.mlab import specgram
nfft = 256
overlap = nfft - 56  # overlap value for spectrogram

def get_specgram(signal, rate):
    arr2D, freqs, bins = specgram(
        signal,
        window=np.blackman(nfft),
        Fs=rate,
        NFFT=nfft,
        noverlap=overlap,
        pad_to=32 * nfft,
    )
    return arr2D, freqs, bins

def plot_image(arr2D, bins, freqs):
    fig, ax = plt.subplots(1,1)
    extent = (bins[0], bins[-1], freqs[-1], freqs[0])
    im = ax.imshow(
        arr2D,
        aspect="auto",
        extent=extent,
        interpolation="none",
        cmap="Greys",
        norm=None,
    )
    plt.gca().invert_yaxis()
    plt.show()

def normalize_image(img):
    # normalize
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img
    return img


def create_image(filename):
    imgSize=(256, 32)
    dataAugmentation=False

    imgname = filename+".png"   

    # Load  image in grayscale if exists
    img = cv2.imread(imgname, 0) 

    # TODO: re-enable this IF statement
    #if img is None:
    rate, data = wavfile.read(filename)
    arr2D, freqs, bins = get_specgram(data, rate)

    # Get the image data array shape (Freq bins, Time Steps)
    shape = arr2D.shape

    # Find the CW spectrum peak - look across all time steps
    f = int(np.argmax(arr2D[:]) / shape[1])

    time_steps = (4.0/(len(data)/rate))*shape[1]

    # Create a 32x128 array centered to spectrum peak
    img = cv2.resize(arr2D[f - 16 : f + 16][:], imgSize)

    img = normalize_image(img)

    cv2.imwrite(imgname, img*256.)

    img = normalize_image(img)  
    # transpose for TF
    img = cv2.transpose(img)
    return img

