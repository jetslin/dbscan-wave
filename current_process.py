import numpy as np
import dbscan
from scipy.fftpack import fft
from scipy.ndimage import gaussian_filter
from scipy import signal
import random
import importlib

FS = 128000 # sampling frequency
TIME = 10 # duration

# === dbscanner ===
class DBScanner:
    def __init__(self, eps, min_points):
        self._max_holding_samples = 20

        self._eps = eps
        self._min_points = min_points
        self._samples = []
    
    def add_sample(self, sample):
        yf = fft(sample)
        blurred = gaussian_filter(np.abs(yf), sigma=7)
        sample = np.array([blurred]).T

        self._samples.append(sample)

        importlib.reload(dbscan)

        result = dbscan.dbscan(self._build_matrix(self._samples), self._eps, self._min_points)
        if len(self._samples) > self._max_holding_samples:
            # 1. sort samples by classes
            # 2. remove a sample
            zipped = list(zip(result, self._samples))

            tracker = {}
            for i in range(len(zipped)):
                key = str(zipped[i][0])
                if key in tracker:
                    tracker[key] = tracker[key] + 1
                else:
                    tracker[key] = 1
            
            sorted_classes = sorted(tracker.items(), key=lambda x: x[1], reverse=True)
            class_name = sorted_classes[0][0]

            tracker = []
            for i in range(len(zipped)):
                key = str(zipped[i][0])
                if key == class_name:
                    tracker.append(i)
            
            remove_index = random.choice(tracker)
            self._samples.pop(remove_index)
        
        return result, result[-1]

    def _build_matrix(self, samples):
        first = samples[0]
        for i in range(len(samples) - 1):
            next = samples[i + 1]
            con = np.concatenate((first, next), 1)
            first = con

        return first


def gen_wave(shape, fs, f, t, phase):
    samples = np.linspace(0, t, int(fs*t), endpoint=False)
    if shape == 'sine':
        return np.sin(2 * np.pi * f * samples + phase)
    if shape == 'triangle':
        return signal.sawtooth(2 * np.pi * f * samples + phase)
    if shape == 'square':
        return signal.square(2 * np.pi * f * samples + phase)


waves = {'sine': [], 'triangle': [], 'square': []}
yfs = []

for shape in waves:
    for i in range(5):
        waves[shape].append(gen_wave(shape, FS, 10, TIME, np.pi * i / 5))
        yfs.append(waves[shape][i])

db = DBScanner(100000, 2)

# sample loop
for i in range(15):
    output, one = db.add_sample(yfs[i])
    print(one)
