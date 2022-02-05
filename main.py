import os
import numpy as np
import utils
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.io import wavfile


ordered_files, notes = utils.load_and_order_notes('./notes')

utils.NOTES_PER_SECOND = 4
utils.NUM_SECONDS = 15

distances = [distance.euclidean, distance.cosine, distance.cityblock, distance.chebyshev, distance.braycurtis]

data_path = './1DData/curry-leaves-1296x728-header.npy'
data_org = np.load(data_path)
data = utils.extract_data(data_org, "pair_dist", distance_func=distances[0])
data_q = utils.quantize_data(data, apply_autotune=True, sorted_notes=ordered_files)

plt.title("Input Data")
plt.plot(data)
plt.show()

plt.title("Quantized Data")
plt.plot(data_q)
plt.show()

melody = np.zeros(((utils.NUM_SECONDS + 1) * 44100), np.float32)
for i, d in enumerate(data_q):
    note = notes[d]
    melody[int(44100 / utils.NOTES_PER_SECOND * i): int(44100 / utils.NOTES_PER_SECOND * i + 44100)] += note

melody = (melody.copy() - np.min(melody.copy())) / (np.max(melody.copy()) - np.min(melody.copy())) / 2.

wavfile.write(os.path.join("./generated", f"{data_path.split(os.sep)[-1].split('.')[0]}.wav"), rate=44100, data=melody)
