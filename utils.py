import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import glob
import os
from random import choice
import librosa


NUM_SECONDS = 10
NOTES_PER_SECOND = 3
NUM_NOTES = 48
NOTES_ORDER = "CDEFGAB"
MODES = ["none", "pair_dist", "g_pt_dist"]

C_maj = [n + str(i) for i in range(2, 6) for n in ["C", "D", "E", "F", "G", "A", "B"]]
Db_maj = [n + str(i) for i in range(2, 6) for n in ["Db", "Eb", "F", "Gb", "Ab", "Bb", "C"]]
D_maj = [n + str(i) for i in range(2, 6) for n in ["D", "E", "Gb", "G", "A", "B", "Db"]]
Eb_maj = [n + str(i) for i in range(2, 6) for n in ["Eb", "F", "G", "Ab", "Bb", "C", "D"]]
F_maj = [n + str(i) for i in range(2, 6) for n in ["F", "G", "A", "Bb", "C", "D", "E"]]
Gb_maj = [n + str(i) for i in range(2, 6) for n in ["Gb", "Ab", "Bb", "B", "Db", "Eb", "F"]]
G_maj = [n + str(i) for i in range(2, 6) for n in ["G", "A", "B", "C", "D", "E", "Gb"]]
Ab_maj = [n + str(i) for i in range(2, 6) for n in ["Ab", "Bb", "C", "Db", "Eb", "F", "G"]]
A_maj = [n + str(i) for i in range(2, 6) for n in ["A", "B", "Db", "D", "E", "Gb", "Ab"]]
Bb_maj = [n + str(i) for i in range(2, 6) for n in ["Bb", "C", "D", "Eb", "F", "G", "A"]]
B_maj = [n + str(i) for i in range(2, 6) for n in ["B", "Db", "Eb", "E", "Gb", "Ab", "Bb"]]

C_min = [n + str(i) for i in range(2, 6) for n in ["C", "D", "Eb", "F", "G", "Ab", "Bb"]]
Db_min = [n + str(i) for i in range(2, 6) for n in ["Db", "Eb", "E", "Gb", "Ab", "A", "B"]]
D_min = [n + str(i) for i in range(2, 6) for n in ["D", "E", "F", "G", "A", "Bb", "C"]]
Eb_min = [n + str(i) for i in range(2, 6) for n in ["Eb", "F", "Gb", "Ab", "Bb", "B", "Db"]]
F_min = [n + str(i) for i in range(2, 6) for n in ["F", "G", "Ab", "Bb", "C", "Db", "Eb"]]
Gb_min = [n + str(i) for i in range(2, 6) for n in ["Gb", "Ab", "A", "B", "Db", "D", "F"]]
G_min = [n + str(i) for i in range(2, 6) for n in ["G", "A", "Bb", "C", "D", "Eb", "F"]]
Ab_min = [n + str(i) for i in range(2, 6) for n in ["Ab", "Bb", "B", "Db", "Eb", "F", "Gb"]]
A_min = [n + str(i) for i in range(2, 6) for n in ["A", "B", "C", "D", "E", "F", "G"]]
Bb_min = [n + str(i) for i in range(2, 6) for n in ["Bb", "C", "Db", "Eb", "F", "Gb", "Ab"]]
B_min = [n + str(i) for i in range(2, 6) for n in ["B", "Db", "D", "E", "Gb", "G", "A"]]

SCALES = [C_maj, Db_maj, D_maj, Eb_maj, F_maj, Gb_maj, G_maj, Ab_maj, A_maj, Bb_maj, B_maj,
          C_min, Db_min, D_min, Eb_min, F_min, Gb_min, G_min, Ab_min, A_min, Bb_min, B_min]


def load_and_order_notes(notes_folder):
    global NUM_NOTES
    notes = glob.glob(os.path.join(notes_folder, '*.aiff'))
    NUM_NOTES = len(notes)

    def order_notes(note):
        file_name = note.split(os.sep)[-1]
        note_name = file_name.split('.')[-2]
        octave = int(note_name[-1])
        note_idx = NOTES_ORDER.index(note_name[0])
        return octave * 12 + note_idx - 0.5 * int("b" in note_name)

    sorted_notes = list(sorted(notes, key=order_notes))
    notes_numpy = []
    print("Loading notes...")
    for note in sorted_notes:
        a, _ = librosa.load(note, sr=44100)
        notes_numpy.append(a[10000:54100])
    return sorted_notes, np.array(notes_numpy)


def quantize_data(signal, apply_autotune=True, sorted_notes=None):
    if apply_autotune:
        if sorted_notes is None:
            raise AttributeError("Sorted notes list necessary for autotune.")
    mod_signal = (signal.copy() - np.min(signal.copy())) / (np.max(signal.copy()) - np.min(signal.copy()))
    len_s = np.arange(mod_signal.shape[0])
    new_x = np.linspace(np.min(len_s), np.max(len_s), NOTES_PER_SECOND * NUM_SECONDS)
    mod_signal = interp1d(len_s, mod_signal, kind='quadratic')(new_x)
    s = int(NOTES_PER_SECOND * 1.5)
    mod_signal = savgol_filter(mod_signal, s if s % 2 == 1 else (s + 1), 2)
    quant = np.int32(mod_signal * (NUM_NOTES - 1))
    print("Notes quantized!")
    if not apply_autotune:
        return quant
    print("Applying autotune...")
    idx_notes, scale = autotune(quant, sorted_notes)
    quant = idx_notes[np.int32(mod_signal * (len(scale) - 1))]
    return quant


def autotune(q_data, sorted_notes):
    sorted_notes = list(map(lambda l: l.split(os.sep)[-1].split('.')[-2], sorted_notes))
    coincidences = []
    for num_coincidences in range(2, NUM_NOTES):
        try:
            notes = [sorted_notes[q_data[i]] for i in range(num_coincidences)]
        except Exception:
            continue
        for scale in SCALES:
            if all([(n in scale) and (n in sorted_notes) for n in notes]):
                coincidences.append([scale, num_coincidences])
    if len(coincidences) == 0:
        print("Impossible to autotune... :C")
    max_n = max(coincidences, key=lambda l: l[1])[1]
    best_scale = choice(list(filter(lambda l: l[1] == max_n, coincidences)))[0]
    scale_notes_idx = np.array([sorted_notes.index(s) for s in best_scale])
    return scale_notes_idx, best_scale


def extract_data(data, mode, distance_func, g_pt=None):
    assert mode in MODES
    new_data = []
    if mode == "pair_dist":
        for d in data:
            new_data.append(distance_func(d[0], d[1]))
    elif mode == "g_pt_dist":
        assert g_pt is not None
        for i in range(data.shape[0] + 1):
            pt1 = data[i][0] if i < data.shape[0] else data[i-1][1]
            new_data.append(distance_func(g_pt, pt1))
    else:
        new_data = data
    return new_data
