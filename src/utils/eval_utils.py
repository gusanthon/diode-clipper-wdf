import scipy
import numpy as np
import pandas as pd
import sys

def gen_test_wave(fs, f, amp, t, kind):
    N = int(t * fs)
    n = np.arange(0, N / fs, 1 / fs)
    if kind == "sin":
        x = np.sin(2 * np.pi * f * n) * amp
    elif kind == "cos":
        x = np.cos(2 * np.pi * f * n) * amp
    elif kind == "delta":
        x = np.zeros(N)
        x[0] = amp
    return x


def read_to_linear(path, bd, lin=0):
    fs, x = scipy.io.wavfile.read(path)
    if lin:
        return fs, x
    if bd == 16:
        x = x.astype(np.float32, order="C") / 32768.0
    elif bd == 32:
        x = x.astype(np.float32, order="C") / 2147354889.0
    return fs, x



def read_ltspice_wave(filename, out_label="V(vout)"):
    x = pd.read_csv(filename, delim_whitespace=True)
    return x[out_label]

def THD(x):
    pass


