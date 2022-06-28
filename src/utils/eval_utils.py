import scipy
import numpy as np
import pandas as pd
import signal_generator as sg

from scipy.signal.windows import general_cosine
from scipy.fftpack import next_fast_len
from numpy.fft import rfft, irfft
from numpy import argmax, mean, log, concatenate, zeros
import numpy as np


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
    elif kind == 'log_sweep':
        sweep_generator = sg.SignalGenerator(
            "log_sweep",
            t,
            amplitude=amp,
            sampleRate=fs,
        )
        x = sweep_generator()

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


def read_ltspice_file(filename, out_label="V(vout)"):
    x = pd.read_csv(filename, delim_whitespace=True)
    return x[out_label]

## spice command helpers
cutoffs = [70, 150, 250, 500, 1000, 2000, 4000, 8000, 16000]

def get_C_from_cutoff(f):
    Z = 800 # for passive lpf
    return (1 / Z) / (2.0 * np.pi * f)
# [print(f"cutoff - {cutoff} hz : {get_R_from_cutoff(cutoff)} [Ohm]\n") for cutoff in cutoffs]

def get_R_from_cutoff(f):
    C = 47e-9 # for diode clipper
    return 1 / (2 * np.pi * C * f)
# [print(f"cutoff - {cutoff} hz : {get_C_from_cutoff(cutoff)} [F]\n") for cutoff in cutoffs]



def rms_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return np.sqrt(mean(np.absolute(a)**2))

flattops = {
    'dantona3': [0.2811, 0.5209, 0.1980],
    'dantona5': [0.21557895, 0.41663158, 0.277263158, 0.083578947,
                 0.006947368],
    'SFT3F': [0.26526, 0.5, 0.23474],
    'SFT4F': [0.21706, 0.42103, 0.28294, 0.07897],
    'SFT5F': [0.1881, 0.36923, 0.28702, 0.13077, 0.02488],
    'SFT3M': [0.28235, 0.52105, 0.19659],
    'SFT4M': [0.241906, 0.460841, 0.255381, 0.041872],
    'SFT5M': [0.209671, 0.407331, 0.281225, 0.092669, 0.0091036],
    'FTSRS': [1.0, 1.93, 1.29, 0.388, 0.028],
    'FTNI': [0.2810639, 0.5208972, 0.1980399],
    'FTHP': [1.0, 1.912510941, 1.079173272, 0.1832630879],
    'HFT70': [1, 1.90796, 1.07349, 0.18199],
    'HFT95': [1, 1.9383379, 1.3045202, 0.4028270, 0.0350665],
    'HFT90D': [1, 1.942604, 1.340318, 0.440811, 0.043097],
    'HFT116D': [1, 1.9575375, 1.4780705, 0.6367431, 0.1228389, 0.0066288],
    'HFT144D': [1, 1.96760033, 1.57983607, 0.81123644, 0.22583558, 0.02773848,
                0.00090360],
    'HFT169D': [1, 1.97441842, 1.65409888, 0.95788186, 0.33673420, 0.06364621,
                0.00521942, 0.00010599],
    'HFT196D': [1, 1.979280420, 1.710288951, 1.081629853, 0.448734314,
                0.112376628, 0.015122992, 0.000871252, 0.000011896],
    'HFT223D': [1, 1.98298997309, 1.75556083063, 1.19037717712, 0.56155440797,
                0.17296769663, 0.03233247087, 0.00324954578, 0.00013801040,
                0.00000132725],
    'HFT248D': [1, 1.985844164102, 1.791176438506, 1.282075284005,
                0.667777530266, 0.240160796576, 0.056656381764, 0.008134974479,
                0.000624544650, 0.000019808998, 0.000000132974],
    }

def parabolic(f, x):
    """
    Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    """
    if int(x) != x:
        raise ValueError('x must be an integer sample index')
    else:
        x = int(x)
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def THDN(signal, fs, weight=None):
    """Measure the THD+N for a signal and print the results
    Prints the estimated fundamental frequency and the measured THD+N.  This is
    calculated from the ratio of the entire signal before and after
    notch-filtering.
    This notch-filters by nulling out the frequency coefficients ±10% of the
    fundamental
    TODO: Make R vs F reference a parameter (currently is R)
    TODO: Or report all of the above in a dictionary?
    """
    # Get rid of DC and window the signal
    signal = np.asarray(signal) + 0.0  # Float-like array
    # TODO: Do this in the frequency domain, and take any skirts with it?
    signal -= mean(signal)

    window = general_cosine(len(signal), flattops['HFT248D'])
    windowed = signal * window
    del signal

    # Zero pad to nearest power of two
    new_len = next_fast_len(len(windowed))
    windowed = concatenate((windowed, zeros(new_len - len(windowed))))

    # Measure the total signal before filtering but after windowing
    total_rms = rms_flat(windowed)

    # Find the peak of the frequency spectrum (fundamental frequency)
    f = rfft(windowed)
    i = argmax(abs(f))
    true_i = parabolic(np.log(abs(f)), i)[0]
    frequency = fs * (true_i / len(windowed))
    print("Est. freq: "+str(frequency)[:6]+" hz")
    # Filter out fundamental by throwing away values ±10%
    lowermin = int(true_i * 0.9)
    uppermin = int(true_i * 1.1)
    f[lowermin: uppermin] = 0
    # TODO: Zeroing FFT bins is bad

    # Transform noise back into the time domain and measure it
    noise = irfft(f)
    # TODO: RMS and A-weighting in frequency domain?  Parseval?

    # TODO: Return a dict or list of frequency, THD+N?
    return rms_flat(noise) / total_rms