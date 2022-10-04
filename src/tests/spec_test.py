import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))


import scipy.io.wavfile
from utils.eval_utils import gen_test_wave
import matplotlib.pyplot as plt
from examples.DiodeClipper import DiodeClipper

fs = 44100
sweep = gen_test_wave(fs, None, 1, 10, "log_sweep")

DC = DiodeClipper(fs)
out_sweep = DC(sweep)
scipy.io.wavfile.write("log_sweep.wav", fs, sweep)
scipy.io.wavfile.write("out_log_sweep.wav", fs, out_sweep)
