import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import matplotlib.pyplot as plt
import numpy as np
from Circuit import DiodeClipper
from utils.eval_utils import SNRsystem, gen_test_wave
from utils.plot_utils import get_freqz_error_vs_spice
p = '/Users/gusanthon/Documents/UPF/Thesis/diode-clipper-transient-analysis-44100.txt'
with open(p) as f:
    trans = np.loadtxt(f,skiprows=1,usecols=1)
fs = 174000

x = gen_test_wave(fs,1000,1,0.005,'sin')

Clipper = DiodeClipper(fs)
y = Clipper(x)
fr = Clipper.get_impulse_response()
plt.plot(trans,label='spice')
plt.plot(y,label='wdf')
plt.legend()
plt.show()

print(get_freqz_error_vs_spice(fr,fs,p))