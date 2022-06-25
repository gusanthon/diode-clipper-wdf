
import PassiveLPF
import DiodeClipper
import numpy as np
from utils.path_utils import data_path
from utils.plot_utils import compare_vs_spice,compare_plot,plot_fft
from utils.eval_utils import gen_test_wave

##############
# CIRCUIT CLASS TESTS
#########

fs = 96000
sin = gen_test_wave(fs, 1000, 1, 0.5, "sin")
delta = gen_test_wave(fs, 1000, 1, 0.1, "delta")


# #### PASSIVE LPF
lpf = PassiveLPF.PassiveLPF(fs, cutoff=1000)
out = np.zeros(len(delta))

for i in range(len(delta)):
    out[i] = lpf(delta[i])

spice_path = data_path / "spice" / "passive_LPF_1000hz.txt"
compare_vs_spice(out, fs, spice_path)


#### DIODE CLIPPER
Clipper = DiodeClipper.DiodeClipper(
    fs, cutoff=1000, input_gain_db=0, output_gain_db=0, n_diodes=2
)
#
# get frequency response
out = np.zeros(len(delta))

for i in range(len(delta)):
    out[i] = Clipper(delta[i])

spice_path = data_path / "spice" / 'diode-clipper-frequency-analysis-1000hz.txt'
compare_vs_spice(out, fs, spice_path, title="diode clipper")

#
# sine wave analysis
out = np.zeros(len(sin))

for i in range(len(sin)):
    out[i] = Clipper(sin[i])

compare_plot(sin, out, 200)

plot_fft(out, fs, title="diode clipper output spectrum")
