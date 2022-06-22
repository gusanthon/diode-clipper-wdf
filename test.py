from wdf import *
from utils import *

##############
# CIRCUIT CLASS TESTS
#########

fs = 44100
sin = gen_test_wave(fs,1000,1,.5,'sin')
delta = gen_test_wave(fs,1000,1,.1,'delta')


# #### PASSIVE LPF
lpf = Passive_LPF(fs,cutoff=1000)
out = np.zeros(len(delta))

for i in range(len(delta)):
  out[i] = lpf(delta[i])

spice_path = '/Users/gusanthon/Documents/UPF/Thesis/diode-clipper-wdf/spice/passive_LPF_1000hz.txt'
compare_vs_spice(out,fs,spice_path)


#### DIODE CLIPPER
diode_clipper = Diode_clipper(fs,cutoff=1000,input_gain_db=30,output_gain_db=2,n_diodes=2)
#
# get frequency response
out = np.zeros(len(delta))

for i in range(len(delta)):
  out[i] = diode_clipper(delta[i])

spice_path = '/Users/gusanthon/Documents/UPF/Thesis/diode-clipper-wdf/spice/diode-clipper-frequency-analysis-1000hz.txt'
compare_vs_spice(out,fs,spice_path,title='diode clipper')

#
# sine wave analysis
out = np.zeros(len(sin))

for i in range(len(sin)):
  out[i] = diode_clipper(sin[i])

compare_plot(sin,out,200)

plot_fft(out,fs,title='diode clipper output spectrum')
