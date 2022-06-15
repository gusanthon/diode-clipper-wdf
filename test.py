from wdf import *
from utils import *

##############
# CIRCUIT CLASS TESTS
#########

sin = gen_test_wave(44100,1000,1,.5,'sin')
delta = gen_test_wave(44100,1000,1,.1,'delta')

#### PASSIVE LPF
lpf = Passive_LPF(44100)
out = np.zeros(len(delta))

for i in range(len(delta)):
  out[i] = lpf(delta[i])

path = '/Users/gusanthon/Documents/UPF/Thesis/diode-clipper-wdf/spice/passive_LPF_1000hz.txt'
compare_vs_spice(out,44100,path)


#### DIODE CLIPPER
diode_clipper = Diode_clipper(44100,cutoff=1000,input_gain_db=0,output_gain_db=0,n_diodes=2)

#get frequency response
for i in range(len(delta)):
  out[i] = diode_clipper(delta[i])

path = '/Users/gusanthon/Documents/UPF/Thesis/diode-clipper-wdf/spice/diode-clipper-frequency-analysis-1000hz.txt'
compare_vs_spice(out,44100,path,title='diode clipper')

#sine wave analysis
out = np.zeros(len(sin))
for i in range(len(sin)):
  out[i] = diode_clipper(sin[i])

plt.plot(sin[:200],label='input')
plt.plot(out[:200],label='output')
plt.legend()
plt.show()

plot_fft(out)