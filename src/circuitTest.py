from wdf import *
from utils.eval_utils import * 
from utils.plot_utils import *

class Circuit():
    def __init__(self,root,source,output) -> None:
        self.root = root
        self.source = source
        self.output = output

    def process(self,sample):
        self.source.set_voltage(sample)
        self.root.accept_incident_wave(self.root.next.propagate_reflected_wave())
        self.root.next.accept_incident_wave(self.root.propagate_reflected_wave())
        return self.output.wave_to_voltage()

    def __call__(self, *args: any, **kwds: any) -> any:
        return self.process(args[0])

fs = 44100
cutoff = 1000
C = 47e-9
R = 1./(2 * np.pi * C * cutoff)

R1 = Resistor(R)
Vs = ResistiveVoltageSource()

S1 = SeriesAdaptor(Vs,R1)
C1 = Capacitor(C,fs)

P1 = ParallelAdaptor(S1,C1)
Dp = DiodePair(P1,2.62e-9)

clipper = Circuit(
    root=Dp,
    source=Vs,
    output=C1
)

delta = gen_test_wave(fs,1000,1,.5,'delta')
out = np.zeros(len(delta))
sin = gen_test_wave(fs,1000,1,.5,'sin')
out = np.zeros(len(sin))
for i in range(len(sin)):
    out[i] = clipper(sin[i])

plt.plot(sin[:200],label='input')
plt.plot(out[:200],label='output')
plt.legend()
plt.show()
out = np.zeros(len(delta))
for i in range(len(delta)):
    out[i] = clipper(delta[i])

plot_freqz(out,fs)


# Z = 800
# C = (1 / Z) / (2.0 * np.pi * cutoff)
# R1 = Resistor(10)
# R2 = Resistor(1e4)
# C1 = Capacitor(C,fs)
# C2 = Capacitor(C, fs)
# S1 = SeriesAdaptor(R2, C2)
# P1 = ParallelAdaptor(C1, S1)
# S2 = SeriesAdaptor(R1, P1)
# Vs = IdealVoltageSource(P1)

# lpf = Circuit(
#     root=Vs,
#     source=Vs,
#     output = C2
# )
# out = np.zeros(len(delta))
# for i in range(len(delta)):
#     out = lpf(delta[i])

# plot_freqz(out,fs)