
from wdf import Resistor,ResistiveVoltageSource,SeriesAdaptor,ParallelAdaptor,Capacitor,DiodePair
import numpy as np

class DiodeClipper:
    def __init__(
        self, sample_rate, cutoff=1000, input_gain_db=0, output_gain_db=0, n_diodes=2
    ) -> None:
        self.fs = sample_rate
        self.cutoff = cutoff
        self.input_gain = 10 ** (input_gain_db / 20)
        self.output_gain = 10 ** (output_gain_db / 20)

        self.C = 47e-9
        self.R = 1.0 / (2 * np.pi * self.C * self.cutoff)

        self.R1 = Resistor(self.R)
        self.Vs = ResistiveVoltageSource()

        self.S1 = SeriesAdaptor(self.Vs, self.R1)
        self.C1 = Capacitor(self.C, self.fs)

        self.P1 = ParallelAdaptor(self.S1, self.C1)
        self.Dp = DiodePair(self.P1, 2.52e-9, n_diodes=n_diodes)

    def process(self, sample):
        sample *= self.input_gain
        self.Vs.set_voltage(sample)
        self.Dp.accept_incident_wave(self.P1.propagate_reflected_wave())
        self.P1.accept_incident_wave(self.Dp.propagate_reflected_wave())
        return -(self.C1.wave_to_voltage() * self.output_gain)

    def __call__(self, *args: any, **kwds: any) -> any:
        return self.process(args[0])

    def set_cutoff(self, new_cutoff):
        self.cutoff = new_cutoff
        self.R = 1.0 / 2 * np.pi * self.cutoff * self.C
        self.R1.set_resistance(self.R)

    def set_input_gain(self, gain_db):
        self.input_gain = 10 ** (gain_db / 20)

    def set_output_gain(self, gain_db):
        self.output_gain = 10 ** (gain_db / 20)

    def set_num_diodes(self, new_n_diodes):
        self.Dp.set_diode_params(self.Dp.Is, self.Dp.Vt, new_n_diodes)

##TODO
## eval tools class