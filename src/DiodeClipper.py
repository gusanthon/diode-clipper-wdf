from utils.eval_utils import gen_test_wave
from wdf import (
    Resistor,
    ResistiveVoltageSource,
    SeriesAdaptor,
    ParallelAdaptor,
    Capacitor,
    DiodePair,
)
import numpy as np


class DiodeClipper:
    def __init__(
        self, sample_rate, cutoff=1000, input_gain_db=0, output_gain_db=0, n_diodes=2
    ) -> None:

        self.def_cutoff = cutoff
        self.def_in_gain = input_gain_db
        self.def_out_gain = output_gain_db

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

        self.elements = [
            self.R1,
            self.Vs,
            self.S1,
            self.C1,
            self.P1,
            self.Dp,
        ]

    def process_sample(self, sample):
        sample *= self.input_gain
        self.Vs.set_voltage(sample)
        self.Dp.accept_incident_wave(self.P1.propagate_reflected_wave())
        self.P1.accept_incident_wave(self.Dp.propagate_reflected_wave())
        return -(self.C1.wave_to_voltage() * self.output_gain)

    def process_signal(self,signal):
        return np.array([self.process_sample(sample) for sample in signal])

    def __call__(self, *args: any, **kwds: any) -> any:
        if isinstance(args[0], float) or isinstance(args[0], int):
            return self.process_sample(args[0])
        elif hasattr(args[0],'__iter__'):
            return self.process_signal(args[0])

    def set_cutoff(self, new_cutoff):
        self.cutoff = new_cutoff
        self.R = 1.0 / (2 * np.pi * self.C * self.cutoff)
        self.R1.set_resistance(self.R)

    def set_input_gain(self, gain_db):
        self.input_gain = 10 ** (gain_db / 20)

    def set_output_gain(self, gain_db):
        self.output_gain = 10 ** (gain_db / 20)

    def set_num_diodes(self, new_n_diodes):
        self.Dp.set_diode_params(self.Dp.Is, self.Dp.Vt, new_n_diodes)

    def __str__(self):
        return "{0}({1}".format(self.__class__.__name__, self.__dict__)

    def reset(self):
        [element.reset() for element in self.elements]
        self.set_input_gain(self.def_in_gain)
        self.set_output_gain(self.def_out_gain)
        self.set_cutoff(self.def_cutoff)

    def get_freq_response(self,delta_dur=1):
        delta = gen_test_wave(self.fs,None,1,delta_dur,'delta')
        return self.process_signal(delta)