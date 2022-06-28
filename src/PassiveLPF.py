from wdf import Resistor, Capacitor, SeriesAdaptor, ParallelAdaptor, IdealVoltageSource
import numpy as np


class PassiveLPF:
    def __init__(self, sample_rate, cutoff=1000) -> None:
        self.fs = sample_rate
        self.cutoff = cutoff
        self.def_cutoff = cutoff

        self.Z = 800
        self.C = (1 / self.Z) / (2.0 * np.pi * cutoff)

        self.R1 = Resistor(10)
        self.R2 = Resistor(1e4)

        self.C1 = Capacitor(self.C, self.fs)
        self.C2 = Capacitor(self.C, self.fs)

        self.S1 = SeriesAdaptor(self.R2, self.C2)
        self.P1 = ParallelAdaptor(self.C1, self.S1)
        self.S2 = SeriesAdaptor(self.R1, self.P1)

        self.Vs = IdealVoltageSource(self.P1)

        self.elements = [
            self.R1,
            self.R2,
            self.C1,
            self.C2,
            self.S1,
            self.P1,
            self.S2,
        ]

    def process_sample(self, sample):
        self.Vs.set_voltage(sample)
        self.Vs.accept_incident_wave(self.S2.propagate_reflected_wave())
        self.S2.accept_incident_wave(self.Vs.propagate_reflected_wave())
        return self.C2.wave_to_voltage()

    def process_signal(self,signal):
        return np.array([self.process_sample(sample) for sample in signal])

    def __call__(self, *args: any, **kwds: any) -> any:
        if isinstance(args[0], float) or isinstance(args[0], int):
            return self.process_sample(args[0])
        elif hasattr(args[0],'__iter__'):
            return self.process_signal(args[0])

    def set_cutoff(self, new_cutoff):
        self.cutoff = new_cutoff
        self.C = (1.0 / self.Z) / (2 * np.pi * self.cutoff)
        self.C1.set_capacitance(self.C)
        self.C2.set_capacitance(self.C)

    def reset(self):
        [element.reset() for element in self.elements()]
        self.set_cutoff(self.def_cutoff)

    def __str__(self):
        return "{0}({1}".format(self.__class__.__name__, self.__dict__)

