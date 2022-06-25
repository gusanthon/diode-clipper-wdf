from wdf import Resistor,Capacitor,SeriesAdaptor,ParallelAdaptor,IdealVoltageSource
import numpy as np

class PassiveLPF:
    def __init__(self, sample_rate, cutoff=1000) -> None:
        self.fs = sample_rate
        self.cutoff = cutoff

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

    def process(self, sample):
        self.Vs.set_voltage(sample)
        self.Vs.accept_incident_wave(self.S2.propagate_reflected_wave())
        self.S2.accept_incident_wave(self.Vs.propagate_reflected_wave())
        return self.C2.wave_to_voltage()

    def __call__(self, *args: any, **kwds: any) -> any:
        return self.process(args[0])

    def set_cutoff(self, new_cutoff):
        self.cutoff = new_cutoff
        self.C = (1.0 / self.Z) / (2 * np.pi * self.cutoff)
        self.C1.set_capacitance(self.C)
        self.C2.set_capacitance(self.C)

##TODO
## eval tools class