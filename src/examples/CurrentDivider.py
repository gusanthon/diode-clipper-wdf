import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from wdf import *
from examples import Circuit

class CurrentDivider(Circuit):
    def __init__(self, fs, R1_val, R2_val) -> None:
        self.fs = fs

        self.R1 = Resistor(R1_val)
        self.R2 = Resistor(R2_val)

        self.P1 = ParallelAdaptor(self.R1, self.R2)
        self.I1 = PolarityInverter(self.P1)
        self.Is = IdealCurrentSource(self.I1)

        elements = [
            self.R1,
            self.R2,
            self.P1,
            self.I1,
            self.Is,
        ]

        super().__init__(elements, self.Is, self.Is, self.R1)

    def process_sample(self, sample):
        self.source.set_current(sample)
        self.root.accept_incident_wave(self.root.next.propagate_reflected_wave())
        self.root.next.accept_incident_wave(self.root.propagate_reflected_wave())
        return self.output.wave_to_current()
