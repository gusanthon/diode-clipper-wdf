import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from wdf import *
from examples.Circuit import Circuit
from Rtype import RootRTypeAdaptor


class TR_808_HatResonator(Circuit):

    def __init__(self, fs, cutoff, resonance) -> None:

        self.fs = fs
        self.cutoff = cutoff
        self.resonance = resonance

        self.Vin = ResistiveVoltageSource(22e3)
        self.C4 = Capacitor(1e-9, self.fs)
        self.S1 = SeriesAdaptor(self.Vin, self.C4)

        self.R197 = Resistor(820e3)
        self.C58 = Capacitor(.027e-6, self.fs)
        self.C59 = Capacitor(.027e-6, self.fs)
        self.R196 = Resistor(680)
        self.R_adaptor = RootRTypeAdaptor([self.S1, self.R197, self.C58, self.C59, self.R196], self.impedance_calc)

        elements = [
            self.Vin,
            self.C4,
            self.S1,
            self.R197,
            self.C58,
            self.C59,
            self.R196,
            self.R_adaptor
        ]

        self.set_components()

        super().__init__(elements, self.Vin, self.R_adaptor, self.R196)

    def process_sample(self, sample):
        self.Vin.set_voltage(-sample)
        self.R_adaptor.compute()
        return self.output.wave_to_voltage() + self.C59.wave_to_voltage()

    def set_components(self):
        Rfb = 82e3
        R_g = 10000  ** ((1 - self.resonance) ** 0.37)
        C = 1 / (2 * np.pi * self.cutoff * np.sqrt(Rfb * R_g))
        self.R197.set_resistance(Rfb)
        self.R196.set_resistance(R_g)
        self.C58.set_capacitance(C)
        self.C59.set_capacitance(C)

    def set_cutoff(self, new_cutoff):
        if not self.cutoff == new_cutoff:
            self.cutoff = new_cutoff
            self.set_components()

    def set_resonance(self, new_res):
        if not self.resonance == new_res:
            self.resonance == new_res
            self.set_components()

    def impedance_calc(self, R):
        Ag = 100
        Ri = 1e9
        Ro = 1e-1 
        Ra, Rb, Rc, Rd, Re = R.get_port_impedances()
        R.set_S_matrix ([ [ -((Ra * Rb + (Ra - Rb) * Rc) * Rd + (Ra * Rb + (Ra - Rb) * Rc + (Ra - Rb) * Rd) * Re - (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra - Rb) * Rc + (Ra - Rc) * Rd - (Rb + Rc + Rd) * Re - (Rb + Rc + Rd) * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * (Ra * Rc * Rd - Ra * Rc * Ro + (Ra * Rc + Ra * Rd) * Re) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), 2 * (Ra * Rb * Rd + Ra * Rb * Re - (Ra * Rb + Ra * Rd) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), 2 * (Ra * Rb * Re + Ra * Rc * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), 2 * (Ra * Rb * Rd - (Ra * Rb + Ra * Rc + Ra * Rd) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro) ],
                            [ 2 * (Ag * Rb * Rd * Ri - Rb * Rc * Rd + Rb * Rc * Ro - (Rb * Rc + Rb * Rd) * Re) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -((Ra * Rb - (Ra - Rb) * Rc) * Rd + (Ra * Rb - (Ra - Rb) * Rc - (Ra - Rb) * Rd) * Re - (((Ag + 1) * Rc - Rb) * Rd - ((Ag + 1) * Rb - (Ag + 1) * Rc - (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb - (Ra - Rb) * Rc - (Ra + Rc) * Rd + (Rb - Rc - Rd) * Re + (Rb - Rc - Rd) * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * (Ra * Rb * Rd + Ra * Rb * Re + ((Ag + 1) * Rb * Rd + (Ag + 1) * Rb * Re) * Ri - (Ra * Rb + Rb * Re + Rb * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * ((Ag + 1) * Rb * Re * Ri + Ra * Rb * Re - (Ra * Rb + Rb * Rc + Rb * Re + Rb * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * ((Ag + 1) * Rb * Rd * Ri + Ra * Rb * Rd + Rb * Rc * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro) ],
                            [ 2 * (Ag * Rc * Rd * Ri + Rb * Rc * Rd + Rb * Rc * Re - (Rb * Rc + Rc * Rd) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * (Ra * Rc * Rd + Ra * Rc * Re + ((Ag + 1) * Rc * Re + Rc * Rd) * Ri - (Ra * Rc + Rc * Re + Rc * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), ((Ra * Rb - (Ra + Rb) * Rc) * Rd + (Ra * Rb - (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re - (((Ag + 1) * Rc - Rb) * Rd - ((Ag + 1) * Rb - (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb - (Ra + Rb) * Rc + (Ra - Rc) * Rd + (Rb - Rc + Rd) * Re + (Rb - Rc + Rd) * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * ((Ag + 1) * Rc * Re * Ri + (Ra + Rb) * Rc * Re - (Ra * Rc + Rc * Re + Rc * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * ((Ag + 1) * Rc * Rd * Ri + (Ra + Rb) * Rc * Rd - (Rb * Rc + Rc * Rd) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro) ],
                            [ 2 * (Rb * Rd * Re - (Ag * Rb + Ag * Rc) * Rd * Ri + Rc * Rd * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * (Ra * Rd * Re + (Ag * Rc * Rd + (Ag + 1) * Rd * Re) * Ri - ((Ra + Rc) * Rd + Rd * Re + Rd * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * ((Ra + Rb) * Rd * Re - (Ag * Rb * Rd - (Ag + 1) * Rd * Re) * Ri - (Ra * Rd + Rd * Re + Rd * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -((Ra * Rb + (Ra + Rb) * Rc) * Rd - (Ra * Rb + (Ra + Rb) * Rc - (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd - ((Ag + 1) * Rb + (Ag + 1) * Rc - (Ag + 1) * Rd) * Re) * Ri + (Ra * Rb + (Ra + Rb) * Rc - (Ra + Rc) * Rd + (Rb + Rc - Rd) * Re + (Rb + Rc - Rd) * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), 2 * (((Ag + 1) * Rb + (Ag + 1) * Rc) * Rd * Ri - Rc * Rd * Ro + (Ra * Rb + (Ra + Rb) * Rc) * Rd) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro) ],
                            [ 2 * (Rb * Rd * Re + (Ag * Rb + Ag * Rc + Ag * Rd) * Re * Ri - (Rb + Rc + Rd) * Re * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * (Ra * Rd * Re - (Ag * Rc - Rd) * Re * Ri + Rc * Re * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), -2 * ((Ra + Rb) * Rd * Re + (Ag * Rb + (Ag + 1) * Rd) * Re * Ri - (Rb + Rd) * Re * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), 2 * (((Ag + 1) * Rc + Rb) * Re * Ri - Rc * Re * Ro + (Ra * Rb + (Ra + Rb) * Rc) * Re) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro), ((Ra * Rb + (Ra + Rb) * Rc) * Rd - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd - ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd - (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (((Ag + 1) * Rc + Rb) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + (Ag + 1) * Rd) * Re) * Ri - (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Ri) * Ro) ] ])
 


hr = TR_808_HatResonator(44100, 10000, .01)
hr.set_cutoff(10000)
# hr.record_mono_audio(10)
hr.plot_freqz()
