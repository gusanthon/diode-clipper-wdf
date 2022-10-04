import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import matplotlib.pyplot as plt
from utils.eval_utils import gen_test_wave, SNRsystem
from utils.plot_utils import plot_fft, plot_freqz
import scipy.io.wavfile
from wdf import *
import time
from Rtype import RTypeAdaptor, RootRTypeAdaptor
from pyaudio import PyAudio, paFloat32

class Circuit:

    def __init__(self, elements, source, root, output) -> None:
        self.elements = elements
        self.source = source
        self.root = root
        self.output = output

    def process_sample(self, sample):
        self.source.set_voltage(sample)
        self.root.accept_incident_wave(self.root.next.propagate_reflected_wave())
        self.root.next.accept_incident_wave(self.root.propagate_reflected_wave())
        return self.output.wave_to_voltage()

    def process_signal(self, signal):
        self.reset()
        return np.array([self.process_sample(sample) for sample in signal])

    def __call__(self, *args: any, **kwds: any) -> any:
        if isinstance(args[0], float) or isinstance(args[0], int):
            return self.process_sample(args[0])
        elif hasattr(args[0], "__iter__"):
            return self.process_signal(args[0])

    def get_impulse_response(self, delta_dur=1, amp=1):
        return self.process_signal(
            gen_test_wave(self.fs, None, amp, delta_dur, "delta")
        )

    def plot_freqz(self, delta_dur=1, amp=1):
        plot_freqz(self.get_impulse_response(delta_dur=delta_dur, amp=amp), self.fs)

    def set_sample_rate(self, new_fs):
        if not self.fs == new_fs:
            self.fs = new_fs
            for el in self.elements:
                if hasattr(el,'fs'):
                    el.prepare(new_fs)

    def reset(self):
        for el in self.elements:
            el.reset()

    def impedance_calc(self, R):
        pass

    def record_mono_audio(
            self, duration, chunk=1024, file_name_input="", file_name_output="", callback=None
    ):
        """
        record audio in mono for duration seconds, block size = chunk,
        save input file or output file by passing names as strings
        to params, optional callback function to be executed after
        audio is processed, before written to out buffer
        """

        p = PyAudio()
        stream = p.open(
            format=paFloat32,
            channels=1,
            rate=self.fs,
            input=True,
            frames_per_buffer=chunk,
        )

        player = p.open(
            format=paFloat32,
            channels=1,
            rate=self.fs,
            output=True,
            frames_per_buffer=chunk,
        )

        wet = np.zeros(duration * self.fs, dtype=np.float32)
        dry = np.zeros(duration * self.fs, dtype=np.float32)
        times = np.zeros(int(duration * self.fs / chunk))
        idx = 0

        for i in range(int(duration * self.fs / chunk)):
            start = time.time()

            data = np.frombuffer(stream.read(chunk, exception_on_overflow=False), dtype=np.float32)
            processed = np.zeros(len(data), dtype=np.float32)

            for j in range(len(data)):

                processed[j] = self.process_sample(data[j])
                wet[idx] = processed[j]
                dry[idx] = data[j]
                idx += 1

                if callback:
                    callback(self)

            player.write(processed, chunk)
            end = time.time()
            t = round((end - start) * 1000, 5)
            times[i] = t

        stream.stop_stream()
        stream.close()
        p.terminate()
        print(
            f'AVG time taken to read from input buffer, process audio and write to output buffer :\n {np.average(times)} ms\n')
        dry = np.array(dry, dtype=np.float32)
        wet = np.array(wet, dtype=np.float32)

        if file_name_input:
            scipy.io.wavfile.write(file_name_input, self.fs, dry)
        if file_name_output:
            scipy.io.wavfile.write(file_name_output, self.fs, wet)

        return wet, dry


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
        [print(f"{el} : {el.a}\n") for el in elements]
        super().__init__(elements, self.Is, self.Is, self.R1)

    def process_sample(self, sample):
        self.source.set_current(sample)
        self.root.accept_incident_wave(self.root.next.propagate_reflected_wave())
        self.root.next.accept_incident_wave(self.root.propagate_reflected_wave())
        return self.output.wave_to_current()


# cd = CurrentDivider(44100,10000,10000,)
# cd.plot_freqz()
# cd.record_mono_audio(10)

class PassiveLPF(Circuit):

    def __init__(self, fs, cutoff) -> None:
        self.fs = fs
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

        elements = [
            self.R1,
            self.R2,
            self.C1,
            self.C2,
            self.S1,
            self.P1,
            self.S2,
            self.Vs,
        ]

        super().__init__(elements, self.Vs, self.Vs, self.C2)

    def set_cutoff(self, new_cutoff):
        if not self.cutoff == new_cutoff:
            self.cutoff = new_cutoff
            self.C = (1.0 / self.Z) / (2 * np.pi * self.cutoff)
            self.C1.set_capacitance(self.C)
            self.C2.set_capacitance(self.C)

    def process_sample(self, sample):
        self.Vs.set_voltage(sample)
        self.Vs.accept_incident_wave(self.S2.propagate_reflected_wave())
        self.S2.accept_incident_wave(self.Vs.propagate_reflected_wave())
        return self.C2.wave_to_voltage()


class DiodeClipper(Circuit):

    def __init__(
            self, sample_rate, cutoff=1000, input_gain_db=0, output_gain_db=0, n_diodes=2
    ) -> None:

        self.def_cutoff = cutoff
        self.def_in_gain = input_gain_db
        self.def_out_gain = output_gain_db

        self.fs = sample_rate
        self.cutoff = cutoff
        self.input_gain = 10 ** (input_gain_db / 20)
        self.input_gain_db = input_gain_db
        self.output_gain = 10 ** (output_gain_db / 20)
        self.output_gain_db = output_gain_db
        self.n_diodes = n_diodes

        self.C = 47e-9
        self.R = 1.0 / (2 * np.pi * self.C * self.cutoff)

        self.R1 = Resistor(self.R)
        self.Vs = ResistiveVoltageSource()

        self.S1 = SeriesAdaptor(self.Vs, self.R1)
        self.C1 = Capacitor(self.C, self.fs)

        self.P1 = ParallelAdaptor(self.S1, self.C1)
        self.Dp = DiodePair(self.P1, 2.52e-9, n_diodes=n_diodes)

        elements = [
            self.R1,
            self.Vs,
            self.S1,
            self.C1,
            self.P1,
            self.Dp,
        ]

        super().__init__(elements, self.Vs, self.Dp, self.C1)

    def set_cutoff(self, new_cutoff):
        if not self.cutoff == new_cutoff:
            self.cutoff = new_cutoff
            self.R = 1.0 / (2 * np.pi * self.C * self.cutoff)
            self.R1.set_resistance(self.R)

    def set_input_gain(self, gain_db):
        if not self.input_gain_db == gain_db:
            self.input_gain = 10 ** (gain_db / 20)
            self.input_gain_db = gain_db

    def set_output_gain(self, gain_db):
        if not self.output_gain_db == gain_db:
            self.output_gain = 10 ** (gain_db / 20)
            self.output_gain_db = gain_db

    def set_n_diodes(self, new_n_diodes):
        self.Dp.set_n_diodes(new_n_diodes)
        self.n_diodes = new_n_diodes

    def process_sample(self, sample):
        sample *= self.input_gain
        return -super().process_sample(sample) * self.output_gain

class BaxandallEQ(Circuit):
    def __init__(self, fs, bass, treble, adapted : bool = True) -> None:

        self.fs = fs
        self.bass = bass
        self.treble = treble
        self.adapted = adapted
        
        self.Pt = 100.0e3
        self.Pb = 100.0e3

        # Port A
        self.Pt_plus = Resistor(self.Pt * .5)
        self.Resd = Resistor(10e3)
        self.P4 = ParallelAdaptor(self.Pt_plus, self.Resd)
        self.Cd = Capacitor(6.4e-9, fs)
        self.S4 = SeriesAdaptor(self.Cd, self.P4)

        # Port B
        self.Pt_minus = Resistor(self.Pt * .5)
        self.Rese = Resistor(1e3)
        self.P5 = ParallelAdaptor(self.Pt_minus, self.Rese)
        self.Ce = Capacitor(64e-9, fs)
        self.S5 = SeriesAdaptor(self.Ce, self.P5)
        self.Rl = Resistor(1e6)
        self.P1 = ParallelAdaptor(self.Rl, self.S5)

        # Port C
        self.Resc = Resistor(10e3)

        # Port D
        self.Pb_minus = Resistor(self.Pb * .5)
        self.Cc = Capacitor(22e-9, fs)
        self.P3 = ParallelAdaptor(self.Pb_minus, self.Cc)
        self.Resb = Resistor(1e3)
        self.S3 = SeriesAdaptor(self.Resb, self.P3)

        # Port E
        self.Pb_plus = Resistor(self.Pb * .5)
        self.Cb = Capacitor(22e-9, fs)
        self.P2 = ParallelAdaptor(self.Pb_plus, self.Cb)
        self.Resa = Resistor(10e3)
        self.S2 = SeriesAdaptor(self.Resa, self.P2)

        # Port F
        self.Ca = Capacitor(1e6,self.fs)

        if adapted:
            self.R_adaptor = RTypeAdaptor([self.S4, self.P1, self.Resc, self.S3, self.S2], self.impedance_calc, 5)
            self.S1 = SeriesAdaptor(self.R_adaptor, self.Ca)
            self.Vin = IdealVoltageSource(self.S1)

        else:
            self.Vin = ResistiveVoltageSource()
            self.S1 = SeriesAdaptor(self.Vin,self.Ca)
            self.R_adaptor = RootRTypeAdaptor([self.S4,self.P1,self.Resc, self.S3, self.S2, self.S1], self.impedance_calc)


        elements = [
            self.Pt_plus, self.Resd, self.P4, self.Cd, self.S4, self.Pt_minus, self.Rese, self.P5, self.Ce, self.S5,
            self.Rl, self.P1, self.Resc, self.Pb_minus, self.Cc, self.P3, self.Resb, self.S3, self.Pb_plus, self.Cb,
            self.P2, self.Resa, self.S2, self.R_adaptor, self.Ca, self.S1, self.Vin,
        ]

        self.set_bass(bass)
        self.set_treble(treble)

        super().__init__(elements, self.Vin, self.Vin, self.Rl)

    def impedance_calc(self,R):
        if self.adapted:
            Ra, Rb, Rc, Rd, Re = R.get_port_impedances()
            R.set_S_matrix ([ [ -((Ra * Ra * Rb + Ra * Ra * Rc - Rb * Rc * Rc) * Rd * Rd - (Rb * Rb * Rc + Rb * Rc * Rc + Rb * Rd * Rd + (Rb * Rb + 2 * Rb * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + 2 * Ra * Ra * Rb * Rc + (Ra * Ra - Rb * Rb) * Rc * Rc) * Rd + (Ra * Ra * Rb * Rb + 2 * Ra * Ra * Rb * Rc + (Ra * Ra - Rb * Rb) * Rc * Rc + (Ra * Ra - 2 * Rb * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb - Rb * Rc * Rc + (Ra * Ra - Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Ra * Rc + Ra * Rc * Rc) * Rd * Rd + (Ra * Rb * Rc + Ra * Rc * Rc + Ra * Rd * Rd + (Ra * Rb + 2 * Ra * Rc) * Rd) * Re * Re + 2 * (Ra * Ra * Rb * Rc + (Ra * Ra + Ra * Rb) * Rc * Rc) * Rd + (2 * Ra * Ra * Rb * Rc + 2 * (Ra * Ra + Ra * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rc) * Rd * Rd + (Ra * Ra * Rb + 2 * Ra * Rc * Rc + 3 * (Ra * Ra + Ra * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((2 * Ra * Ra * Rb + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + Ra * Rb * Rc + Ra * Rb * Rd) * Re * Re + 2 * (Ra * Ra * Rb * Rb + (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (2 * Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + (3 * Ra * Ra * Rb + 2 * Ra * Rb * Rb + (Ra * Ra + 3 * Ra * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Rb * Rb + Ra * Rb * Rc + Ra * Rb * Rd) * Re * Re - (Ra * Ra * Rb * Rc + (Ra * Ra + Ra * Rb) * Rc * Rc) * Rd + (Ra * Ra * Rb * Rb + Ra * Rb * Rb * Rc - (Ra * Ra + Ra * Rb) * Rc * Rc + (Ra * Ra * Rb - Ra * Ra * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((2 * Ra * Ra * Rb + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (2 * Ra * Ra * Rb * Rb + (Ra * Ra + Ra * Rb) * Rc * Rc + (3 * Ra * Ra * Rb + 2 * Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + Ra * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rd * Rd + (2 * Ra * Ra * Rb + Ra * Rb * Rb) * Rc + (2 * Ra * Ra * Rb + 2 * Ra * Rb * Rb + (2 * Ra * Ra + 3 * Ra * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -(Ra * Rc * Rd + (Ra * Rb + Ra * Rc + Ra * Rd) * Re) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re) ],
                        [ -((Ra * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + (Rb * Rb * Rc + Rb * Rc * Rc + Rb * Rd * Rd + (Rb * Rb + 2 * Rb * Rc) * Rd) * Re * Re + 2 * (Ra * Rb * Rb * Rc + (Ra * Rb + Rb * Rb) * Rc * Rc) * Rd + (2 * Ra * Rb * Rb * Rc + 2 * (Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Rb + 2 * Rb * Rc) * Rd * Rd + (Ra * Rb * Rb + 2 * Rb * Rc * Rc + 3 * (Ra * Rb + Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((Ra * Ra * Rc + Ra * Rc * Rc) * Rd * Rd - (Ra * Rb * Rb + Rb * Rb * Rc - Ra * Rc * Rc - Ra * Rd * Rd + (Rb * Rb - 2 * Ra * Rc) * Rd) * Re * Re - (Ra * Ra * Rb * Rb + 2 * Ra * Rb * Rb * Rc - (Ra * Ra - Rb * Rb) * Rc * Rc) * Rd - (Ra * Ra * Rb * Rb + 2 * Ra * Rb * Rb * Rc - (Ra * Ra - Rb * Rb) * Rc * Rc - (Ra * Ra + 2 * Ra * Rc) * Rd * Rd + 2 * (Ra * Rb * Rb - Ra * Rc * Rc - (Ra * Ra - Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Ra * Rb + Ra * Rb * Rc) * Rd * Rd + (2 * Ra * Rb * Rb + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb) * Rd) * Re * Re + 2 * (Ra * Ra * Rb * Rb + (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (2 * Ra * Ra * Rb * Rb + Ra * Rb * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + (2 * Ra * Ra * Rb + 3 * Ra * Rb * Rb + (3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((2 * Ra * Rb * Rb + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra * Rb + 2 * Ra * Rb * Rb) * Rc) * Rd + (2 * Ra * Ra * Rb * Rb + (Ra * Rb + Rb * Rb) * Rc * Rc + (2 * Ra * Ra * Rb + 3 * Ra * Rb * Rb) * Rc + (2 * Ra * Ra * Rb + 2 * Ra * Rb * Rb + (3 * Ra * Rb + 2 * Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Ra * Rb + Ra * Rb * Rc) * Rd * Rd + (Ra * Ra * Rb * Rb + Ra * Ra * Rb * Rc - (Ra * Rb + Rb * Rb) * Rc * Rc) * Rd - (Ra * Rb * Rb * Rc - Ra * Rb * Rd * Rd + (Ra * Rb + Rb * Rb) * Rc * Rc - (Ra * Rb * Rb - Rb * Rb * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Rb + Rb * Rc) * Rd + (Rb * Rc + Rb * Rd) * Re) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re) ],
                        [ ((2 * Ra * Rb * Rc + (Ra + 2 * Rb) * Rc * Rc) * Rd * Rd + (Rb * Rb * Rc + Rb * Rc * Rc + Rb * Rc * Rd) * Re * Re + 2 * (Ra * Rb * Rb * Rc + (Ra * Rb + Rb * Rb) * Rc * Rc) * Rd + (2 * Ra * Rb * Rb * Rc + (Ra + 2 * Rb) * Rc * Rd * Rd + 2 * (Ra * Rb + Rb * Rb) * Rc * Rc + ((Ra + 3 * Rb) * Rc * Rc + (3 * Ra * Rb + 2 * Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Ra * Rc + Ra * Rc * Rc) * Rd * Rd + (2 * Ra * Rb * Rc + (2 * Ra + Rb) * Rc * Rc + (2 * Ra + Rb) * Rc * Rd) * Re * Re + 2 * (Ra * Ra * Rb * Rc + (Ra * Ra + Ra * Rb) * Rc * Rc) * Rd + (2 * Ra * Ra * Rb * Rc + Ra * Rc * Rd * Rd + 2 * (Ra * Ra + Ra * Rb) * Rc * Rc + ((3 * Ra + Rb) * Rc * Rc + (2 * Ra * Ra + 3 * Ra * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((Ra * Ra * Rb - (Ra + Rb) * Rc * Rc) * Rd * Rd + (Ra * Rb * Rb - (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rd) * Re * Re + (Ra * Ra * Rb * Rb - (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc) * Rd + (Ra * Ra * Rb * Rb - (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb - (Ra + Rb) * Rc * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((2 * (Ra + Rb) * Rc * Rc + 2 * (Ra + Rb) * Rc * Rd + (2 * Ra * Rb + Rb * Rb) * Rc) * Re * Re + (Ra * Ra * Rb * Rc + (Ra * Ra + Ra * Rb) * Rc * Rc) * Rd + ((2 * Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc * Rc + (2 * Ra * Ra * Rb + Ra * Rb * Rb) * Rc + (2 * (Ra + Rb) * Rc * Rc + (2 * Ra * Ra + 3 * Ra * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((2 * (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + ((Ra * Ra + 3 * Ra * Rb + 2 * Rb * Rb) * Rc * Rc + (Ra * Ra * Rb + 2 * Ra * Rb * Rb) * Rc) * Rd + (Ra * Rb * Rb * Rc + 2 * (Ra + Rb) * Rc * Rd * Rd + (Ra * Rb + Rb * Rb) * Rc * Rc + (2 * (Ra + Rb) * Rc * Rc + (3 * Ra * Rb + 2 * Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -(Ra * Rc * Rd - Rb * Rc * Re) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re) ],
                        [ ((Ra * Rb * Rc + (Ra + Rb) * Rc * Rc) * Rd * Rd - (Rb * Rd * Rd + (Rb * Rb + Rb * Rc) * Rd) * Re * Re - ((Ra * Rb - Ra * Rc) * Rd * Rd + (Ra * Rb * Rb + Rb * Rb * Rc - (Ra + Rb) * Rc * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + ((2 * Ra + Rb) * Rd * Rd + (2 * Ra * Rb + (2 * Ra + Rb) * Rc) * Rd) * Re * Re + ((2 * Ra * Ra + 2 * Ra * Rb + (3 * Ra + 2 * Rb) * Rc) * Rd * Rd + (2 * Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (2 * Ra * Ra + 3 * Ra * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((Ra * Ra * Rb + (Ra * Ra + Ra * Rb) * Rc) * Rd * Rd + (2 * (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + ((2 * Ra * Ra + 3 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + (2 * Ra * Ra * Rb + Ra * Rb * Rb + (2 * Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd - (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc - (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc) * Re * Re - (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc - (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Ra * Rb + 2 * (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + ((Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + 2 * (Ra + Rb) * Rc * Rc + (3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -(Rb * Rd * Re + (Ra * Rb + (Ra + Rb) * Rc) * Rd) / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re) ],
                        [ ((Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + 2 * Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + 2 * Rb * Rb + (2 * Ra + 3 * Rb) * Rc) * Rd) * Re * Re + ((2 * Ra * Rb + (Ra + 2 * Rb) * Rc) * Rd * Rd + (2 * Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (3 * Ra * Rb + 2 * Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((Ra * Rb * Rc + (Ra + Rb) * Rc * Rc - Ra * Rd * Rd - (Ra * Rb - Rb * Rc) * Rd) * Re * Re - ((Ra * Ra + Ra * Rc) * Rd * Rd + (Ra * Ra * Rb + Ra * Ra * Rc - (Ra + Rb) * Rc * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Rb * Rb + 2 * (Ra + Rb) * Rd * Rd + (Ra * Rb + Rb * Rb) * Rc + (3 * Ra * Rb + 2 * Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + ((Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + (Ra * Ra * Rb + 2 * Ra * Rb * Rb + (Ra * Ra + 3 * Ra * Rb + 2 * Rb * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -((Ra * Rb * Rb + 2 * (Ra + Rb) * Rc * Rc + (3 * Ra * Rb + Rb * Rb) * Rc + (Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + (Ra * Ra * Rb + 2 * (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb) * Rc) * Rd) * Re) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd - (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd) / ((Ra * Ra * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb) * Rc) * Rd * Rd + (Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra + Rb) * Rd * Rd + (2 * Ra * Rb + Rb * Rb) * Rc + (2 * Ra * Rb + Rb * Rb + 2 * (Ra + Rb) * Rc) * Rd) * Re * Re + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc) * Rd + (Ra * Ra * Rb * Rb + (Ra * Ra + 2 * Ra * Rb + Rb * Rb) * Rc * Rc + (Ra * Ra + 2 * Ra * Rb + 2 * (Ra + Rb) * Rc) * Rd * Rd + 2 * (Ra * Ra * Rb + Ra * Rb * Rb) * Rc + 2 * (Ra * Ra * Rb + Ra * Rb * Rb + (Ra + Rb) * Rc * Rc + (Ra * Ra + 3 * Ra * Rb + Rb * Rb) * Rc) * Rd) * Re), -(Ra * Rb + (Ra + Rb) * Rc + Ra * Rd) * Re / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re) ],
                        [ -(Rc * Rd + (Rb + Rc + Rd) * Re) / (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re), -((Ra + Rc) * Rd + (Rc + Rd) * Re) / (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re), -(Ra * Rd - Rb * Re) / (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re), -(Ra * Rb + (Ra + Rb) * Rc + Rb * Re) / (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re), -(Ra * Rb + (Ra + Rb) * Rc + Ra * Rd) / (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re), 0 ] ])
            Rf = ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re) / (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re)
            return Rf

        Ra, Rb, Rc, Rd, Re, Rf = R.get_port_impedances()
        R.set_S_matrix ([ [ -2 * ((Rb + Rc) * Rd + (Rb + Rc + Rd) * Re + (Rb + Rc + Rd) * Rf) * Ra / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) + 1, -2 * (Rc * Rd + (Rc + Rd) * Re + Rc * Rf) * Ra / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), 2 * (Rb * Rd + Rb * Re + (Rb + Rd) * Rf) * Ra / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Rb * Re - Rc * Rf) * Ra / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), 2 * (Rb * Rd + (Rb + Rc + Rd) * Rf) * Ra / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Rc * Rd + (Rb + Rc + Rd) * Re) * Ra / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) ],
                [ -2 * (Rc * Rd + (Rc + Rd) * Re + Rc * Rf) * Rb / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * ((Ra + Rc) * Rd + (Ra + Rc + Rd) * Re + (Ra + Rc + Re) * Rf) * Rb / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) + 1, -2 * (Ra * Rd + Ra * Re + (Ra + Re) * Rf) * Rb / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), 2 * (Ra * Re + (Ra + Rc + Re) * Rf) * Rb / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rd - Rc * Rf) * Rb / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * ((Ra + Rc) * Rd + (Rc + Rd) * Re) * Rb / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) ],
                [ 2 * (Rb * Rd + Rb * Re + (Rb + Rd) * Rf) * Rc / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rd + Ra * Re + (Ra + Re) * Rf) * Rc / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * ((Ra + Rb) * Rd + (Ra + Rb) * Re + (Ra + Rb + Rd + Re) * Rf) * Rc / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) + 1, 2 * ((Ra + Rb) * Re + (Ra + Re) * Rf) * Rc / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * ((Ra + Rb) * Rd + (Rb + Rd) * Rf) * Rc / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rd - Rb * Re) * Rc / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) ],
                [ -2 * (Rb * Re - Rc * Rf) * Rd / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), 2 * (Ra * Re + (Ra + Rc + Re) * Rf) * Rd / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), 2 * ((Ra + Rb) * Re + (Ra + Re) * Rf) * Rd / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Re + (Ra + Rc + Re) * Rf) * Rd / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) + 1, -2 * (Ra * Rb + (Ra + Rb) * Rc + Rc * Rf) * Rd / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rb + (Ra + Rb) * Rc + Rb * Re) * Rd / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) ],
                [ 2 * (Rb * Rd + (Rb + Rc + Rd) * Rf) * Re / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rd - Rc * Rf) * Re / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * ((Ra + Rb) * Rd + (Rb + Rd) * Rf) * Re / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rb + (Ra + Rb) * Rc + Rc * Rf) * Re / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd + (Rb + Rc + Rd) * Rf) * Re / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) + 1, -2 * (Ra * Rb + (Ra + Rb) * Rc + Ra * Rd) * Re / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) ],
                [ -2 * (Rc * Rd + (Rb + Rc + Rd) * Re) * Rf / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * ((Ra + Rc) * Rd + (Rc + Rd) * Re) * Rf / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rd - Rb * Re) * Rf / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rb + (Ra + Rb) * Rc + Rb * Re) * Rf / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rb + (Ra + Rb) * Rc + Ra * Rd) * Rf / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf), -2 * (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf / ((Ra * Rb + (Ra + Rb) * Rc) * Rd + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rb) * Rd) * Re + (Ra * Rb + (Ra + Rb) * Rc + (Ra + Rc) * Rd + (Rb + Rc + Rd) * Re) * Rf) + 1 ] ]);

    def set_bass(self, new_bass):

        if not self.bass == new_bass:
            if new_bass <= 0:
                new_bass = 1e-20
            elif new_bass >= 1:
                new_bass = .99999999999999
            self.Pb_plus.set_resistance(self.Pb * new_bass)
            self.Pb_minus.set_resistance(self.Pb * (1 - new_bass))
            self.bass = new_bass

    def set_treble(self, new_treble):

        if not self.treble == new_treble:
            if new_treble <= 0:
                new_treble = 1e-20
            elif new_treble >= 1:
                new_treble = .99999999999999
            self.Pt_plus.set_resistance(self.Pt * new_treble)
            self.Pt_plus.set_resistance(self.Pt * (1 - new_treble))
            self.treble = new_treble

    def process_sample(self, sample):
        if self.adapted:
            return super().process_sample(sample)

        self.Vin.set_voltage(sample)
        self.R_adaptor.compute()
        return self.output.wave_to_voltage()


class SallenKeyFilter(Circuit):
    def __init__(self, fs, cutoff, q_val) -> None:

        self.cap_val = 1e-8
        self.cap_ratio = 22

        self.fs = fs
        self.cutoff = cutoff
        self.q_val = q_val

        self.C1 = Capacitor(self.cap_val * self.cap_ratio, fs)
        self.R2 = Resistor(1e3)
        self.C2 = Capacitor(self.cap_val / self.cap_ratio, fs)

        self.R_adaptor = RTypeAdaptor([self.C1, self.R2, self.C2], self.impedance_calc, 0)

        self.R1 = Resistor(1e6)
        self.S1 = SeriesAdaptor(self.R_adaptor, self.R1)
        self.Vin = IdealVoltageSource(self.S1)

        elements = [
            self.C1,
            self.R2,
            self.C2,
            self.R_adaptor,
            self.R1,
            self.S1,
            self.Vin
        ]

        self.set_components()

        super().__init__(elements, self.Vin, self.Vin, self.Vin)

    def process_sample(self, sample):
        return super().process_sample(sample) + self.R1.wave_to_voltage() + self.C1.wave_to_voltage()

    def set_components(self):
        Rval = 1. / self.cap_val * 2 * np.pi * self.cutoff
        # Rratio = 0.64174
        Rratio = (self.cap_ratio + np.sqrt(self.cap_ratio * self.cap_ratio - 4 * self.q_val * self.q_val)) / (
                    2 * self.q_val)
        self.R1.set_resistance(Rval * Rratio)
        self.R2.set_resistance(Rval / Rratio)

    def set_cutoff(self, new_cutoff):
        if not self.cutoff == new_cutoff:
            self.cutoff = new_cutoff
            self.set_components()

    def set_q_val(self, new_q):
        if not self.q_val == new_q:
            self.q_val = new_q
            self.set_components()

    def impedance_calc(self,R):
        Ag = 100.0
        Ri = 1.0e9
        Ro = 1.0e-1

        Rb, Rc, Rd = R.get_port_impedances()            
        R.set_S_matrix ([ [ 0, -(Rc * Rd + ((Ag + 1) * Rc + (Ag + 1) * Rd) * Ri - Rc * Ro) / ((Rb + Rc) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + Rd) * Ri - (Rb + Rc + Ri) * Ro), -(Rb * Rd + ((Ag + 1) * Rb - Ag * Rd) * Ri - (Rb + Ri) * Ro) / ((Rb + Rc) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + Rd) * Ri - (Rb + Rc + Ri) * Ro), (((Ag + 1) * Rb + Ag * Rc) * Ri - (Rb + Rc + Ri) * Ro) / ((Rb + Rc) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + Rd) * Ri - (Rb + Rc + Ri) * Ro) ],
                            [ -(Rb * Rc * Rd - Rb * Rc * Ro + ((Ag + 1) * Rb * Rc + Rb * Rd) * Ri) / (Rb * Rc * Rd + ((Ag + 1) * Rb * Rc + (Ag + 1) * Rb * Rd) * Ri - (Rb * Rc + (Rb + Rc) * Rd + (Rc + Rd) * Ri) * Ro), -(Rb * Rb * Rc * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rd) * Ri * Ri + (Rb * Rb * Rc - (Rc + Rd) * Ri * Ri + (Rb * Rb - Rc * Rc) * Rd - (Rc * Rc + 2 * Rc * Rd) * Ri) * Ro * Ro + (2 * (Ag + 1) * Rb * Rb * Rc * Rd + (Ag + 1) * Rb * Rb * Rd * Rd) * Ri - (2 * Rb * Rb * Rc * Rd + (Rb * Rb - Rc * Rc) * Rd * Rd - ((Ag + 1) * Rc * Rc + (Ag + 2) * Rc * Rd + Rd * Rd) * Ri * Ri + (2 * (Ag + 1) * Rb * Rb * Rc - 2 * Rc * Rd * Rd + (2 * (Ag + 1) * Rb * Rb - (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro), (Rb * Rb * Rc * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + Ag * Rb * Rd * Rd + ((2 * Ag * Ag + 3 * Ag + 1) * Rb * Rb + (Ag * Ag + Ag) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + 2 * (Rb * Rb + Rb * Rc) * Rd + (Rb * Rc + 2 * Rb * Rd) * Ri) * Ro * Ro + (2 * (Ag + 1) * Rb * Rb * Rc * Rd + ((2 * Ag + 1) * Rb * Rb + Ag * Rb * Rc) * Rd * Rd) * Ri - (2 * Rb * Rb * Rc * Rd + 2 * (Rb * Rb + Rb * Rc) * Rd * Rd + ((Ag + 1) * Rb * Rc + (2 * Ag + 1) * Rb * Rd) * Ri * Ri + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * Rb * Rd * Rd + ((4 * Ag + 3) * Rb * Rb + 3 * (Ag + 1) * Rb * Rc) * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro), (((Ag - 1) * Rb * Rb * Rc + Ag * Rb * Rc * Rc) * Rd * Ri + ((Ag * Ag - 1) * Rb * Rb * Rc + (Ag * Ag + Ag) * Rb * Rc * Rc - ((Ag + 1) * Rb * Rb - Ag * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + Rb * Rc * Ri) * Ro * Ro - (((Ag - 1) * Rb * Rc - Rb * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * Ag * Rb * Rb * Rc + (2 * Ag + 1) * Rb * Rc * Rc - Rb * Rb * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro) ],
                            [ -((Ag + 1) * Rb * Rc * Ri + Rb * Rc * Rd - (Rb * Rc + Rc * Ri) * Ro) / (Rb * Rc * Rd + ((Ag + 1) * Rb * Rc + (Ag + 1) * Rb * Rd) * Ri - (Rb * Rc + (Rb + Rc) * Rd + (Rc + Rd) * Ri) * Ro), (Rb * Rc * Rc * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rd) * Ri * Ri + (Rb * Rc * Rc + 2 * (Rb * Rc + Rc * Rc) * Rd + (Rc * Rc + 2 * Rc * Rd) * Ri) * Ro * Ro + (2 * (Ag + 1) * Rb * Rc * Rc * Rd + (Ag + 1) * Rb * Rc * Rd * Rd) * Ri - (2 * Rb * Rc * Rc * Rd + 2 * (Rb * Rc + Rc * Rc) * Rd * Rd + ((Ag + 1) * Rc * Rc + (Ag + 1) * Rc * Rd) * Ri * Ri + (2 * (Ag + 1) * Rb * Rc * Rc + 2 * Rc * Rd * Rd + (3 * (Ag + 1) * Rb * Rc + (2 * Ag + 3) * Rc * Rc) * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro), -(Rb * Rc * Rc * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc - (Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rd - (Ag + 1) * Rb * Rd * Rd) * Ri * Ri + (Rb * Rc * Rc - Rd * Ri * Ri - (Rb * Rb - Rc * Rc) * Rd + (Rc * Rc - 2 * Rb * Rd) * Ri) * Ro * Ro + (2 * (Ag + 1) * Rb * Rc * Rc * Rd - (Ag + 1) * Rb * Rb * Rd * Rd) * Ri - (2 * Rb * Rc * Rc * Rd - (Rb * Rb - Rc * Rc) * Rd * Rd + ((Ag + 1) * Rc * Rc - 2 * (Ag + 1) * Rb * Rd - Rd * Rd) * Ri * Ri + (2 * (Ag + 1) * Rb * Rc * Rc - 2 * Rb * Rd * Rd - (2 * (Ag + 1) * Rb * Rb - (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro), (((Ag + 1) * Rb * Rb * Rc + (Ag + 2) * Rb * Rc * Rc) * Rd * Ri + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 3 * Ag + 2) * Rb * Rc * Rc + 2 * (Ag + 1) * Rb * Rc * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + Rc * Ri * Ri + (2 * Rb * Rc + Rc * Rc) * Ri) * Ro * Ro - ((2 * (Ag + 1) * Rb * Rc + (Ag + 2) * Rc * Rc + 2 * Rc * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + (2 * Ag + 3) * Rb * Rc * Rc + (3 * Rb * Rc + 2 * Rc * Rc) * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro) ],
                            [ ((Ag + 1) * Rb * Rd * Ri - ((Rb + Rc) * Rd + Rd * Ri) * Ro) / (Rb * Rc * Rd + ((Ag + 1) * Rb * Rc + (Ag + 1) * Rb * Rd) * Ri - (Rb * Rc + (Rb + Rc) * Rd + (Rc + Rd) * Ri) * Ro), -((Ag + 1) * Rb * Rc * Rd * Rd * Ri + ((Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rd + (Ag * Ag + 2 * Ag + 1) * Rb * Rd * Rd) * Ri * Ri - (Rc * Rd * Ri + (Rb * Rc + Rc * Rc) * Rd) * Ro * Ro + ((Rb * Rc + Rc * Rc) * Rd * Rd - ((Ag + 1) * Rc * Rd + (Ag + 1) * Rd * Rd) * Ri * Ri + ((Ag + 1) * Rc * Rc * Rd - ((Ag + 1) * Rb + Ag * Rc) * Rd * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro), (((Ag + 1) * Rb * Rb + 2 * (Ag + 1) * Rb * Rc) * Rd * Rd * Ri + ((Ag * Ag + 3 * Ag + 2) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + 2 * (Ag * Ag + 2 * Ag + 1) * Rb * Rc) * Rd) * Ri * Ri + ((2 * Rb + Rc) * Rd * Ri + Rd * Ri * Ri + (Rb * Rb + Rb * Rc) * Rd) * Ro * Ro - ((Rb * Rb + Rb * Rc) * Rd * Rd + ((Ag + 2) * Rd * Rd + 2 * ((Ag + 1) * Rb + (Ag + 1) * Rc) * Rd) * Ri * Ri + (((Ag + 3) * Rb + (Ag + 2) * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 1) * Rb * Rc) * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro), -(((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd * Ri + (Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd - ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc - (Ag + 1) * Rb * Rd * Rd) * Ri * Ri - (Rb * Rb * Rc + Rb * Rc * Rc + Rc * Ri * Ri + (2 * Rb * Rc + Rc * Rc) * Ri) * Ro * Ro - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd - (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc - Rd * Rd) * Ri * Ri - 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc - (Rb + Rc) * Rd * Rd) * Ri) * Ro) / ((Rb * Rb * Rc + Rb * Rc * Rc) * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb * Rc + (Ag * Ag + 2 * Ag + 1) * Rb * Rc * Rc + (Ag + 1) * Rb * Rd * Rd + ((Ag * Ag + 2 * Ag + 1) * Rb * Rb + (Ag * Ag + 3 * Ag + 2) * Rb * Rc) * Rd) * Ri * Ri + (Rb * Rb * Rc + Rb * Rc * Rc + (Rc + Rd) * Ri * Ri + (Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd + (2 * Rb * Rc + Rc * Rc + 2 * (Rb + Rc) * Rd) * Ri) * Ro * Ro + (((Ag + 1) * Rb * Rb + (Ag + 2) * Rb * Rc) * Rd * Rd + 2 * ((Ag + 1) * Rb * Rb * Rc + (Ag + 1) * Rb * Rc * Rc) * Rd) * Ri - ((Rb * Rb + 2 * Rb * Rc + Rc * Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rc + (Ag + 1) * Rc * Rc + (2 * (Ag + 1) * Rb + (Ag + 2) * Rc) * Rd + Rd * Rd) * Ri * Ri + 2 * (Rb * Rb * Rc + Rb * Rc * Rc) * Rd + (2 * (Ag + 1) * Rb * Rb * Rc + 2 * (Ag + 1) * Rb * Rc * Rc + 2 * (Rb + Rc) * Rd * Rd + (2 * (Ag + 1) * Rb * Rb + 3 * (Ag + 2) * Rb * Rc + (Ag + 2) * Rc * Rc) * Rd) * Ri) * Ro) ] ])

        Ra = (Rb * Rc * Rd + ((Ag + 1) * Rb * Rc + (Ag + 1) * Rb * Rd) * Ri - (Rb * Rc + (Rb + Rc) * Rd + (Rc + Rd) * Ri) * Ro) / ((Rb + Rc) * Rd + ((Ag + 1) * Rb + (Ag + 1) * Rc + Rd) * Ri - (Rb + Rc + Ri) * Ro)
        return Ra


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
    
