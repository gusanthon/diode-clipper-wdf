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
from pyaudio import PyAudio, paFloat32
import scipy.io.wavfile
import time

class DiodeClipper:
    def __init__(
        self, sample_rate, cutoff=1000, input_gain_db=0, output_gain_db=0, n_diodes=2
    ) -> None:

        self.def_cutoff = cutoff
        self.def_in_gain_db = input_gain_db
        self.def_out_gain_db = output_gain_db

        self.fs = sample_rate
        self.cutoff = cutoff
        self.input_gain = 10 ** (input_gain_db / 20)
        self.input_gain_db = input_gain_db
        self.output_gain = 10 ** (output_gain_db / 20)
        self.output_gain_db = output_gain_db

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
        return -(self.C1.wave_to_voltage() * self.output_gain)  ### ¡! phase inverted !¡

    def process_signal(self, signal):
        return np.array([self.process_sample(sample) for sample in signal])

    def __call__(self, *args: any, **kwds: any) -> any:
        if isinstance(args[0], float) or isinstance(args[0], int):
            return self.process_sample(args[0])
        elif hasattr(args[0], "__iter__"):
            return self.process_signal(args[0])

    def set_cutoff(self, new_cutoff):
        self.cutoff = new_cutoff
        self.R = 1.0 / (2 * np.pi * self.C * self.cutoff)
        self.R1.set_resistance(self.R)

    def set_input_gain(self, gain_db):
        self.input_gain = 10 ** (gain_db / 20)
        self.input_gain_db = gain_db

    def set_output_gain(self, gain_db):
        self.output_gain = 10 ** (gain_db / 20)
        self.output_gain_db = gain_db

    def set_num_diodes(self, new_n_diodes):
        self.Dp.set_diode_params(self.Dp.Is, self.Dp.Vt, new_n_diodes)

    def __str__(self):
        return f"{self.__class__.__name__}, ({self.__dict__})"

    def reset(self):
        [element.reset() for element in self.elements]
        self.set_input_gain(self.def_in_gain_db)
        self.set_output_gain(self.def_out_gain_db)
        self.set_cutoff(self.def_cutoff)

    def set_sample_rate(self, fs):
        self.fs = fs
        self.C1.prepare(self.fs)

    def get_freq_response(self, amp=1, delta_dur=1):
        return self.process_signal(
            gen_test_wave(self.fs, None, amp, delta_dur, "delta")
        )

    def record_mono_audio(
        self, duration, chunk=1024, file_name_input="", file_name_output="", callback = None
    ):

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

        wet = np.zeros(duration * self.fs,dtype=np.float32)
        dry = np.zeros(duration * self.fs,dtype=np.float32)
        times = np.zeros(int(duration * self.fs / chunk))
        idx = 0

        for i in range(int(duration * self.fs / chunk)):
            start = time.time()

            data = np.frombuffer(stream.read(chunk), dtype=np.float32)
            processed = np.zeros(len(data),dtype=np.float32)

            for j in range(len(data)):

                processed[j] = self.process_sample(data[j])
                wet[idx] = processed[j]
                dry[idx] = data[j]
                idx += 1

                if callback:
                    callback(self)

            player.write(processed, chunk)
            end = time.time()
            t = round((end - start) * 1000,5) 
            times[i] = t

        stream.stop_stream()
        stream.close()
        p.terminate()
        print(f'AVG time between reading from in buffer to writing to out buffer :\n {np.average(times)} ms\n')

        dry = np.array(dry, dtype=np.float32)
        wet = np.array(wet, dtype=np.float32)

        if file_name_input:
            scipy.io.wavfile.write(file_name_input, self.fs, dry)
        if file_name_output:
            scipy.io.wavfile.write(file_name_output, self.fs, wet)

        return wet, dry 
