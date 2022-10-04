import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.eval_utils import gen_test_wave
from wdf import Resistor, Capacitor, SeriesAdaptor, ParallelAdaptor, IdealVoltageSource
import numpy as np
from pyaudio import PyAudio, paFloat32
import scipy.io.wavfile
import time


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

    def process_signal(self, signal):
        self.reset()
        return np.array([self.process_sample(sample) for sample in signal])

    def __call__(self, *args: any, **kwds: any) -> any:
        if isinstance(args[0], float) or isinstance(args[0], int):
            return self.process_sample(args[0])
        elif hasattr(args[0], "__iter__"):
            return self.process_signal(args[0])

    def set_cutoff(self, new_cutoff):
        self.cutoff = new_cutoff
        self.C = (1.0 / self.Z) / (2 * np.pi * self.cutoff)
        self.C1.set_capacitance(self.C)
        self.C2.set_capacitance(self.C)

    def reset(self):
        [element.reset() for element in self.elements()]
        self.set_cutoff(self.def_cutoff)

    def get_impulse_response(self, amp=1, delta_dur=1):
        return self.process_signal(
            gen_test_wave(self.fs, None, amp, delta_dur, "delta")
        )

    def set_sample_rate(self, fs):
        self.fs = fs
        self.C1.prepare(self.fs)
        self.C2.prepare(self.fs)

    def __str__(self):
        return f"{self.__class__.__name__}, ({self.__dict__})"

    def record_mono_audio(
        self,
        duration,
        chunk=1024,
        file_name_input="",
        file_name_output="",
        callback=None,
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

            data = np.frombuffer(stream.read(chunk), dtype=np.float32)
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
            f"AVG time taken to read from input buffer, process audio and write to output buffer :\n {np.average(times)} ms\n"
        )

        dry = np.array(dry, dtype=np.float32)
        wet = np.array(wet, dtype=np.float32)

        if file_name_input:
            scipy.io.wavfile.write(file_name_input, self.fs, dry)
        if file_name_output:
            scipy.io.wavfile.write(file_name_output, self.fs, wet)

        return wet, dry
