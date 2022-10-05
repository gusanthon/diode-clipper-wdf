import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from utils.eval_utils import gen_test_wave
from utils.plot_utils import plot_freqz
import scipy.io.wavfile
from wdf import *
import time
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
            f'AVG time taken to read from input buffer, process audio and write to output buffer :\n {np.average(times)} ms\n'
        )

        if file_name_input:
            scipy.io.wavfile.write(file_name_input, self.fs, dry)
        if file_name_output:
            scipy.io.wavfile.write(file_name_output, self.fs, wet)

        return wet, dry

