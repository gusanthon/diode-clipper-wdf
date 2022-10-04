import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))


from DiodeClipper import DiodeClipper
from PassiveLPF import PassiveLPF


fs = 96000
Clipper = DiodeClipper(fs, cutoff=20000)
Lpf = PassiveLPF(fs, cutoff=20e3)


def gain_mod(circuit):
    input_gain = circuit.input_gain_db

    if int(input_gain) - 0.0001 < input_gain and input_gain < int(input_gain) + 0.0001:
        print(f"{input_gain} dB")
    input_gain += 0.0001
    if input_gain >= 100:
        input_gain = 0

    circuit.set_input_gain(input_gain)


def filter_mod(circuit):
    cutoff = circuit.cutoff
    cutoff -= 0.01
    if cutoff <= 20:
        cutoff = 20e3
    # gain_mod(circuit)
    circuit.set_cutoff(cutoff)


# Lpf.record_mono_audio(duration=10)
