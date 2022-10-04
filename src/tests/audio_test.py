import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))


from DiodeClipper import DiodeClipper
from utils.eval_utils import read_to_linear, gen_test_wave, SNRsystem
import scipy.io.wavfile
import numpy as np

from pydub import AudioSegment


def make_wav_mono(path):
    sound = AudioSegment.from_wav(path)
    sound = sound.set_channels(1)
    sound.export(path, format="wav")


# make_wav_mono('/Users/gusanthon/Desktop/guitar_samp1.wav')

# fs,x = read_to_linear('/Users/gusanthon/Desktop/guitar_samp1.wav',32)

fs = 44100

Circuit = DiodeClipper(fs, input_gain_db=5)

Circuit.set_input_gain(0)

# output = Circuit(x)

# freq_response = Circuit.get_freq_response()

sin = gen_test_wave(fs, 1000, 1, 1, "sin")

output = Circuit(sin)

SNRsystem(sin, output)


# scipy.io.wavfile.write('/Users/gusanthon/Desktop/5_db_processed_guitar_samp1.wav',fs,output)
