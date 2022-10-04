import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))


# import os
from pyaudio import PyAudio, paFloat32, paContinue
import numpy as np

# from DiodeClipper import DiodeClipper
# from PassiveLPF import PassiveLPF
# import scipy.io.wavfile
# from Circuit import *


rate = 44100
chunk = 2**12
width = 2

p = PyAudio()

# callback function to stream audio, another thread.
def callback(in_data, frame_count, time_info, status):
    audio = np.frombuffer(in_data, dtype=np.float32)
    return (audio, paContinue)


# create a pyaudio object
inStream = p.open(
    format=paFloat32,
    channels=1,
    rate=rate,
    input=True,
    frames_per_buffer=chunk,
    stream_callback=callback,
)

"""
Setting up the array that will handle the timeseries of audio data from our input
"""
audio = np.zeros((chunk), dtype=np.float32)

inStream.start_stream()

while True:
    try:
        pass  # any function to run parallel to the audio thread, running forever, until ctrl+C is pressed.

    except KeyboardInterrupt:

        inStream.stop_stream()
        inStream.close()
        p.terminate()
        print("* Killed Process")
        quit()


# CHUNK = 2**10
# RATE = 44100
# LEN = 20

# # p = pyaudio.PyAudio()

# clipper = DiodeClipper(RATE)
# lpf = PassiveLPF(RATE)
# clipper.set_input_gain(30)
# clipper.record_mono_audio(5)

# stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
# player = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK)

# print(f'input latency : {stream.get_input_latency()}')
# print(f'output latency : {stream.get_output_latency()}')


out = []
cutoff = 20000
down = True

#########################################################################################################
### LPF TEST
#########################################################################################################
# for i in range((int(LEN * RATE / CHUNK))): #go for a LEN seconds

#     data = np.frombuffer(stream.read(CHUNK),dtype=np.float32)
#     processed = np.array(lpf.process_signal(data),dtype=np.float32)

#     if down:
#         cutoff -= 100
#     else:
#         cutoff += 100
#     if cutoff <= 20:
#         cutoff += 100
#         down = False
#     elif cutoff >= 20e3:
#         cutoff -= 100
#         down = True

#     lpf.set_cutoff(cutoff)

#     for i in range(len(processed)):
#         out.append( processed[i])

#     player.write(processed,CHUNK)
#########################################################################################################
### DIODE CLIPPER TEST
###########################################################################

# input_gain = -30
# LEN = 5
# out = []
# clipper.set_input_gain(50)
# for i in range((int(LEN * RATE / CHUNK))): #go for a LEN seconds

#     data = np.frombuffer(stream.read(CHUNK),dtype=np.float32)
#     # clipper.set_input_gain(input_gain)
#     processed = np.array(clipper.process_signal(data),dtype=np.float32)

#     input_gain += .1

#     for i in range(len(processed)):
#         out.append( processed[i])

#     player.write(processed,CHUNK)

# out = np.array(out)

# stream.stop_stream()
# stream.close()
# p.terminate()

# scipy.io.wavfile.write('./test.wav',RATE,out)
# os.system('open test.wav')
