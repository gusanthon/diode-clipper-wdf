import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from audioop import rms
import json
import numpy as np
import matplotlib.pyplot as plt
from utils.eval_utils import gen_test_wave, rms_flat
from utils.path_utils import data_dir
from DiodeClipper import DiodeClipper
from utils.plot_utils import freqz, plot_bode, plot_fft, plot_freqz, plot_magnitude_response
import scipy




def _plot_freqz(x: np.ndarray, fs: int, title: str = "Frequency response"):
    # Plot the frequency response of a signal x.
    w, h = scipy.signal.freqz(x, 1, 2**13)
    # w, h = signal.freqz(x, 1)
    magnitude = 20 * np.log10(np.abs(h) + np.finfo(float).eps)
    phase = np.angle(h)
    magnitude_peak = np.max(magnitude)
    top_offset = 10
    bottom_offset = 70
    frequencies = w / (2 * np.pi) * fs

    # _, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 5))
    # Make plot pretty.
    xlims = [10**0, 10 ** np.log10(fs / 2)]
    plt.semilogx(frequencies, magnitude, label="WDF")
    # plt.set_xlim(xlims)
    # plt.set_ylim([magnitude_peak - bottom_offset, magnitude_peak + top_offset])
    # plt.set_xlabel("Frequency [Hz]")
    # plt.set_ylabel("Magnitude [dBFs]")
    plt.grid()
    plt.show()





# from utils.plot_utils import plot_fft, freqz, imag_to_mag,imag_to_phase

fs = 44100
sweep = gen_test_wave(fs, None, 1, 10, "log_sweep")
# inv_sweep = gen_test_wave(fs,None,1,10,'inv_sweep')
# dc = DiodeClipper(fs)

# processed_sweep = dc(sweep)

# IR_out = np.convolve(processed_sweep,inv_sweep)
# IR_in = np.convolve(sweep,inv_sweep)


def polar_to_rect(mag, phase):
    real = mag * np.cos(phase)
    imag = mag * np.sin(phase)
    return real + (imag * 1j)


def imag_to_mag(z):
    return 20 * np.log10(np.sqrt(z.real * z.real + z.imag * z.imag))

def imag_to_phase(z):
    # a, b = map(float, z.split(","))
    b = z.imag
    a = z.real
    return np.arctan2(b, a)

def get_closest(d, search_key):
    if d.get(search_key):
        return search_key, d[search_key]
    key = min(d.keys(), key=lambda key: abs(key - search_key))
    return key, d[key]

with open(data_dir/"IR_out.json", "r") as f:
    IR_out = np.array(json.load(f))

with open(data_dir/"TF_complete.json", "r") as file:
    TF_complete = np.array(json.load(file))

with open(data_dir/"IR_in.json", "r") as file:
    IR_in = np.array(json.load(file))

# sin = gen_test_wave(fs, 1000, 1, 1, "sin")
# plot_fft(sin,fs)
# TF_complete = IR_out / IR_in

# plt.plot(TF_complete)
# plt.show()
# process_fft = freqz(IR_out,fs)
process_fft = np.fft.fft(IR_out,fs)
unprocess_fft = np.fft.fft(IR_in,fs)


# plot_freqz(np.fft.fft(unprocess_fft/process_fft),fs)
# plt.plot(process_fft/unprocess_fft)
# # plt.show()
# plot_freqz(process_fft/unprocess_fft,fs)
# plot_freqz(IR_out,fs)
IR_main = IR_out[(len(sweep) - fs) : (len(sweep) + fs)]
sub_IR_1 = IR_out[360000 : 380000]
sub_IR_2 = IR_out[333000 : 345000]
sub_IR_3 = IR_out[311000 : 323000]
sub_IR_4 = IR_out[295000 : 307000]
sub_IR_5 = IR_out[284000 : 292000]
sub_IR_6 = IR_out[272000 : 282000]
sub_IR_7 = IR_out[263000 : 275000]
sub_IR_8 = IR_out[256000 : 266000]
all_IRs = [IR_main,sub_IR_1,sub_IR_2,sub_IR_3,sub_IR_4,sub_IR_5,sub_IR_6,sub_IR_7,sub_IR_8]
# plt.plot(IR_out)
# plt.show()

# _plot_freqz(IR_main,fs)


def get_mag_from_IR(IR):
    ft = (np.fft.fft(IR))
    ft = ft[:len(ft)//2]
    return  ( (np.sqrt(ft.real**2 + ft.imag**2))) 

sig = (rms_flat(get_mag_from_IR(IR_main)))
# print(rms_flat(get_mag_from_IR(sub_IR_1)))
# print(rms_flat(get_mag_from_IR(sub_IR_2)))
i=1

for IR in all_IRs:
    cont = (rms_flat(get_mag_from_IR(IR)))
    # print(f"harmonic #{i} : {-10 * np.log10((cont/sig)) } \n")
    print(f"harmonic #{i} : {2 * ( (20 * np.log10(sig)) -  ( 20* np.log10(cont))) } \n")
    i+=1

colors = ['grey','olive','purple','green','pink','brown','cyan','red','orange','blue']

for i in range(len(all_IRs)):
    f,h,angles = freqz(all_IRs[i],fs)
    h -= 80
    plot_magnitude_response(f,h,title="Harmonics' ", label=f"harmonic #{i+1}", c = 'tab:'+colors[i])

plt.xlabel('frequency [hz]')
plt.ylabel('magnitude [dB]')
plt.legend()
plt.show()


# plot_freqz(IR_main,fs)
# print(rms_flat(IR_main))
# print(rms_flat(sub_IR_1))
# print(rms_flat(sub_IR_2))
# IR_1_lin = freqz(IR_main,fs)
# for i in range(len(sub_IR_1)):
#     main_mag = dict(zip(IR_main))
#     closest,magval = get_closest()
# plt.plot(IR_main)
# plt.plot(sub_IR_1)
# mags1 = []
# plt.show ()
# plot_fft(sub_IR_1,fs)
# plot_fft(sub_IR_2,fs)
# for z in IR_1_lin:
#     mags1.append(np.sqrt(z.real * z.real + z.imag * z.imag))
# print(mags1)
# plot_freqz(sub_IR_2,fs)
# IR_2_lin = np.fft.fft(sub_IR_1,4096)
# print((IR_1_lin))
# print((IR_1_lin))
# IR_1_lin = plot_fft(IR_main,fs)
# IR_2_lin = plot_fft(sub_IR_1,fs)
# print(len(IR_1_lin))
# print(len(IR_2_lin))
# print(len(IR_1_lin) == len(IR_2_lin))
# print(max(y))
# print(min(y))
# plot_freqz(sub_IR_8,fs)
# plot_freqz(sub_IR_7,fs)
# plot_freqz(sub_IR_6,fs)
# plot_freqz(sub_IR_5,fs)
# plot_freqz(sub_IR_4,fs)
# plot_freqz(sub_IR_3,fs)
# plot_freqz(sub_IR_2,fs)
# plot_freqz(sub_IR_1,fs)
# plot_freqz(IR_main,fs)







# plt.plot(IR_out)
# plt.show()

# f,H,angles = freqz(IR_in,fs)
# rect_in = polar_to_rect(H,angles)
# f,h,angles = freqz(IR_out,fs)
# rect_out = polar_to_rect(H,angles)
# TF = rect_in / rect_out
# print(TF)
# mag = imag_to_mag(TF)
# # print(mag)

# p = (np.angle(imag_to_phase(TF)))
# plt.plot(f,p)
# plt.show()
# mag = imag_to_mag(TF)
# phase = imag_to_phase(TF)
# plt.plot(f,mag)
# plt.show()
# plt.plot(f,phase)
# plt.show()
##TF = FFT (unprocess ) / FFT (PROCESS ED)


# print(rect)


# IR_sub1 = IR_out[int(370867-(fs/2)):int(370867+(fs/2))]
# print(IR_sub1.shape)
# plt.plot(IR_sub1)
# plt.show()
# plot_fft(IR_sub1,fs)
# plot_fft(np.pad(IR_sub1,int(fs/2)),fs)
# plot_fft(IR_sub1,fs)
