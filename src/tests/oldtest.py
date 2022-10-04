import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))


import json
import numpy as np
from utils.eval_utils import gen_test_wave, THDN
from utils.plot_utils import (
    compare_freqz_vs_spice,
    freqz,
    get_freqz_error_vs_spice,
    plot_fft,
    compare_waveforms,
    mean_squared_error,
    plot_magnitude_response,
    lin_to_log,
)
from utils.path_utils import data_dir
import DiodeClipper
# import PassiveLPF
import scipy

import matplotlib.pyplot as plt

spice_dir = data_dir / "spice"

cutoffs = [70, 150, 250, 500, 1000, 2000, 4000, 8000, 16000]
gains = range(-20, 50, 5)

sample_rates = [44100, 48000, 88200, 96000,]# 176400, 192000]
# sample_rates = [44100, 48000, 88200, 96000]
rates_khz = {44100: 44.1, 48000: 48, 88200: 88.2, 96000: 96, 176400: 176.4, 192000: 192}

#######################################
fs = 44100
Clipper = DiodeClipper.DiodeClipper(fs)

clipper_data = {
    sample_rate: {
        "cutoffs": {},
    }
    for sample_rate in sample_rates
}

# l = [
#     0.01708,
#     0.01743,
#     0.02869,
#     0.437,
#     6.89743,
#     15.51905,
#     21.85888,
#     26.42446,
#     29.61084,
#     31.97835,
#     33.5589,
#     35.05479,
#     37.11475,
#     38.30848,
# ]

# x = range(-20, 50, 5)
# plt.plot(x, l)
# plt.show()
# Clipper.set_input_gain(10000)
# inp = gen_test_wave(fs, 1000, 1, 1, "sin")
# out = Clipper(inp)
# distortion = round(100 * THDN(out, fs), 5)
# print(distortion)
# plt.plot(inp[:200])
# plt.plot(out[:200])
# plt.show()
############################################################################################
###  MEASURE THD+N AT DIFFERENT INPUT GAINS
############################################################################################

# for sample_rate in sample_rates:
#     sin = gen_test_wave(sample_rate, 1000, 1, 1, "sin")
#     Clipper.set_sample_rate(sample_rate)
#     for gain in gains:
#         Clipper.set_input_gain(gain)
#         out = Clipper(sin)
#         distortion = round(100 * THDN(out,sample_rate),5)
#         print(
#             f"Diode Clipper with {gain} dB input gain, fs = {sample_rate} hz, THDN : {distortion} %\n\n~~~~~~~"
#         )
#         Clipper.reset()
#         # compare_waveforms(sin,out,200)
#         clipper_data_gains[sample_rate][gain] = distortion

# thds_44k = list(clipper_data_gains[44100].values())
# thds_192k = list(clipper_data_gains[192000].values())
# thds_48k = list(clipper_data_gains[48000].values())
# print(mean_squared_error(thds_192k,thds_44k))

############################################################################################
###  MEASURE ERROR OF DIODE CLIPPER FREQUENCY RESPONSE AT DIFFERENT CUTOFFS
############################################################################################



#====================== freqz comparison vs spice ===============#

for sample_rate in sample_rates:
    Clipper.set_sample_rate(sample_rate)
    khz = rates_khz[sample_rate]
    for cutoff in cutoffs:
        Clipper.set_cutoff(cutoff)
        freq_response = Clipper.get_freq_response()
        spice_path = (
            spice_dir /
            f"diodeClipper/{sample_rate}/diode-clipper-{khz}-frequency-analysis-{cutoff}hz.txt"
        )
        # compare_freqz_vs_spice(freq_response,sample_rate,spice_path,title=f'diode clipper cutoff = {cutoff} hz')

        # mse_m, mse_p, mse_f = get_freqz_error_vs_spice(
        #     freq_response, sample_rate, spice_path, error="mse"
        # )
        # euc_m, euc_p, euc_f = get_freqz_error_vs_ spice(
        #     freq_response, sample_rate, spice_path, error="euclidean"
        # )
        esr_m, esr_p = get_freqz_error_vs_spice(freq_response, sample_rate, spice_path, error = 'esr')

        # print(f"\nDiode clipper cutoff = {cutoff} hz, with fs = {sample_rate}\n")
        # print(f"magnitude mse : {mse_m} [dB]")
        # print(f"phase mse : {mse_p} [radians]\n")
        # print(f"magnitude euclidean error : {euc_m} [dB]")
        # print(f"phase euclidean error : {euc_p} [radians]\n\n~~~~~~~~~~~")

        print(f"\nDiode clipper cutoff = {cutoff} hz, with fs = {sample_rate}\n")
        print(f"magnitude esr : {esr_m} [dB]")
        print(f"phase esror : {esr_p} [radians]\n")
        clipper_data[sample_rate]["cutoffs"][cutoff] = {
            # 'freq_response' : freq_response.tolist(),
            'esr' : {'mag' : esr_m.tolist(), 'phase': esr_p.tolist()},
            # "mse": {"mag": mse_m.tolist(), "phase": mse_p.tolist()},
            # "euclidean": {"mag": euc_m.tolist(), "phase": euc_p.tolist()},
        }

print(clipper_data)

with open('ESR_clipper_data.json1','w') as f:
    json.dump(clipper_data,f)
# # print(clipper_data_gains)
# with open("clipper_data_gains.json", "w") as outfile:
#     json.dump(clipper_data_gains, outfile)

#######################################

# Lpf = PassiveLPF.PassiveLPF(44100)

############################################################################################
###  MEASURE ERROR OF LPF FREQUENCY RESPONSE AT DIFFERENT CUTOFFS
############################################################################################
# lpf_data = {sample_rate : {} for sample_rate in sample_rates}

# for sample_rate in sample_rates:
#     khz = rates_khz[sample_rate]
#     Lpf.set_sample_rate(sample_rate)
#     for cutoff in cutoffs:
#         Lpf.set_cutoff(cutoff)
#         freq_response = Lpf.get_freq_response()
#         spice_path = spice_dir / f"passive-LPF-frequency-analysis-{cutoff}hz.txt"
#         spice_path = spice_dir / f"passiveLPF/{sample_rate}" / f"passive-LPF-{khz}-frequency-analysis-{cutoff}hz.txt"

#         # compare_freqz_vs_spice(freq_response,fs,spice_path,title = f"passive LPF cutoff = {cutoff} hz")
#         mse_m,mse_p,mse_f = get_freqz_error_vs_spice(freq_response,sample_rate,spice_path,error='mse')
#         euc_m,euc_p,euc_f = get_freqz_error_vs_spice(freq_response,sample_rate,spice_path,error='euclidean')

#         print(f"\npassive LPF cutoff = {cutoff} hz, with fs = {sample_rate}\n")
#         print(f"magnitude mse : {mse_m} [dB]")
#         print(f"phase mse : {mse_p} [radians]\n")
#         print(f"magnitude euclidean error : {euc_m} [dB]")
#         print(f"phase euclidean error : {euc_p} [radians]\n\n~~~~~~~~~~~")
#         lpf_data[sample_rate][cutoff] = {
#         # 'freq_response' : freq_response.tolist(),
#             'mse' : {'mag' : mse_m.tolist(),'phase' : mse_p.tolist()},
#             'euclidean' : {'mag' : euc_m.tolist(), 'phase' : euc_p.tolist()}

#             }
# print(lpf_data)
# with open("lpf_data.json", "w") as outfile:
#     json.dump(lpf_data, outfile)
