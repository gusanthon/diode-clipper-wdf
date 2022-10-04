import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from utils.eval_utils import gen_test_wave, THDN
from utils.plot_utils import (
    compare_freqz_vs_spice,
    get_freqz_error_vs_spice,
    plot_fft,
    compare_waveforms,
)
from utils.path_utils import data_dir
import DiodeClipper
# import PassiveLPF
import json
import os

spice_dir = data_dir / "spice"

fs = 44100

delta = gen_test_wave(fs, 1000, 1, 1, "delta")
sin = gen_test_wave(fs, 1000, 1, 1, "sin")
sample_rates = [44100, 48000, 88200, 96000, 176400, 192000]
# sample_rates = [176400,192000]
cutoffs = [70, 150, 250, 500, 1000, 2000, 4000, 8000, 16000]
gains = range(0, 50, 5)

dc_dir = spice_dir / "diodeClipper"
lpf_dir = spice_dir / "passiveLPF"
#######################################

Clipper = DiodeClipper.DiodeClipper(48000)
clipper_data = {
    sample_rate: {"cutoffs": {}, "gains": {}} for sample_rate in sample_rates
}

############################################################################################
###  MEASURE THD+N AT DIFFERENT INPUT GAINS
############################################################################################
# for sample_rate in sample_rates:
#     Clipper.set_sample_rate(sample_rate)
#     for gain in gains:
#         Clipper.set_input_gain(gain)
#         out = Clipper(sin)
#         thdn = round(100 * THDN(out,fs),5)
#         print(
#             f"Diode Clipper THDN with {gain} dB input gain: {thdn} %\n\n~~~~~~~"
#         )
#         Clipper.reset()
#         clipper_data[sample_rate]['gains'][gain] = thdn
#         # compare_waveforms(sin,out,200)

############################################################################################
###  MEASURE ERROR OF DIODE CLIPPER FREQUENCY RESPONSE AT DIFFERENT CUTOFFS
############################################################################################
# Clipper.reset()
# Clipper.set_sample_rate(48000)
# Clipper.set_cutoff(16000)
# freq_response = Clipper.get_freq_response()
# compare_freqz_vs_spice(freq_response,48000,'/Users/gusanthon/Documents/UPF/Thesis/diode-clipper-wdf/data/spice/passiveLPF/48000/passive-LPF-48-frequency-analysis-16000hz.txt',title=f'diode clipper cutoff = {16000} hz')

sample_rate = 96000

for sample_rate in sample_rates:
    Clipper.set_sample_rate(sample_rate)
for cutoff in cutoffs:
    Clipper.set_cutoff(cutoff)
    freq_response = Clipper.get_freq_response()
    spice_path = spice_dir / f"diode-clipper-frequency-analysis-{cutoff}hz.txt"
    # spice_path = spice_dir / f"{sample_rate}" / f"diode-clipper-{sample_rate}-frequency-analysis-{cutoff}hz.txt"
    compare_freqz_vs_spice(freq_response,fs,spice_path,title=f'diode clipper cutoff = {cutoff} hz')
    mse_m,mse_p,mse_f = get_freqz_error_vs_spice(freq_response,fs,spice_path,error='mse')
    euc_m,euc_p,euc_f = get_freqz_error_vs_spice(freq_response,fs,spice_path,error='euclidean')

    clipper_data[sample_rate]['cutoffs'][cutoff] = {
        # 'freq_response' : freq_response.tolist(),
        'error' : {
            'mse' : {'mag' : mse_m.tolist(),'phase' : mse_p.tolist()},
            'euclidean' : {'mag' : euc_m.tolist(), 'phase' : euc_p.tolist()}
            }
        }
    print(f"\nDiode clipper cutoff = {cutoff} hz, with fs = {sample_rate}\n")
    print(f"magnitude mse : {mse_m} [dB]")
    print(f"phase mse : {mse_p} [radians]\n")
    print(f"magnitude euclidean error : {euc_m} [dB]")
    print(f"phase euclidean error : {euc_p} [radians]\n\n~~~~~~~~~~~")


# print(clipper_data)

# with open('clipper_data_test.json','w') as outfile:
#     json.dump(clipper_data,outfile)
#######################################

Lpf = PassiveLPF.PassiveLPF(48000)
Lpf.set_cutoff(16000)
freq_response = Lpf.get_freq_response()
compare_freqz_vs_spice(
    freq_response,
    48000,
    "/Users/gusanthon/Documents/UPF/Thesis/diode-clipper-wdf/data/spice/passiveLPF/48000/passive-LPF-48-frequency-analysis-16000hz.txt",
    title=f"lpf cutoff = {16000} hz",
)

############################################################################################
###  MEASURE ERROR OF LPF FREQUENCY RESPONSE AT DIFFERENT CUTOFFS
############################################################################################

error = "mse"  # 'euclidean'
for cutoff in cutoffs:
    Lpf.set_cutoff(cutoff)
    freq_response = Lpf.get_freq_response()
    # spice_path = spice_dir / f"passive-LPF-frequency-analysis-{cutoff}hz.txt"
    spice_path = f"/Users/gusanthon/Documents/UPF/Thesis/diode-clipper-wdf/data/spice/passiveLPF/44100/passive-LPF-44.1-frequency-analysis-{cutoff}hz.txt"
    compare_freqz_vs_spice(
        freq_response, fs, spice_path, title=f"passive LPF cutoff = {cutoff} hz"
    )
    m, p, f = get_freqz_error_vs_spice(freq_response, fs, spice_path, error=error)
    print(f"\nPassive LPF cutoff = {cutoff} hz, with fs = {sample_rate}\n")
    print(f"magnitude {error} error : {m} [dB]")
    print(f"phase {error} error : {p} [radians]\n\n~~~~~~~~~~~")
