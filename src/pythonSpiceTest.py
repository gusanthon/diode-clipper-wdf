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
import PassiveLPF


spice_dir = data_dir / "spice"

fs = 44100

delta = gen_test_wave(fs, 1000, 1, 1, "delta")
sin = gen_test_wave(fs, 1000, 1, 1, "sin")

cutoffs = [70, 150, 250, 500, 1000, 2000, 4000, 8000, 16000]
gains = range(0,50,5)

#######################################

Clipper = DiodeClipper.DiodeClipper(fs)

############################################################################################
###  MEASURE THD+N AT DIFFERENT INPUT GAINS
############################################################################################

for gain in gains:
    Clipper.set_input_gain(gain)
    out = Clipper(sin)
    print(
        f"Diode Clipper THDN with {gain} dB input gain: {round(100 * THDN(out,fs),5)} %\n\n~~~~~~~"
    )
    Clipper.reset()
    compare_waveforms(sin,out,200)   


############################################################################################
###  MEASURE ERROR OF DIODE CLIPPER FREQUENCY RESPONSE AT DIFFERENT CUTOFFS
############################################################################################

error = 'mse' # 'euclidean'
for cutoff in cutoffs:
    Clipper.set_cutoff(cutoff)
    freq_response = Clipper(delta)
    spice_path = spice_dir / f"diode-clipper-frequency-analysis-{cutoff}hz.txt"
    compare_freqz_vs_spice(freq_response,fs,spice_path,title=f'diode clipper cutoff = {cutoff} hz')
    m,p,f = get_freqz_error_vs_spice(freq_response,fs,spice_path,error=error)
    print(f"\nDiode clipper cutoff = {cutoff} hz, with fs = {fs}\n")
    print(f"magnitude {error} error : {m} [dB]")
    print(f"phase {error} error : {p} [radians]\n\n~~~~~~~~~~~")

#######################################

Lpf = PassiveLPF.PassiveLPF(fs)

############################################################################################
###  MEASURE ERROR OF LPF FREQUENCY RESPONSE AT DIFFERENT CUTOFFS
############################################################################################

error = 'mse' # 'euclidean'
for cutoff in cutoffs:
    Lpf.set_cutoff(cutoff)
    freq_response = Lpf(delta)
    spice_path = spice_dir / f"passive-LPF-frequency-analysis-{cutoff}hz.txt"
    # compare_freqz_vs_spice(freq_response,fs,spice_path,title = f"passive LPF cutoff = {cutoff} hz")
    m,p,f = get_freqz_error_vs_spice(freq_response,fs,spice_path,error=error)
    print(f"\nPassive LPF cutoff = {cutoff} hz, with fs = {fs}\n")
    print(f"magnitude {error} error : {m} [dB]")
    print(f"phase {error} error : {p} [radians]\n\n~~~~~~~~~~~")
