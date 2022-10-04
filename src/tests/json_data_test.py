import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.path_utils import data_dir

import json
import matplotlib.pyplot as plt


with open(
    data_dir/"frequency_response/clipper_data.json", "r"
) as json_file:
    clipper_data = json.load(json_file)

mse_mags_avgs = {}
mse_phase_avgs = {}
euclidean_mag_avgs = {}
euclidean_phase_avgs = {}

for sample_rate in clipper_data:
    tot_mse_mag = 0
    tot_mse_phase = 0
    tot_euclidean_mag = 0
    tot_euclidean_phase = 0
    ct = 0
    for cutoff in clipper_data[sample_rate]["cutoffs"]:
        mse_mag = clipper_data[sample_rate]["cutoffs"][cutoff]["mse"]["mag"]
        mse_phase = clipper_data[sample_rate]["cutoffs"][cutoff]["mse"]["phase"]

        euc_mag = clipper_data[sample_rate]["cutoffs"][cutoff]["euclidean"]["mag"]
        euc_phase = clipper_data[sample_rate]["cutoffs"][cutoff]["euclidean"]["phase"]

        tot_mse_mag += mse_mag
        tot_mse_phase += mse_phase
        tot_euclidean_mag += euc_mag
        tot_euclidean_phase += euc_phase
        ct += 1

    avg_mse_mag = tot_mse_mag / ct
    avg_mse_phase = tot_mse_phase / ct
    avg_euc_mag = tot_euclidean_mag / ct
    avg_euc_phase = tot_euclidean_phase / ct

    mse_mags_avgs[sample_rate] = avg_mse_mag
    mse_phase_avgs[sample_rate] = avg_mse_phase
    euclidean_mag_avgs[sample_rate] = avg_euc_mag
    euclidean_phase_avgs[sample_rate] = avg_euc_phase
datas = [mse_mags_avgs, mse_phase_avgs, euclidean_mag_avgs, euclidean_phase_avgs]


for data in datas:
    for fs in data:
        print(f"\nDiode clipper cutoff, with fs = {fs}\n")
        print(f"avg magnitude mse : {mse_mags_avgs[fs]} [dB]")
        print(f"avg phase mse : {mse_phase_avgs[fs]} [radians]\n")
        print(f"avg magnitude euclidean error : {euclidean_mag_avgs[fs]} [dB]")
        print(
            f"avg phase euclidean error : {euclidean_phase_avgs[fs]} [radians]\n\n~~~~~~~~~~~"
        )

# with open("avg_clipper_data.json", "w") as outfile:
#     json.dump(datas, outfile)
for fs in mse_mags_avgs:
    print(mse_mags_avgs[fs])
    plt.plot(fs, mse_mags_avgs[fs], color="blue", marker="x")
plt.show()
