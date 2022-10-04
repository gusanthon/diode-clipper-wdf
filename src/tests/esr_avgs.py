import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils.path_utils import data_dir
import json
import numpy as np

with open(data_dir/'frequency_response/ESR_clipper_data.json') as f:
    esr_data = json.load(f)

print(esr_data)
mag_avgs = {}
phase_avgs = {}
for sample_rate in esr_data:
    tot_mag = 0
    tot_phase = 0
    for cutoff in esr_data[sample_rate]['cutoffs']:
        tot_mag += esr_data[sample_rate]['cutoffs'][cutoff]['esr']['mag']
        tot_phase += esr_data[sample_rate]['cutoffs'][cutoff]['esr']['phase']
    # print(len(esr_data[sample_rate]['cutoffs']))
    # print(tot_mag/len(esr_data[sample_rate]['cutoffs']))
    mag_avgs[sample_rate] = tot_mag/len(esr_data[sample_rate]['cutoffs'])
    phase_avgs[sample_rate] = tot_phase/len(esr_data[sample_rate]['cutoffs'])

print(mag_avgs)
tot = 0
for fs in mag_avgs:
    tot += phase_avgs[fs]
print(tot/len(mag_avgs))


# print(phase_avgs)