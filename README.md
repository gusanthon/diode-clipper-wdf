

# Evaluating the Nonlinearities of a Diode Clipper Circuit Based on Wave Digital Filters

Paper: https://zenodo.org/record/7116075

Based on Jatin Chowdhury's C++ wdf library: https://github.com/jatinchowdhury18/WaveDigitalFilters

Diode Clipper circuit & others in examples directory are built with elements from wave digital filter library in wdf.py

Working on generalizing circuit behavior of wdf's with Circuit.py, a class from which any wave digital circuit built using this library may inherit basic functionalities.

## Usage

```python
from examples.DiodeClipper import DiodeClipper

from utils.eval_utils import gen_test_wave

sin = gen_test_wave(fs = 44100, f = 1000, amp = 1, t = 1, kind = 'sin')

Circuit = DiodeClipper(44100)

Circuit.set_input_gain(20)

output = Circuit(sin)

Circuit.plot_freqz()
```
