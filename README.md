

# Evaluating the Nonlinearities of a Diode Clipper Circuit based on Wave Digital Filters

Diode Clipper circuit & others are built with elements from wave digital filter library in wdf.py

Working on generalizing circuit behavior of wdf's with Circuit.py

## Usage

```python
from DiodeClipper import DiodeClipper

from utils.eval_utils import gen_test_wave

sin = gen_test_wave(fs = 44100, f = 1000, amp = 1, t = 1, kind = 'sin')

Circuit = DiodeClipper(44100)

Circuit.set_input_gain(20)

output = Circuit(sin)

Circuit.plot_freqz()
```
