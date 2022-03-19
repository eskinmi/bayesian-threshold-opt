# bayesian-threshold-optimisation

This repository contains a small script that implements the naive bayesian threshold optimisation utilities.

### application

```python
import numpy as np
from src.optimisers import THOpt

y = []  # add labels
X = [[], []]  # add features
ths = [np.linspace(0, 1, 10), np.linspace(0.1, 0.5, 20)]

opti = THOpt(y, X, ths, prior=0.1)
ths, proba, _ = opti.opt(0.95)
```
