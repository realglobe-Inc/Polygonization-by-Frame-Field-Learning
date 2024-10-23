
from memory_profiler import profile
import numpy as np

@profile()
def test():
    N = 10000000
    D = 3
    X = np.random.randn(N, D)

test()