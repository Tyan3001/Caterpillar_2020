import numpy as np 

def foruth_poly(data):
    y = [i for i in range(1, len(data) + 1)]
    coefs = np.polyfit(data, y, 4)
    ignore = 0.01
    filtered = [c for c in coefs if c < ignore]
    def bar(x):
        fun = 0
        i = 0
        for d in filtered:
            fun += d * x ** (len(filtered) - i)
            i += 1
        return fun
    return bar
