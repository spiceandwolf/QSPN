import numpy as np

u = None
l = None
e = None
Nx = None
Ny = None
b = None
k = None
max_ratio = None
min_bias = None

def set_dynfactor_function(attr_bound):
    global u, l, e, Nx, Ny, b, k, max_ratio, min_bias
    if attr_bound is not None:
        assert len(attr_bound) == 4 and attr_bound[1] < attr_bound[2]
        u = attr_bound[2]
        l = attr_bound[1]
        e = np.exp(1)
        Nx = attr_bound[0]
        Ny = 1.0
        b = Ny*(u-(e**Nx)*l)/(1-(e**Nx))
        k = np.log(u-b)
        max_ratio = attr_bound[3]
        min_bias = 1 / max_ratio

def calc_dynfactor_ratio(mcv_freq, others_freq):
    global u, l, e, Nx, Ny, b, k, max_ratio, min_bias
    assert max_ratio is not None
    ratio = mcv_freq / others_freq if mcv_freq < max_ratio * others_freq else max_ratio
    ratio = ratio / max_ratio - min_bias
    return max(min(float((np.power(e, -Nx*ratio+k) + b)/Ny) + min_bias, 1), 0)