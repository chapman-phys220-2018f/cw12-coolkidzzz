###
# Name:Gabriella Nutt 
#Student ID: 2307512
#Email: nutt@chapman.edu
# Raha Pirzadeh 
#Student ID: 2290732
#Email: pirzadeh@chapman.edu
#Course: PHYS220/MATH220/CPSC220 Fall 2018
#Assignment: CW12
###

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import sombrero as som

def test_sombrero():
    assert som.sombrero(-0.9, 0, np.arange(0, 2*np.pi*50, 0.001), 0.18)[-1] < -0.816
    assert som.sombrero(-0.9, 0, np.arange(0, 2*np.pi*50, 0.001), 0.18)[-1] > -0.816