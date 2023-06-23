#%pylab inline

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pyiga'))
import pyiga


import scipy
import itertools
from pathlib import Path
#print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())
from pyiga import bspline, assemble, vform, geometry, vis, solvers

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def geo_annulus(r_in=3.7, r_out=4.5):
    
    # define geometry
    # realistic model of an artery: r_inner= 4mm, r_outer= 7mm (preprint)
    # human carotid artery: r_i= 3.7 mm, r_o= 4.5mm (Holzapfel)

    #r_out = 4
    #r_in = 3.1

    geos = [

        geometry.quarter_annulus(r1=r_in, r2=r_out),
        geometry.quarter_annulus(r1=r_in, r2=r_out).rotate_2d(3*np.pi/2),
        geometry.quarter_annulus(r1=r_in, r2=r_out).rotate_2d(np.pi),
        geometry.quarter_annulus(r1=r_in, r2=r_out).rotate_2d(np.pi/2)
    ]
    
    return geos


