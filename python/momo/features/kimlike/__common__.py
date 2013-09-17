import numpy as np
from math import *

FEATURE_LENGTH = 9

DENSITIES  = np.array( [ 0., 2.0, 5.0], dtype = np.float32 )
SPEEDS     = np.array( [0.0, 0.015, 0.025], dtype = np.float32 )
ANGLES     = np.array( [-1, cos( 3 * pi / 4 ), cos( pi / 4 )], dtype = np.float32 )

dns_i = 0
spd_i = 3
dir_i = 6
dst_i = 9
