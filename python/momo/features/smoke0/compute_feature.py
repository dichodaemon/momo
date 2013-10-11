import momo
import numpy as np
from math import *
from __common__ import *

def compute_feature( reference, frame, radius = 3 ):
  feature = np.array( [0.] * FEATURE_LENGTH, dtype = np.float32 )

  for i in xrange( len( frame ) ):
    rel_x = frame[i][:2] - reference[:2]
    l_x = np.linalg.norm( rel_x )
    
    feature[0] = l_x
    feature[1] = reference[2]
    feature[2] = reference[3]
    feature[3] = reference[0]
    feature[4] = reference[1]
  return feature



