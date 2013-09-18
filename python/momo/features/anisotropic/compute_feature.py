import momo
import numpy as np
from math import *
from __common__ import *

def max_idx( value, reference ):
  cur_idx = -1
  for i in xrange( len( reference ) ):
    if value >= reference[i]:
      cur_idx = i
  return cur_idx
      

def compute_feature( reference, frame, radius = 3 ):
  feature = np.array( [0.] * FEATURE_LENGTH, dtype = np.float32 )

  for i in xrange( len( frame ) ):
    rel_v = frame[i][2:] - reference[2:]
    rel_x = frame[i][:2] - reference[:2]
    l_v = np.linalg.norm( rel_v )
    l_x = np.linalg.norm( rel_x )

    n     = rel_x / l_x
    e     = rel_v / l_v
    cos_phi1 = np.dot( -n, e )
    cos_phi2 = np.dot( -n, frame[i][2:] / np.linalg.norm( frame[i][2:] ) )

    force1 = ( 1 + cos_phi1 ) * exp( 2 * radius - l_x ) * l_v
    force2 = ( LAMBDA + 0.5 * ( 1 - LAMBDA ) * ( 1 + cosPhi2 ) ) * exp( 2 * radius - l_x ) 
    feature[0] += force1
    feature[1] += force2
  return feature



