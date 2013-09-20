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
    rel_x = frame[i][:2] - reference[:2]
    l_x = np.linalg.norm( rel_x )

    n     = rel_x / l_x
    e     = frame[i][:2] / np.linalg.norm( frame[i][:2] )
    cos_phi = np.dot( -n, e )
    force = ( LAMBDA + 0.5 * ( 1 - LAMBDA ) * ( 1 + cos_phi ) ) * exp( 2 * radius - l_x ) 
    i = max_idx( force, ANGLES )
    if force > 0.5:
      feature[max( i, 0 )] += 1
  return feature



