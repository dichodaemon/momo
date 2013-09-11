import momo
import numpy as np
from math import *
from __common__ import *

def max_idx( value, reference ):
  cur_idx = 0
  for i in xrange( len( reference ) ):
    if value >= reference[i]:
      cur_idx = i
  return cur_idx
      

def compute_feature( reference, frame, radius = 3 ):
  density = 0
  feature = np.array( [0.] * FEATURE_LENGTH, dtype = np.float32 )
  bin_count = np.zeros( (3,), dtype = np.float32 )
  bin_sum   = np.zeros( (3,), dtype = np.float32 )

  for i in xrange( len( frame ) ):
    dist  = momo.distance( frame[i][:2], reference[:2] )
    if dist < radius:
      density += 1
      rel_v = frame[i][2:] - reference[2:]
      rel_x = frame[i][:2] - reference[:2]
      l_v = np.linalg.norm( rel_v )
      l_x = np.linalg.norm( rel_x )
      a = np.dot( rel_v / l_v, rel_x / l_x )
      i = max_idx( a, ANGLES )
      bin_count[i] += 1
      bin_sum[i] += l_v

  if density > 0:
    feature[max_idx( density, DENSITIES )] = 1
    for angle in xrange( 3 ):
      if bin_count[angle] > 0:
        l = bin_sum[angle] / bin_count[angle]
        feature[3 + angle * 3 + max_idx( l, SPEEDS )] = 1
  return feature



