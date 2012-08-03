from momo import *

import numpy as np
from math import *

feature_size = 4

def compute( s1, s2, frame ):
  if len( frame ) == 1:
    nearest = frame[0]
  else:
    min_dist = 1E6
    nearest = None
    for p in frame[1:]:
      dist = distance( s2[:2], p[:2] )
      if dist < min_dist:
        min_dist = dist
        nearest = p
  result = nearest - s2
  return result

def mean( features, weights = None ):  
  if weights == None:
    mu = sum( features * 1.0 / features.shape[0] )
  else:
    p    = weights / np.sum( weights )
    mu = sum( features * np.transpose( np.tile( p, (features.shape[1], 1) ) ) )
  return mu

def difference( v1, v2 ):
  return v2 - v1

def covariance( mean, features, weights = None ):
  diff = np.zeros( ( features.shape[0], feature_size ) )
  for i in xrange( features.shape[0] ):
    diff[i] = difference( mean, features[i] )
  if weights == None:
    return np.dot( np.transpose( diff ) * 1.0 / features.shape[0], diff )
  else:
    p    = weights / np.sum( weights )
    return np.dot( np.transpose( diff ) * np.tile( p, (features.shape[1], 1 ) ), diff )


