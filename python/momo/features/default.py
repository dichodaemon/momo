from momo import *

import numpy as np
from math import *

feature_size = 2

def mean( features, weights = 1.0 ):  
  p    = weights / np.sum( weights )
  mu = sum( features * np.transpose( np.tile( p, (2, 1) ) ) )
  return mu

def difference( v1, v2 ):
  return v2 - v1

def covariance( mean, features, weights ):
  p = weights / np.sum( weights )
  diff = np.zeros( ( features.shape[0], feature_size ) )
  for i in xrange( features.shape[0] ):
    diff[i] = difference( mean, features[i] )
  return np.dot( np.dot( np.transpose( diff ), np.diag( p ) ), diff )


