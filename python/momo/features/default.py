from momo import *

import numpy as np
from math import *

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
  diff = np.zeros( ( features.shape[0], mean.shape[0] ) )
  for i in xrange( features.shape[0] ):
    diff[i] = difference( mean, features[i] )
  if weights == None:
    result = np.dot( np.transpose( diff ) * 1.0 / features.shape[0], diff )
  else:
    p    = weights / np.sum( weights )
    result = np.dot( np.transpose( diff ) * np.tile( p, (features.shape[1], 1 ) ), diff )
  return result + 1E-6 * np.eye( mean.shape[0] )
