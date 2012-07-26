from momo import *

import numpy as np
from math import *

feature_size = 2

def compute( reference, frame ):
  if len( frame ) == 1:
    nearest = frame[0]
  else:
    min_dist = 1E6
    nearest = None
    for p in frame[1:]:
      dist = distance( reference[:2], p[:2] )
      if dist < min_dist:
        min_dist = dist
        nearest = p
  alpha = angle.as_vector( reference[:2] - nearest[:2] )
  theta = nearest[2:]
  beta  = angle.difference( alpha, theta )
  return np.hstack( [min_dist, beta] )

def mean( features, weights = None ):  
  temp_features = []
  for i in xrange( features.shape[0] ):
    temp_features.append( np.hstack( [[features[i][0]], angle.as_vector( features[i,1] )] ) )
  temp_features = np.array( temp_features )
  if weights == None:
    mu = sum( temp_features * 1.0 / features.shape[0] )
  else:
    p    = weights / np.sum( weights )
    mu = sum( temp_features * np.transpose( np.tile( p, (3, 1) ) ) )
  return np.array( [mu[0], angle.as_angle( mu[1:] )] )

def difference( v1, v2 ):
  return np.array( [v2[0] - v1[0], angle.difference( v1[1], v2[1] )] )

def covariance( mean, features, weights = None ):
  diff = np.zeros( ( features.shape[0], feature_size ) )
  for i in xrange( features.shape[0] ):
    diff[i] = difference( mean, features[i] )
  if weights == None:
    return np.dot( np.transpose( diff ) * 1.0 / features.shape[0], diff )
  else:
    p    = weights / np.sum( weights )
    return np.dot( np.transpose( diff ) * np.tile( p, (3, 1 ) ), diff )


