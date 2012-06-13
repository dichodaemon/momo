from prediction import *

import numpy as np
from math import *

feature_size = 2

def compute( p1, p2 ):
  dist  = distance( p1[:2], p2[:2] )
  alpha = angle.as_vector( p1[:2] - p2[:2] )
  theta = p2[2:4]
  beta  = angle.difference( alpha, theta )
  return np.hstack( [dist, beta] )

def mean( features, weights = 1.0 ):  
  p    = weights / np.sum( weights )
  temp_features = []
  for i in xrange( features.shape[0] ):
    temp_features.append( np.hstack( [[features[i][0]], angle.as_vector( features[i,1] )] ) )
  temp_features = np.array( temp_features )
  mu = sum( temp_features * np.transpose( np.tile( p, (3, 1) ) ) )
  return np.array( [mu[0], angle.as_angle( mu[1:] )] )

def difference( v1, v2 ):
  return np.array( [v2[0] - v1[0], angle.difference( v1[1], v2[1] )] )

def covariance( mean, features, weights ):
  p = weights / np.sum( weights )
  diff = np.zeros( ( features.shape[0], feature_size ) )
  for i in xrange( features.shape[0] ):
    diff[i] = difference( mean, features[i] )
  return np.dot( np.dot( np.transpose( diff ), np.diag( p ) ), diff )


