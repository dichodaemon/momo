import angle
import numpy as np

def distance( v1, v2 ):
  diff = v1 - v2
  return np.dot( diff, diff )**0.5

def mahalanobis( v1, v2, inv_covariance ):
  diff = v1 - v2
  return np.dot( np.dot( diff, inv_covariance ), diff )

def synchronized( v1, v2 ):
  return all( v1[:, 2] == v2[:, 2] )

def compute_angles( data ):
  angles = np.zeros( data.shape )
  for i in xrange( data.shape[0] ):
    p1 = data[i]
    if i > 0:
      p1 = data[i - 1]
    p2 = data[i]
    if i < data.shape[0] - 1:
      p2 = data[i + 1]
    angles[i] = angle.as_vector( p2 - p1 )
  return np.hstack( [data, angles] )

def compute_features( module, p1, p2 ):
  if not synchronized( p1, p2 ):
    raise "Arrays are not synchronized"

  p2 = compute_angles( p2[:, :2] )

  f = np.zeros( (p2.shape[0], module.feature_size ) )
  for i in xrange( p2.shape[0] ):
    f[i] = module.compute( p1[i], p2[i] )
  return f

