import angle
import numpy as np
import random

def split_line( line ):
  v = line.split()
  return [
    int( v[0] ), float( v[1] ), int( v[2] ), 
    float( v[3] ), float( v[4] ), float( v[5] ), float( v[6] )
  ]

def read_data( filename ):
  f = open( filename )
  data = [split_line( l ) for l in f]
  f.close()
  return data

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

def frames( data ):
  frame    = data[0][0]
  snapshot = []
  for d in data:
    n_frame = d[0]
    if n_frame == frame:
      snapshot.append( d )
    else:
      yield( snapshot )
      frame = n_frame
      snapshot = []

def compute_features( module, data ):
  f = []
  for frame in frames( data ):
    tmp = []
    for o_frame, o_time, o_id, o_x, o_y, o_dx, o_dy in frame:
      tmp.append( np.array( [o_x, o_y, o_dx, o_dy] ) )
    for i in xrange( len( frame ) ):
      f.append( module.compute( tmp ) )
      t = tmp.pop( 0 )
      tmp.append( t )
  return np.array( random.sample( f, max( len( f ), 3000 ) ) )

