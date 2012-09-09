import angle
import numpy as np
import random
import os

BASE_DIR = os.path.abspath( os.path.join( os.path.dirname( __file__ ), "..", ".." ) )

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
      snapshot = [d]

def compute_features( module, data ):
  f = []
  old_frames = {}
  for frame in frames( data ):
    tmp = []
    for o_frame, o_time, o_id, o_x, o_y, o_dx, o_dy in frame:
      tmp.append( [o_id, np.array( [o_x, o_y, o_dx, o_dy] )] )
    for i in xrange( len( frame ) ):
      o_id, o = tmp.pop( 0 )
      if o_id in old_frames:
        f.append( module.compute( old_frames[o_id], o, [t[1] for t in tmp] ) )
      tmp.append( [o_id, o] )
    for o_id, o in tmp:
      old_frames[o_id] = o
  return np.array( f )

