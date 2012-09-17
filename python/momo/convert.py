import numpy as np
import momo
from math import *
import random

ANGLES     = [0, pi / 4, pi / 2, 3 * pi / 4, pi, -3 * pi / 4, -pi / 2, -pi / 4]

class convert( object ):
  def __init__( self, data, delta ):
    self.x = min( data, key = lambda x: x[3] )[3] - delta / 2
    self.y = min( data, key = lambda y: y[4] )[4] - delta / 2
    self.width  = max( data, key = lambda x: x[3] )[3] - self.x + delta / 2
    self.height = max( data, key = lambda y: y[4] )[4] - self.y + delta / 2
    self.delta = delta
    self.grid_width = int( ceil( self.width / self.delta ) )
    self.grid_height = int( ceil( self.height / self.delta ) )

    self.width = self.delta * self.grid_width
    self.height = self.delta * self.grid_height
    self.x2 = self.x + self.width
    self.y2 = self.y + self.height

  def rebase_frame( self, frame ):
    origin = self.to_world2( np.array( [0, 0, 0] ) )
    origin[2:] *= 0
    return ( frame[:] - origin ).astype( np.float32 )
  
  def from_world( self, v ):
    angle = v[2]
    while angle >  pi:
      angle -= 2 * pi
    while angle < - pi:
      angle += 2 * pi
    dist = 2 * pi
    k = 0
    for i in xrange( len( ANGLES ) ):
      a = ANGLES[i]
      d = abs( a - angle )
      if d < dist:
        k = i
        dist = d
    return np.array( [
      int( round( ( v[0] - self.x ) / self.delta ) - 0.5 ),\
      int( round( ( v[1] - self.y ) / self.delta ) - 0.5 ), k
    ], dtype = np.int32 )

  def from_world2( self, v ):
    tmp = np.array( [v[0], v[1], atan2( v[3], v[2] )] )
    return self.from_world( tmp )

  def to_world( self, v ):
    return np.array( [
      self.x + self.delta * ( v[0] + 0.5 ), 
      self.y + self.delta * ( v[1] + 0.5 ), 
      ANGLES[v[2]]
    ], dtype = np.float32 )

  def to_world2( self, v, speed = 1.0 ):
    r = self.to_world( v )
    return np.array( [r[0], r[1], cos( r[2] ) * speed, sin( r[2] ) * speed], dtype = np.float32 )

  def random_world2( self ):
    return self.to_world2( self.random() )

  def random_world( self ):
    return self.to_world( self.random() )
  
  def random( self ):
    x = random.randint( 3, self.grid_width - 4 )
    y = random.randint( 3, self.grid_height - 4 )
    k = random.randint( 0, 7 )
    return np.array( [x, y, k], dtype = np.int32 )
    
  def preprocess_data( self, data ):
    frame_data = {}
    for frame in momo.frames( data ):
      tmp = []
      for o_frame, o_time, o_id, o_x, o_y, o_dx, o_dy in frame:
        tmp.append( [o_id, np.array( [o_x, o_y, o_dx, o_dy] )])

      for i in xrange( len( tmp ) ):
        o_id, o = tmp.pop( 0 )
        if not o_id in frame_data:
          frame_data[o_id] = {
            "times": [],
            "frame_nums": [],
            "states": [], 
            "frames": []
          }
        frame_data[o_id]["states"].append( o )
        frame_data[o_id]["frames"].append( [f[1] for f in tmp] )
        frame_data[o_id]["times"].append( tmp[0][1][1] )
        frame_data[o_id]["frame_nums"].append( tmp[0][1][0] )
        tmp.append( [o_id, o] )

    # Rasterize to grid
    for o_id, frame in frame_data.items():
      old_grid_s = None
      states = []
      frames = []
      for index in xrange( len( frame["states"] ) ):
        state = frame["states"][index]
        angle = momo.angle.as_angle( np.array( state[2:] ) )
        grid_s = self.from_world( np.array( [state[0], state[1], angle] ) )
        if type( old_grid_s ) == type( None ) or not (old_grid_s == grid_s).all():
          old_grid_s = grid_s
          states.append( frame["states"][index] )
          frames.append( np.array( frame["frames"][index], dtype = np.float32 ) )
      frame["states"] = states
      frame["frames"] = frames
    return frame_data
