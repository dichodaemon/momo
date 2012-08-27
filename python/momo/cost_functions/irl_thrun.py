import momo
import cvxopt
import numpy as np
import cPickle
from math import *

class irl_thrun( object ):
  def __init__( self, module, data = None ):
    self.module = module
    self.x = 0
    self.y = 0
    self.width = 0
    self.height = 0

    if data != None:
      frame_data = self.preprocess( data )

      self.w      = 2 * np.random.rand( frame_data.values()[0]["feature_sum"].shape[0] ) - 1.
      feature_sum = self.w * 0.
      for o_id in xrange( 50, 51 ):
        feature_sum += frame_data[o_id]["feature_sum"]

      temp_sum = self.w * 0.
      
      for o_id in xrange( 50, 53 ):
        temp_sum += self.plan_features( frame_data[o_id] )
      print self.w
  
  def preprocess( self, data ):
    min_x = 0.
    min_y = 0.
    max_x = 0.
    max_y = 0.
    frame_data = {}
    for frame in momo.frames( data ):
      tmp = []
      for o_frame, o_time, o_id, o_x, o_y, o_dx, o_dy in frame:
        if o_x < min_x:
          min_x = o_x
        if o_y < min_y:
          min_y = o_y
        if o_x > max_x:
          max_x = o_x
        if o_y > max_y:
          max_y = o_y
        tmp.append( [o_id, np.array( [o_x, o_y, o_dx, o_dy] )] )
      for i in xrange( len( frame ) ):
        o_id, o = tmp.pop( 0 )
        feature = self.module.compute( o, o, [t[1] for t in tmp] )
        if not o_id in frame_data:
          frame_data[o_id] = {
            "features": [],
            "states": [], 
            "frames": []
          }
        frame_data[o_id]["states"].append( o )
        frame_data[o_id]["features"].append( feature )
        frame_data[o_id]["frames"].append( [f[1] for f in tmp] )
        tmp.append( [o_id, o] )
    self.x = min_x
    self.y = min_y
    self.width = max_x - min_x
    self.height = max_y - min_y
    if self.width > self.height:
      self.delta = self.width / 64
    else:
      self.delta = self.height / 64

    self.grid = momo.planning.grid( 
      self.x, self.y, self.width, self.height, self.delta
    )

    print "*" * 80
    print "%i paths to process" % len( frame_data.items() )
    print "Upper corner: (%f, %f)" % ( self.x, self.y )
    print "Dimensions: (%f, %f)" % ( self.width, self.height )
    print "Grid dimensions: (%i, %i)" %( ceil( self.width / self.delta ), ceil( self.height / self.delta ) )

    for o_id, frame in frame_data.items():
      oi = None
      oj = None
      ok = None
      result = frame["features"][0] * 0
      features = []
      states = []
      frames = []
      for index in xrange( len( frame["states"] ) ):
        x, y, vx, vy = frame["states"][index]
        if abs( vx * vy ) > 0.0:
          angle = momo.angle.as_angle( np.array( [vx, vy] ) )
          i, j, k = self.grid.from_world( x, y, angle )
          if i != oi or j != oj or k != ok:
            oi = i
            oj = j
            ok = k
            states.append( frame["states"][index] )
            features.append( frame["features"][index] )
            frames.append( frame["frames"][index] )
          if "feature_sum" not in frame:
            frame["feature_sum"] = frame["features"][0] * 0
          frame["feature_sum"] += frame["features"][index]
      frame["features"] = features
      frame["states"] = states
      frame["frames"] = frames
    return frame_data

  def plan_features( self, frame_data, h = 3 ):
    count = 0
    start = frame_data["states"][0]
    goal  = frame_data["states"][-1]
    for index in xrange( len( frame_data["features"] ) ):
      if count % h == 0:
        for i, j, k, x, y, angle, value in self.grid:
          angle = momo.angle.as_vector( angle )
          state = np.array( [x, y, angle[0], angle[1]] )
          self.grid[i, j, k] = self( state, state, frame_data["frames"][index] )
      count += 1
    return 0


  def __call__( self, s1, s2, frame ):
    value = self.module.compute( s1, s2, frame )
    return np.dot( value, self.w )

  def save( self, stream ):
    pass
    #cPickle.dump( [self.module.__name__, self.mu, self.sigma, self.inv_sigma], stream )

  @staticmethod
  def load( stream ):
    module, mu, sigma, inv_sigma = cPickle.load( stream )
    module = momo.features.__dict__[module.split( "." )[-1]]
    result = mahalanobis( module )
    result.mu = mu
    result.sigma = sigma
    result.inv_sigma = inv_sigma
    return result

