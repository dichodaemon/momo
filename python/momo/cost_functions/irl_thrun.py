import momo
import cvxopt
import numpy as np
import cPickle
from math import *
import pylab as pl

class irl_thrun( object ):
  def __init__( self, module, data = None ):
    self.module = module
    self.x = 0
    self.y = 0
    self.width = 0
    self.height = 0

    if data != None:
      frame_data = self.preprocess( data )

      #self.w      = 2 * np.random.rand( frame_data.values()[0]["feature_sum"].shape[0] ) - 1.
      self.w      = np.random.rand( frame_data.values()[0]["feature_sum"].shape[0] ) 
      #self.w /= self.w
      feature_sum = self.w * 0.
      for o_id in xrange( 50, 51 ):
        feature_sum += frame_data[o_id]["feature_sum"]

      temp_sum = self.w * 0.
      
      for o_id in xrange( 50, 53 ):
        temp_sum += self.plan_features( frame_data[o_id] )
  
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
    self.width = ( max_x - min_x ) * 1.01
    self.height = ( max_y - min_y ) * 1.01
    if self.width > self.height:
      self.delta  = self.width / 64
    else:
      self.delta = self.height / 64

    self.grid = momo.planning.grid( 
      self.x, self.y, self.width, self.height, self.delta
    )

    self.width = self.grid.width
    self.height = self.grid.height

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
    start = frame_data["states"][0] * 1
    goal  = frame_data["states"][-1]
    executed = []
    start[1] += 5
    #goal[1] += 5
    print "*" * 80
    total = frame_data["features"][0] * 0.
    for index in xrange( len( frame_data["features"] ) ):
      print index
      if count % h == 0:
        for i, j, k, x, y, angle, value in self.grid:
          angle = momo.angle.as_vector( angle )
          state = np.array( [x, y, angle[0], angle[1]] )
          self.grid[i, j, k] = self( state, state, frame_data["frames"][index] )
        path = self.plan_count( start, goal )
        for pos in xrange( min( h, len( path ) ) ):
          executed.append( path[pos] )
        print executed

        pl.plot( 
          [self.grid.to_world( *v )[0] for v in executed[:-min( h, len( path ) ) + 1]],
          [self.grid.to_world( *v )[1] for v in executed[:-min( h, len( path ) ) + 1]],
          "y"
        )
        pl.plot( 
          [self.grid.to_world( *v )[0] for v in path],
          [self.grid.to_world( *v )[1] for v in path],
          "g"
        )
        pl.plot( 
          [v[0] for v in frame_data["states"]],
          [v[1] for v in frame_data["states"]],
          "w"
        )
        pl.plot( 
          [v[0] for v in frame_data["frames"][index]],
          [v[1] for v in frame_data["frames"][index]],
          "m."
        )
        pl.draw()

        start = self.grid.to_world( *path[min( h, len( path ) - 1 )] )
        angle = momo.angle.as_vector( start[2] )
        start = np.array( [start[0], start[1], angle[0], angle[1]] )
      count += 1
    return 0

  def plan_count( self, start, goal ):
    angle = momo.angle.as_angle( start[2:] )
    start = self.grid.from_world( start[0], start[1], angle )
    angle = momo.angle.as_angle( goal[2:] )
    goal  = self.grid.from_world( goal[0], goal[1], angle )
    print "-" * 80
    print "Plan", start, goal
    print "-" * 80
    visited   = set()
    pending   = [start]
    g = {}
    h = {}
    parent = {}
    g[start] = 0
    h[start] = momo.distance( np.array( start[:2] ), np.array( goal[:2] ) )
    parent[start] = None
    
    pl.figure( 1, figsize = ( 20, 10 ), dpi = 75 )
    pl.ion()
    pl.clf()
    pl.axis( "scaled" )
    pl.xlim( self.x, self.x + self.width )
    pl.ylim( self.y, self.y + self.height )
    open_grid = np.zeros( ( self.grid.grid_height, self.grid.grid_width ) )

    while len( pending ) > 0:
      i, j, k = pending.pop( 0 )
      if (i, j, k) == goal:
        break
      current_g = g[(i, j, k)]
      for ci, cj, ck in self.grid.neighbors( i, j, k ):
        children_g = current_g + momo.distance( np.array( [i, j] ), np.array( [ci, cj] ) ) * self.grid[ci, cj, ck]
        if (ci, cj, ck) not in g or children_g < g[(ci, cj, ck)]:
          parent[(ci, cj, ck)] = (i, j, k)
          g[(ci, cj, ck)] = children_g
          if not (ci, cj, ck) in h:
            h[(ci, cj, ck)] = momo.distance( np.array( [ci, cj] ), np.array( goal[:2] ) ) / 1E1
          if self.grid[ci, cj, ck] < 2**0.5 / 1E1:
            print "HELP", self.grid[ci, cj, ck],  h[(ci, cj, ck)]
          if not (ci, cj, ck) in visited:
            pending.append( (ci, cj, ck) )
            visited.add( (ci, cj, ck) )
            if abs( open_grid[cj, ci] ) < 0.01:
              open_grid[cj, ci] = self.grid[ci, cj, ck] + h[(ci, cj, ck)]
            else:
              open_grid[cj, ci] = min( open_grid[cj, ci], children_g + h[(ci, cj, ck)] )
      pending.sort( key = lambda v: self.grid[v[0], v[1], v[2]] + h[(v[0], v[1], v[2])] )
    
    pl.imshow( open_grid, pl.cm.jet, None, None, "none", extent = (self.x, self.x +self.width, self.y, self.y + self.height ) )
    pl.draw()
    result  = []
    current = goal
    while current != None:
      result.append( current )
      current = parent[current]
    result.reverse()
    return result

  def __call__( self, s1, s2, frame ):
    value = self.module.compute( s1, s2, frame )
    return np.dot( value, self.w )

  def save( self, stream ):
    pass
    #cPickle.dump( [self.module.__name__, self.mu, self.sigma, self.inv_sigma], stream )

  @staticmethod
  def load( stream ):
    return
    module, mu, sigma, inv_sigma = cPickle.load( stream )
    module = momo.features.__dict__[module.split( "." )[-1]]
    result = mahalanobis( module )
    result.mu = mu
    result.sigma = sigma
    result.inv_sigma = inv_sigma
    return result

