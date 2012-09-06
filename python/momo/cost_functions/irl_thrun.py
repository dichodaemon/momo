import momo
import cvxopt
from cvxopt import solvers
import numpy as np
import random
import cPickle
from math import *
from irl_functions import preprocess_data
import pylab as pl

DEBUG = False

class irl_thrun( object ):
  def __init__( self, module, data = None ):
    self.module = module

    if data != None:
      ids = set( [46] )
      # Obtain parameters from data
      self.x, self.y, self.width, self.height, self.delta, self.grid, frame_data = preprocess_data( module, data, ids )

      # Initialize weight vector
      self.w  = np.random.rand( frame_data.values()[0]["feature_sum"].shape[0] ) 
      self.w /= self.w
      self.w /= np.linalg.norm( self.w )

      # Compute observed feature sum for selected samples
      #ids = random.sample( frame_data.keys(), min( 6, len( frame_data.keys() ) ) )[:3]
      feature_sum = self.w * 0.
      for o_id in ids:
        feature_sum += frame_data[o_id]["feature_sum"]

      # Optimize weight vector
      mu_p = []
      weights = []
      j = 0
      while True:
        print "-" * 80
        temp_sum = self.w * 0.
        
        for o_id in ids:
          temp_sum += self.plan_features( frame_data[o_id] )
        mu_p.append( temp_sum )
        print "exhibited ", feature_sum
        print "planned   ", temp_sum
        print "difference", feature_sum - temp_sum
        print "total diff", sum( abs( feature_sum -temp_sum ) )
        w, x = self.optimize( j, self.w, mu_p, feature_sum )
        norm = np.linalg.norm( w )
        print "x", x
        print "norm", norm
        diff = self.w - w / np.linalg.norm( w )
        self.w = w / norm
        weights.append( self.w )
        if np.linalg.norm( diff ) < 1E-3:
          print "=" * 80
          print "Finished"
          print "=" * 80
          self.w = weights[np.argmax( x )]
          for o_id in ids:
            self.plan_features( frame_data[o_id] )
          break
        j += 1

  def optimize( self, j, w, mu_p, mu_e ):
    n = len( w ) + j + 1
    p = cvxopt.matrix( np.zeros( ( n, n ) ) )
    q = cvxopt.matrix( np.zeros( n ) )
    for i in xrange( len( w ) ):
      p[i, i] = 1.0
    a = cvxopt.matrix( np.zeros( ( 1, n ) ) )
    for i in xrange( len( w ), n ):
      a[0, i] = 1
    b = cvxopt.matrix( np.ones( 1 ) )
    g = cvxopt.matrix( np.zeros( ( n + len( w ), n ) ) )
    for i in xrange( n ):
      g[i, i] = 1
    for i in xrange( len( w ) ):
      g[n + i, i] = 1
      for j in xrange( j + 1 ):
        g[n + i, len( w ) + j] = -mu_p[j][i]
    h = cvxopt.matrix( np.zeros( n + len( w ) ) )
    for i in xrange( len( w ) ):
      h[n + i] = mu_e[i]
    solvers.options["maxiters"] = 20
    solvers.options["show_progress"] = False
    result = solvers.qp( p, q, - g, h, a, b, "glpk" )
    r_w = w * 0.
    for i in xrange( len( w ) ):
      r_w[i] = result["x"][i]
    r_x = np.zeros( j + 1 )
    for i in xrange( j + 1 ):
      r_x[i] = result["x"][len( w ) + i]
    return r_w, r_x

  def plan_features( self, frame_data, h = 1 ):
    count = 0
    start = frame_data["states"][0] * 1
    goal  = frame_data["states"][-1] * 1
    executed = []
    total = frame_data["features"][0] * 0.
    for index in xrange( len( frame_data["features"] ) ):
      if count % h == 0:
        self.current_frame = frame_data["frames"][index]
        for i, j, k, x, y, angle, value in self.grid:
          angle = momo.angle.as_vector( angle ) * np.linalg.norm( frame_data["states"][index] )
          state = np.array( [x, y, angle[0], angle[1]] )
          self.grid[i, j, k] = self( state, state, frame_data["frames"][index] )

        scale = 10.
        while True:
          path = self.plan_count( start, goal, scale )
          if path == None:
            scale *= 0.9
          else:
            break

        for pos in xrange( min( h, len( path ) ) ):
          o = path[pos]
          executed.append( o )
          o = self.grid.to_world( *o )
          angle = momo.angle.as_vector( o[2] )
          o = np.array( [o[0], o[1], angle[0], angle[1]] )
          feature = self.module.compute( o, o, frame_data["frames"][index + pos] )
          total += feature

        if DEBUG:
          pl.ioff()
        if h != 1:
          pl.plot( 
            [self.grid.to_world( *v )[0] for v in executed[:-min( h, len( path ) ) + 1]],
            [self.grid.to_world( *v )[1] for v in executed[:-min( h, len( path ) ) + 1]],
            "y"
          )
        else:
          pl.plot( 
            [self.grid.to_world( *v )[0] for v in executed[:]],
            [self.grid.to_world( *v )[1] for v in executed[:]],
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
        if DEBUG:
          pl.show()
        else:
          pl.draw()

        start = self.grid.to_world( *path[min( h, len( path ) - 1 )] )
        angle = momo.angle.as_vector( start[2] )
        start = np.array( [start[0], start[1], angle[0], angle[1]] )
        if len( path ) == 1:
          break
      count += 1
      print "Length observed", len( frame_data["frames"] )
      print "Length executed", len( executed )
      print "Length plan", len( path )
      print "Sum", sum( total )
    return total

  def draw_features( self, event ):
    print "*" * 80
    print event.xdata, event.ydata
    print "*" * 80
    circle = pl.Circle( (event.xdata, event.ydata), radius = 1 )
    circle.x = event.xdata
    circle.y = event.ydata
    state    = np.array( [event.xdata, event.ydata, 0.04, 0.] )
    cost     = self( state , state, self.current_frame )
    feature  = self.module.compute( state , state, self.current_frame )
    print feature
    density  = -1
    if feature[0] == 1:
      density = 3
    elif feature[1] == 1:
      density = 2
    elif feature[2] == 1:
      density = 1
    elif feature[3] == 1:
      density = 0
    angle = -1
    if max( feature[4:7] ) == 1:
      angle = 180
    elif max( feature[7:10] ) == 1:
      angle = 110
    elif max( feature[10:13] ) == 1:
      angle = 66
    elif max( feature[13:16] ) == 1:
      angle = 0
    velocity = -1
    if feature[4] + feature[7] + feature[10] + feature[13] == 1:
      velocity = 1
    elif feature[5] + feature[8] + feature[11] + feature[14] == 1:
      velocity = 0.5
    elif feature[6] + feature[9] + feature[12] + feature[15] == 1:
      velocity = 0.0
    pl.text( state[0], state[1], "c:%f, d:%i, a:%f, v:%f" % ( cost, density, angle, velocity ) )
    pl.gca().add_patch( circle )
    pl.draw()

  def plan_count( self, start, goal, scaling ):
    angle = momo.angle.as_angle( start[2:] )
    start = self.grid.from_world( start[0], start[1], angle )
    angle = momo.angle.as_angle( goal[2:] )
    goal  = self.grid.from_world( goal[0], goal[1], angle )
    visited   = set()
    pending   = [start]
    g = {}
    h = {}
    parent = {}
    g[start] = 0
    h[start] = momo.distance( np.array( start[:2] ), np.array( goal[:2] ) )
    parent[start] = None
    
    fig = pl.figure( 1, figsize = ( 20, 10 ), dpi = 75 )
    fig.canvas.mpl_connect( "button_press_event",  self.draw_features )
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
            h[(ci, cj, ck)] = momo.distance( np.array( [ci, cj] ), np.array( goal[:2] ) ) * scaling
          if self.grid[ci, cj, ck] < scaling:
            print self.grid[ci, cj, ck], scaling
            return None
          if not (ci, cj, ck) in visited:
            pending.append( (ci, cj, ck) )
            visited.add( (ci, cj, ck) )
            if abs( open_grid[cj, ci] ) < 1E-10:
              #open_grid[cj, ci] = children_g + h[(ci, cj, ck)]
              open_grid[cj, ci] = self.grid[ci, cj, 4]
            else:
              #open_grid[cj, ci] = min( open_grid[cj, ci], children_g + h[(ci, cj, ck)] )
              open_grid[cj, ci] = min( open_grid[cj, ci], self.grid[ci, cj, 4] )
      pending.sort( key = lambda v: g[(v[0], v[1], v[2])] + h[(v[0], v[1], v[2])] )
      for v in pending:
        print g[(v[0], v[1], v[2])] + h[(v[0], v[1], v[2])]
    
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
    cPickle.dump( [self.module.__name__, self.x, self.y, self.width, self.height, self.delta, self.w], stream )

  @staticmethod
  def load( stream ):
    module, x, y, width, height, delta, w = cPickle.load( stream )
    module = momo.features.__dict__[module.split( "." )[-1]]
    result = irl_thrun( module )
    result.x = x
    result.y = y
    result.width = width
    result.height = height
    result.grid = momo.planning.grid( x, y, width, height, delta )
    result.w = w
    return result

