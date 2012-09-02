import momo
import cvxopt
from cvxopt import solvers
import numpy as np
import random
import cPickle
from math import *
from irl_functions import preprocess_data
import pylab as pl

class irl_henry( object ):
  def __init__( self, module, data = None ):
    self.module = module

    if data != None:
      # Obtain parameters from data
      self.x, self.y, self.width, self.height, self.delta, self.grid, frame_data = preprocess_data( module, data )

      ids = random.sample( frame_data.keys(), min( 3, len( frame_data.keys() ) ) )

  def plan_features( self, frame_data, h = 6 ):
    count = 0
    start = frame_data["states"][0] * 1
    goal  = frame_data["states"][-1] * 1
    executed = []
    total = frame_data["features"][0] * 0.
    for index in xrange( len( frame_data["features"] ) ):
      if count % h == 0:
        for i, j, k, x, y, angle, value in self.grid:
          angle = momo.angle.as_vector( angle )
          state = np.array( [x, y, angle[0], angle[1]] )
          self.grid[i, j, k] = self( state, state, frame_data["frames"][index] )

        scale = 1.
        while True:
          path = self.plan_count( start, goal, scale )
          if path == None:
            scale *= 0.1
          else:
            break

        for pos in xrange( min( h, len( path ) ) ):
          o = path[pos]
          executed.append( o )
          o = self.grid.to_world( *o )
          angle = momo.angle.as_vector( o[2] )
          o = np.array( [o[0], o[1], angle[0], angle[1]] )
          feature = self.module.compute( o, o, frame_data["frames"][index] )
          total += feature

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
    return total

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
            h[(ci, cj, ck)] = momo.distance( np.array( [ci, cj] ), np.array( goal[:2] ) ) * scaling
          if self.grid[ci, cj, ck] < 2**0.5 * scaling:
            print "Rescaling", scaling
            return None
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

