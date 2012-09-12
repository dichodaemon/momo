import momo
import numpy as np
import cPickle
from math import *

DEBUG = False

class irl_thrun( object ):
  def __init__( self, data = None, ids = None, delta = 0.25, args = None ):
    self.features = momo.irl.features.flow
    if args != None and len( args ) != 0:
      self.radius = float( args[0] )
    else:
      self.radius = 3.0

    if data != None:
      convert = momo.convert( data, delta )
      frame_data = convert.preprocess_data( data )

      if ids != None and len( ids ) > 0:
        self.ids = ids
      else:
        l = len( frame_data.keys() ) / 2
        self.ids = range( l, l + 5 )
      self.theta = momo.irl.learning.thrun.learn( 
        self.features, convert, frame_data, self.ids, radius = self.radius 
      )

  def set_convert( self, convert ):
    self.__convert = convert
    self.compute_costs = self.features.compute_costs( self.__convert, self.radius )
    self.planner = momo.planning.dijkstra()

  convert = property( None, set_convert )

  def plan( self, start, goal, velocity, frames, replan = 1 ):
    current = self.__convert.from_world2( start )
    goal = self.__convert.from_world2( goal )

    count = 0
    result = []

    while True:
      if count >= len( frames ):
        count = len( frames ) - 1

      costs = self.compute_costs( velocity, self.theta, frames[count] )
      cummulated, parents = self.planner( costs, goal )
      path = self.planner.get_path( parents, current )

      for p in path[:replan]:
        result.append( self.__convert.to_world2( p, velocity ) )
      if  len( path ) == replan + 1:
        result.append( self.__convert.to_world2( p, velocity ) )
      if len( path ) <= replan + 1:
        break

      current = path[replan]
    return result

  def feature_sum( self, states, frames ):
    result = np.array( [0.] * self.features.FEATURE_LENGTH )
    states = [self.__convert.to_world2( self.__convert.from_world2( s ), np.linalg.norm( s[2:] ) ) for s in states]
    for i in xrange( len( states ) ):
      result += self.features.compute_feature( states[i], frames[i], self.radius )
    return result

  def save( self, stream ):
    cPickle.dump( [self.radius, self.theta], stream )

  @staticmethod
  def load( stream ):
    radius, theta = cPickle.load( stream )
    result = irl_thrun()
    result.radius = radius
    result.theta = theta
    return result

