import numpy as np
import time
import momo
from momo.learning.max_ent.compute_cummulated import *
from math import *

def learn( feature_module, convert, frame_data, ids, radius, h ):
  feature_length = feature_module.FEATURE_LENGTH

  compute_costs = feature_module.compute_costs( convert )
  planner = momo.irl.planning.forward_backward( convert, compute_costs )
  compute_features = feature_module.compute_features( convert, radius )
  accum = compute_cummulated()

  observed_integral = {}
  grid_paths = {}
  for o_id in ids:
    states = frame_data[o_id]["states"]
    frames = frame_data[o_id]["frames"]
    obs, path = compute_observed( feature_module, convert, states, frames, radius )
    observed_integral[o_id] = obs
    grid_paths[o_id] = path

  # Initialize weight vector
  w  = ( np.ones( feature_length ) * 1.0 / feature_length ).astype( np.float64 )

  gamma = 0.5
  decay = 0.99
  min_w = None
  min_e = 1E6

  print "Gamma", gamma
  print "Decay", decay

  for times in xrange( 100 ):

    momo.tick( "Step" )
    sum_obs = np.zeros( feature_length, np.float64 )
    sum_exp = np.zeros( feature_length, np.float64 )

    for o_id in ids:
      states = frame_data[o_id]["states"]
      frames = frame_data[o_id]["frames"]
      l = len( states )
      for i in xrange( max( l - h, 1 ) ):
        momo.tick( "Compute Expectations" )
        expected, cummulated, costs =\
          momo.learning.max_ent.compute_expectations( 
            states[i:], frames[i:], w, h,
            convert, compute_costs, planner, compute_features, accum
          )
        momo.tack( "Compute Expectations" )
        observed = observed_integral[o_id][min( i + h, l - 1 )] * 1
        if i > 0:
          observed -= observed_integral[o_id][i - 1]
        sum_obs += observed
        sum_exp += expected

        if np.any( np.isnan( expected ) ):
          continue
        if np.sum( observed ) != 0 and np.sum( expected ) != 0:
          gradient = observed / np.sum( observed ) - expected / np.sum( expected )
        else:
          gradient = observed * 0.
        error = np.linalg.norm( gradient )
        #momo.plot.gradient_descent_step( cummulated, costs, grid_paths[o_id], error )
    if np.sum( sum_obs ) != 0 and np.sum( sum_exp ) != 0:
      gradient = sum_obs / np.sum( sum_obs ) - sum_exp / np.sum( sum_exp )
    error = np.linalg.norm( gradient )
    if error < min_e:
      min_e = error
      min_w = w
    print sum_obs, sum_exp
    print sum_obs / np.sum( sum_obs ), sum_exp / np.sum( sum_exp )
    print times, error
    if error < 0.05:
      break
    for i in xrange( feature_length ):
      w[i] *= exp( -gamma * decay ** times * gradient[i] )
      #w[i] *= exp( -gamma * gradient[i] )
    w /= np.sum( w )
    print "w", w
    momo.tack( "Step" )
    print "\n".join( momo.stats( "Step" ) )

  print min_e
  print "W", min_w
  return min_w


def compute_observed( feature_module, convert, states, frames, radius ):
  momo.tick( "Compute observed" )
  momo.tick( "Discretize" )
  l = len( states )
  grid_path = [convert.from_world2( s ) for s in states]
  repr_path = [convert.to_world2( convert.from_world2( s ), np.linalg.norm( s[2:] ) ) for s in states]
  momo.tack( "Discretize" )
  momo.tick( "Compute" )
  result = []
  for i in xrange( len( states ) ):
    result.append( feature_module.compute_feature( states[i], frames[i], radius ) )
    if i > 0:
      result[i] += result[i- 1]
  momo.tack( "Compute" )
  momo.tack( "Compute observed" )
  return result, grid_path

