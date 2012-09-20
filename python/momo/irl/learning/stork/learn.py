import numpy as np
import time
import momo
from momo.irl.learning.max_ent.compute_cummulated import *
from math import *

def learn( feature_module, convert, frame_data, ids, radius, h ):
  feature_length = feature_module.FEATURE_LENGTH

  compute_costs = feature_module.compute_costs( convert, radius )
  planner = momo.irl.planning.forward_backward( convert, compute_costs )
  compute_features = feature_module.compute_features( convert, radius )
  accum = compute_cummulated()

  # Initialize weight vector
  w  = ( np.ones( feature_length ) * 1.0 / feature_length ).astype( np.float64 )

  gamma = 1.0

  observed_integral = {}
  grid_paths = {}
  for o_id in ids

  for times in xrange( 200 ):

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
          momo.irl.learning.max_ent.compute_expectations( 
            states[i:], frames[i:], w, h,
            convert, compute_costs, planner, compute_features, accum
          )
        momo.tack( "Compute Expectations" )
        observed = observed_integral[o_id][i + h - 1]
        if i > 0:
          observed -= observed_integral[o_id][i - 1]
        sum_obs += observed
        sum_exp += expected


        gradient = observed / np.sum( observed[:4] ) - expected / np.sum( expected[:4] )
        if np.any( np.isnan( gradient ) ):
          continue
        error = np.linalg.norm( gradient )
        #momo.plot.gradient_descent_step( cummulated, costs, grid_paths[o_id], error )
    gradient = sum_obs / np.sum( sum_obs[:4] ) - sum_exp / np.sum( sum_exp[:4] )
    error = np.linalg.norm( gradient )
    print times, error
    if error < 0.05:
      break
    for i in xrange( feature_length ):
      w[i] *= exp( -gamma * 0.997 ** times * gradient[i] )
    w[i] /= np.linalg.norm( w )
    momo.tack( "Step" )
    print "\n".join( momo.stats( "Step" ) )

  return w


def compute_observed( feature_module, convert, states, frames ):
  momo.tick( "Compute observed" )
  momo.tick( "Discretize" )
  l = len( states )
  grid_path = [convert.from_world2( s ) for s in states]
  repr_path = [convert.to_world2( convert.from_world2( s ), np.linalg.norm( s[2:] ) ) for s in states[:h]]
  momo.tack( "Discretize" )
  momo.tick( "Compute" )
  result = np.zeros( len( states ) )
  for i in xrange( len( states ) ):
    result[i] = feature_module.compute_feature( states[i], frames[i] )
    if i > 0:
      result[i] += result[i- 1]
  momo.tack( "Compute" )
  momo.tack( "Compute observed" )
  return result, grid_path

