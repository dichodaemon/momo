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

  gamma = 0.999

  for times in xrange( 100 ):

    momo.tick( "Step" )
    sum_obs = np.zeros( feature_length, np.float64 )
    sum_exp = np.zeros( feature_length, np.float64 )

    for o_id in ids:
      l = len( frame_data[o_id]["states"] )
      for i in xrange( max( l - h, 1 ) ):
        momo.tick( "Compute Expectations" )
        observed, expected, cummulated, costs, grid_path =\
          momo.irl.learning.max_ent.compute_expectations( 
            feature_module, convert, radius,
            frame_data[o_id]["states"][i:], frame_data[o_id]["frames"][i:],
            w, h,
            compute_costs, planner, compute_features, accum
          )
        momo.tack( "Compute Expectations" )
        sum_obs += observed
        sum_exp += expected

        gradient = observed / np.sum( observed[:4] ) - expected / np.sum( expected[:4] )
        if np.any( np.isnan( gradient ) ):
          continue
        #error = np.linalg.norm( gradient )
        #momo.plot.gradient_descent_step( cummulated, costs, grid_path, error )
    gradient = sum_obs / np.sum( sum_obs[:4] ) - sum_exp / np.sum( sum_exp[:4] )
    error = np.linalg.norm( gradient )
    print times, error
    if error < 0.05:
      break
    for i in xrange( feature_length ):
      w[i] *= exp( -gamma ** times * gradient[i] )
    momo.tack( "Step" )
    print "\n".join( momo.stats( "Step" ) )

  return w


