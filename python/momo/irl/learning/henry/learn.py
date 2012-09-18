import numpy as np
import pylab as pl
import momo
from compute_cummulated import *
from math import *

def learn( feature_module, convert, frame_data, ids, radius, h ):
  feature_length = feature_module.FEATURE_LENGTH
  
  compute_costs = feature_module.compute_costs( convert, radius )
  planner = momo.irl.planning.forward_backward( convert, compute_costs )
  compute_features = feature_module.compute_features( convert, radius )

  # Initialize weight vector
  w  = np.random.rand( feature_length ).astype( np.float64 )
  w /= np.linalg.norm( w )

  gamma = 1.0

  for o_id in ids:
    for times in xrange( min( h / 10, 100 ) + 1 ):
      l = len( frame_data[o_id]["states"] )
      for i in xrange( max( l - h, 1 ) ):
        gradient = compute_gradient( 
          feature_module, planner, convert, compute_features,
          frame_data[o_id]["states"][i:], frame_data[o_id]["frames"][i:],
          w, h
        )
        if np.any( np.isnan( gradient ) ):
          continue
        for i in xrange( feature_length ):
          w[i] *= exp( -gamma * gradient[i] )
        gamma *= 0.99

  return w

def compute_gradient( feature_module, planner, convert, compute_features, states, frames, w, h ):
  l = len( states )
  grid_path = [convert.from_world2( s ) for s in states]
  repr_path = [convert.to_world2( convert.from_world2( s ), np.linalg.norm( s[2:] ) ) for s in states[:h]]
  mu_observed = momo.irl.features.feature_sum( 
    feature_module, 
    repr_path,
    frames 
  )
  mu_observed /= np.sum( mu_observed[:4] )

  velocities = [np.linalg.norm( v[2:] ) for v in states]
  avg_velocity = np.sum( velocities, 0 ) / len( velocities )

  forward, backward, costs = planner( states[0], states[-1], avg_velocity, frames[0], w )
  features = compute_features( avg_velocity, frames[0] )


  accum = compute_cummulated()
  cummulated, w_features  = accum( forward, backward, costs, features, grid_path[0], h )

  mu_expected = np.sum( w_features, axis = 0 )
  mu_expected = np.sum( mu_expected, axis = 0 )
  mu_expected = np.sum( mu_expected, axis = 0 )
  mu_expected /= np.sum( mu_expected[:4] )
  gradient = mu_observed - mu_expected
  error = np.linalg.norm( gradient )

  pl.figure( 1, figsize = ( 30, 5 ), dpi = 75 )
  pl.ion()
  pl.clf()
  momo.plot.cost_plan( np.sum( cummulated, 0 ), costs, grid_path )
  pl.subplots_adjust( left = 0.01, right = 0.99 )
  pl.text( 2, 2, "error: %f" % error, color = "w" )
  pl.draw()

  return mu_observed - mu_expected

