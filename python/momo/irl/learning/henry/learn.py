import numpy as np
import pylab as pl
import momo
from compute_cummulated import *
from math import *

def learn( feature_module, convert, frame_data, ids, radius, replan ):
  feature_length = feature_module.FEATURE_LENGTH
  
  compute_costs = feature_module.compute_costs( convert, radius )
  planner = momo.irl.planning.forward_backward( convert, compute_costs )
  compute_features = feature_module.compute_features( convert, radius )

  # Initialize weight vector
  w  = np.random.rand( feature_length ).astype( np.float64 )

  gamma = 1.0
  for o_id in ids:
    w /= np.linalg.norm( w )
    for i in xrange( 100 ):
      gradient = compute_gradient( 
        feature_module, planner, convert, compute_features,
        frame_data[o_id]["states"], frame_data[o_id]["frames"],
        w, replan
      )
      if np.any( np.isnan( gradient ) ):
        print "w, nan", w
        break
      for i in xrange( feature_length ):
        w[i] *= exp( -gamma * gradient[i] )
      print w.dtype
      gamma *= 0.999
      magnitude = np.linalg.norm( gradient )
      print "gradient", magnitude
      print "gamma", gamma
      if magnitude < 0.1:
        break

  return w

def compute_gradient( feature_module, planner, convert, compute_features, states, frames, w, replan ):
  grid_path = [convert.from_world2( s ) for s in states]
  repr_path = [convert.to_world2( convert.from_world2( s ), np.linalg.norm( s[2:] ) ) for s in states]
  mu_observed = momo.irl.features.feature_sum( 
    feature_module, 
    repr_path,
    frames 
  )
  mu_observed /= np.sum( mu_observed[:4] )

  velocities = [np.linalg.norm( v[2:] ) for v in states]
  avg_velocity = np.sum( velocities, 0 ) / len( velocities )

  pl.figure( 0 )
  pl.ion()
  forward, backward, costs = planner( states[0], states[-1], avg_velocity, frames[0], w )
  features = compute_features( avg_velocity, frames[0] )


  accum = compute_cummulated()
  cummulated, w_features  = accum( forward, backward, costs, features )

  mu_expected = np.sum( w_features, axis = 0 )
  mu_expected = np.sum( mu_expected, axis = 0 )
  mu_expected = np.sum( mu_expected, axis = 0 )
  mu_expected /= np.sum( mu_expected[:4] )

  pl.figure( 1 )
  pl.ion()
  pl.clf()
  pl.axis( "scaled" )
  pl.xlim( 0, forward.shape[2] - 1 )
  pl.ylim( 0, forward.shape[1] - 1 )
  pl.imshow( np.sum( cummulated, 0 ) )
  pl.plot( [v[0] for v in grid_path], [v[1] for v in grid_path], "y." )
  pl.draw()
  np.set_printoptions( precision = 3 )
  np.set_printoptions( suppress = True )
  print "mu_observed", mu_observed
  print "mu_expected", mu_expected
  print "theta", w


  return mu_observed - mu_expected

