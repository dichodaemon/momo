import numpy as np
import pylab as pl
import momo
from math import *

def learn( feature_module, convert, frame_data, ids, radius, replan ):
  feature_length = feature_module.FEATURE_LENGTH
  
  compute_costs = feature_module.compute_costs( convert, radius )
  planner = momo.irl.planning.forward_backward( convert, compute_costs )
  compute_features = feature_module.compute_features( convert, radius )

  # Initialize weight vector
  w  = np.random.rand( feature_length )

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
      gamma *= 0.999
      magnitude = np.linalg.norm( gradient )
      print "gradient", magnitude
      print gamma
      if magnitude < 0.05:
        break

  return w

def compute_gradient( feature_module, planner, convert, compute_features, states, frames, w, replan ):
  mu_observed = momo.irl.features.feature_sum( 
    feature_module, 
    [convert.to_world2( convert.from_world2( s ), np.linalg.norm( s[2:] ) ) for s in states], 
    frames 
  )
  mu_observed /= np.sum( mu_observed[:4] )

  velocities = [np.linalg.norm( v[2:] ) for v in states]
  avg_velocity = np.sum( velocities, 0 ) / len( velocities )

  pl.figure( 0 )
  pl.ion()
  forward, backward, costs = planner( states[0], states[-1], avg_velocity, frames[0], w  * 2 )
  features = compute_features( avg_velocity, frames[0] )

  mu_expected = np.zeros( feature_module.FEATURE_LENGTH )
  tmp = forward * 0.

  for d in xrange( forward.shape[0] ):
    for y in xrange( forward.shape[1] ):
      for x in xrange( forward.shape[2] ):
        #tmp[d, y, x] = forward[d, y, x] * exp( -np.dot( w, features[d, y, x] ) ) * backward[d, y, x] 
        tmp[d, y, x] = forward[d, y, x] * exp( -costs[d, y, x] ) * backward[d, y, x] 
  for d in xrange( forward.shape[0] ):
    for y in xrange( forward.shape[1] ):
      for x in xrange( forward.shape[2] ):
        mu_expected += features[d, y, x] * tmp[d, y, x]
  mu_expected /= np.sum( mu_expected[:4] )

  pl.figure( 1 )
  pl.ion()
  pl.clf()
  pl.imshow( np.sum( tmp, 0 ) )
  pl.draw()
  np.set_printoptions( precision = 3 )
  np.set_printoptions( suppress = True )
  print "mu_observed", mu_observed
  print "mu_expected", mu_expected


  return mu_observed - mu_expected

