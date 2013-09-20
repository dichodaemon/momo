import numpy as np
import pylab as pl
import momo
import cvxopt
import cvxopt.solvers
import cvxopt
from cvxopt import solvers
from math import *

def learn( feature_module, convert, frame_data, ids, radius, h ):
  feature_length = feature_module.FEATURE_LENGTH

  # Initialize weight vector
  w  = ( np.ones( feature_length ) * 1.0 / feature_length ).astype( np.float64 )
  w /= np.linalg.norm( w )

  # Main optimization loop

  all_exp = []
  weights = []
  counts  = []
  for times in xrange( feature_length ):
    sum_obs = np.zeros( feature_length, np.float64 )
    sum_exp = np.zeros( feature_length, np.float64 )

    for o_id in ids:
      l = len( frame_data[o_id]["states"] )
      for i in xrange( max( l - h, 1 ) ):
        observed, expected, cummulated, costs, grid_path  = compute_expectations( 
          feature_module, convert, radius, 
          frame_data[o_id]["states"][i:], frame_data[o_id]["frames"][i:], 
          w, h
        )
        sum_obs += observed
        sum_exp += expected

        #gradient = observed / np.sum( observed[:4] ) - expected / np.sum( expected[:4] )
        #error = np.linalg.norm( gradient )
        #momo.plot.gradient_descent_step( cummulated, costs, grid_path, error )

    gradient = sum_obs / np.sum( sum_obs ) - sum_exp / np.sum( sum_exp )
    error = np.linalg.norm( gradient )
    print "Observed %s, Expected %s" % ( sum_obs, sum_exp )
    print "Gradient", gradient
    print times, error

    all_exp.append( sum_exp / np.sum( sum_exp ) )
    weights.append( w )
    counts.append( np.linalg.norm( sum_obs - sum_exp ) )
    w, x = optimize( times, all_exp, sum_obs )
    norm = np.linalg.norm( w )
    w = w / norm

  w = weights[np.argmin( counts )]
  print w
  return w


def compute_expectations( feature_module, convert, radius, states, frames, w, h ):
  compute_costs = feature_module.compute_costs( convert )
  planner = momo.irl.planning.dijkstra( convert, compute_costs )

  l = len( states )
  grid_path = [convert.from_world2( s ) for s in states]
  repr_path = [convert.to_world2( convert.from_world2( s ), np.linalg.norm( s[2:] ) ) for s in states[:h]]
  mu_observed = momo.features.feature_sum( 
    feature_module, 
    repr_path[:h],
    frames[:h],
    radius
  )

  velocities = [np.linalg.norm( v[2:] ) for v in states[:h]]
  avg_velocity = np.sum( velocities, 0 ) / len( velocities )

  compute_features = feature_module.compute_features( convert, radius )
  features = compute_features( avg_velocity, frames[0] )

  
  w = sum( w ) - w

  path, cummulated, costs  = planner( states[0], states[-1], features, w, avg_velocity )
  mu_expected = np.sum( 
    [
      feature_module.compute_feature( 
        path[i], frames[min( i, len( frames ) )], radius 
      ) 
      for i in xrange( len( path[:h] ) )
    ], 
    0 
  )
  return mu_observed, mu_expected, cummulated, costs, grid_path


def optimize(  j, all_exp, mu_observed ):
  w_len = mu_observed.shape[0]
  n = w_len + j + 1
  p = cvxopt.matrix( np.zeros( ( n, n ) ), tc = "d" )
  q = cvxopt.matrix( np.zeros( n ), tc = "d" )
  for i in xrange( w_len ):
    p[i, i] = 1.0
  a = cvxopt.matrix( np.zeros( ( 1, n ) ), tc = "d" )
  for i in xrange( w_len, n ):
    a[0, i] = 1
  b = cvxopt.matrix( np.ones( 1 ), tc = "d" )
  g = cvxopt.matrix( np.zeros( ( n + w_len, n ) ), tc = "d" )
  for i in xrange( n ):
    g[i, i] = 1
  for i in xrange( w_len ):
    g[n + i, i] = 1
    for tj in xrange( j + 1 ):
      g[n + i, w_len + tj] = float( all_exp[tj][i] ) #Why does this not work with negative?
  h = cvxopt.matrix( np.zeros( n + w_len ), tc = "d" )
  for i in xrange( w_len ):
    h[n + i] = -mu_observed[i]
  solvers.options["maxiters"] = 20
  solvers.options["show_progress"] = False
  result = solvers.qp( p, q, - g, -h, a, b, "glpk" )
  r_w = np.zeros( w_len )
  for i in xrange( w_len ):
    r_w[i] = result["x"][i]
  r_x = np.zeros( j + 1 )
  for i in xrange( j + 1 ):
    r_x[i] = result["x"][w_len + i]
  return r_w, r_x

